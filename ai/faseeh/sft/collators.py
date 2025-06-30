import torch
import logging
from typing import Dict, Any, List
from transformers import DataCollatorForLanguageModeling

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def formatting_prompts_func(example: Dict[str, Any]) -> List[str]:
    """Format conversations using chat template format for AlAghani dataset."""
    output_texts = []
    
    for conversation in example["conversation"]:
        # Extract the three messages: system, user, assistant
        system_msg = conversation[0]['content'].strip()
        user_msg = conversation[1]['content'].strip()  
        assistant_msg = conversation[2]['content'].strip()
        
        # Use the correct format that matches your FaseehTokenizer4
        formatted_text = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>"
            f"{system_msg}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>"
            f"{user_msg}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
            f"{assistant_msg}"
            "<|eot_id|>"
        )
        output_texts.append(formatted_text)
    return output_texts

def get_template_func(tokenizer):
    """Get the instruction template tokens for the data collator."""
    instruction_template = "<|start_header_id|>user<|end_header_id|>"
    return tokenizer.encode(instruction_template, add_special_tokens=False)

def get_response_template_func(tokenizer):
    """Get the response template tokens for the data collator."""
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return tokenizer.encode(response_template, add_special_tokens=False)

class FaseehDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator that masks all tokens except the assistant's response.
    Optimized for FaseehTokenizer4.
    """
    
    def __init__(self, tokenizer, instruction_template=None, response_template=None, mlm=False, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        
        # Use provided templates or generate them
        if response_template is None:
            response_template = get_response_template_func(tokenizer)
        if instruction_template is None:
            instruction_template = get_template_func(tokenizer)
            
        self.instruction_template = instruction_template
        self.response_template = response_template
        self.tokenizer = tokenizer
        
        logging.info(f"Instruction template tokens: {instruction_template}")
        logging.info(f"Response template tokens: {response_template}")
        
        # Debug info
        try:
            inst_decoded = tokenizer.decode(instruction_template, skip_special_tokens=False)
            resp_decoded = tokenizer.decode(response_template, skip_special_tokens=False)
            logging.info(f"Instruction template decoded: '{inst_decoded}'")
            logging.info(f"Response template decoded: '{resp_decoded}'")
        except Exception as e:
            logging.warning(f"Could not decode templates: {e}")

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        found_any = False
        
        for i in range(labels.shape[0]):
            response_start = self._find_response_start(batch["input_ids"][i])
            
            if response_start != -1:
                labels[i, :response_start] = -100
                found_any = True
                logging.debug(f"Found response start at position {response_start} in example {i}")
            else:
                labels[i, :] = -100
                logging.warning(f"Could not find response template in example {i}")
                # Debug: show the problematic sequence
                self._debug_sequence(batch["input_ids"][i])
        
        # if found_any:
        #     logging.info("Successfully found response templates in at least one example")
        # else:
        #     logging.error("Could not find response templates in ANY examples!")
            
        batch["labels"] = labels
        return batch
    
    def _find_response_start(self, input_ids):
        """
        Find where the assistant response starts using multiple strategies
        """
        input_ids_list = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
        
        # Strategy 1: Exact template match
        response_start = self._find_exact_template_match(input_ids_list)
        if response_start != -1:
            logging.debug(f"Found exact template match at position {response_start}")
            return response_start
        
        # Strategy 2: Look for assistant token specifically  
        response_start = self._find_assistant_token_match(input_ids_list)
        if response_start != -1:
            logging.debug(f"Found assistant token match at position {response_start}")
            return response_start
            
        # Strategy 3: Fuzzy matching in decoded text
        response_start = self._find_fuzzy_match(input_ids_list)
        if response_start != -1:
            logging.debug(f"Found fuzzy match at position {response_start}")
            return response_start
            
        return -1
    
    def _find_exact_template_match(self, input_ids_list):
        """Look for exact response template token sequence"""
        template_len = len(self.response_template)
        
        for i in range(len(input_ids_list) - template_len + 1):
            if input_ids_list[i:i + template_len] == self.response_template:
                return i + template_len
        return -1
    
    def _find_assistant_token_match(self, input_ids_list):
        """
        Look specifically for the assistant token (ID 6 based on your output)
        """
        # From your test output, assistant token appears to be ID 6
        assistant_token_id = 6  # Based on your tokenizer output
        
        for i, token_id in enumerate(input_ids_list):
            if token_id == assistant_token_id:
                # Found assistant token, look for the end marker
                # Should be followed by header_end token (ID 3)
                if i + 1 < len(input_ids_list) and input_ids_list[i + 1] == 3:
                    return i + 2  # Start after <|header_end|>
        return -1
    
    def _find_fuzzy_match(self, input_ids_list):
        """
        Try to find assistant response using decoded text patterns
        """
        try:
            decoded_text = self.tokenizer.decode(input_ids_list, skip_special_tokens=False)
            
            # Look for assistant patterns in the decoded text
            assistant_patterns = [
                "<|header_start|>assistant<|header_end|>",
                "assistant<|header_end|>",
                ">assistant<"
            ]
            
            for pattern in assistant_patterns:
                pattern_pos = decoded_text.find(pattern)
                if pattern_pos != -1:
                    # Found the pattern, now estimate token position
                    end_pos = pattern_pos + len(pattern)
                    
                    # Get text up to the end of the pattern
                    prefix_text = decoded_text[:end_pos]
                    
                    # Encode to get approximate token count
                    try:
                        prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                        return min(len(prefix_tokens), len(input_ids_list))
                    except:
                        # If encoding fails, estimate based on character position
                        estimated_pos = int(end_pos / len(decoded_text) * len(input_ids_list))
                        return min(estimated_pos, len(input_ids_list))
                        
        except Exception as e:
            logging.debug(f"Fuzzy matching failed: {e}")
            
        return -1
    
    def _debug_sequence(self, input_ids):
        """Debug helper to show what's in a problematic sequence"""
        try:
            decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            logging.debug(f"Problematic sequence: {decoded[:300]}...")
            
            # Show first 20 tokens with their IDs
            tokens_debug = []
            for i, token_id in enumerate(input_ids.tolist()[:20]):
                try:
                    token_text = self.tokenizer.decode([token_id])
                    tokens_debug.append(f"{i}:{token_id}='{token_text}'")
                except:
                    tokens_debug.append(f"{i}:{token_id}=<decode_error>")
            
            logging.debug(f"First 20 tokens: {' '.join(tokens_debug)}")
            
        except Exception as e:
            logging.debug(f"Could not debug sequence: {e}")

