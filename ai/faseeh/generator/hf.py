import tqdm
import torch
import logging
from datasets import Dataset
from transformers import pipeline
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def formatting_prompts_func(conv):
    system_prompt = "أنت مساعد مفيد ومحترم. تجيب دائماً بشكل مباشر ودقيق."
    formatted_text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>"
        f"{system_prompt}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>"
        f"{conv[1]['content'].strip()}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )
    return formatted_text

class HuggingFaceWrapper:
    def __init__(self, model_name: str, tokenizer=None,lora_adapter=None):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        # add padding token
        self.model = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            tokenizer=self.tokenizer
        )
        if lora_adapter is not None:
            self.model.model.load_adapter(lora_adapter)
        self.model.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # set padding side to left
        self.model.tokenizer.padding_side = "left"
        self.stop_token_ids = [
            self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0],
            self.tokenizer.encode("<|start_of_header|>", add_special_tokens=False)[0]
        ]
        self.stopping_criteria = StoppingCriteriaList([
            StopOnTokens(self.stop_token_ids)
        ])


    def generate(self, dataset: Dataset, 
                 max_new_tokens=200, 
                 temprature=0.7,
                 top_k=30,
                 top_p=0.9):
        logging.info("Generating completions...")
        results = []
        formatted_prompts = []
        for conv in dataset["conversation"]:
            prompt = formatting_prompts_func(conv[:-1] if len(conv) > 1 else conv)
            formatted_prompts.append(prompt)

        results = self.model(
            formatted_prompts,
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False,
            batch_size=128,
            max_new_tokens=max_new_tokens, 
            temperature=temprature,
            top_k=top_k,
            top_p=top_p,
            stopping_criteria=self.stopping_criteria)

        return results

class HuggingFacePretrainedCompletionWrapper(HuggingFaceWrapper):
    def __init__(self, model_name: str, tokenizer=None):
        super().__init__(model_name, tokenizer)
        
    def generate(self, dataset: Dataset, max_new_tokens=200, **kwargs): # Add **kwargs to capture generation params
        logging.info("Generating completions for pre-trained model...")
        
        
        # shuffle the dataset to ensure diverse prompts
        dataset = dataset.shuffle(seed=42)  # Shuffle for diversity in prompts
        
        full_original_contents = []
        if "content" in dataset.column_names:
            full_original_contents = dataset["content"]
        else:
            logging.error("Dataset for HuggingFacePretrainedCompletionWrapper must have a 'content' column.")
            return []
        
        

        # Define the length of the prompt in TOKENS (not characters)
        prompt_length_tokens = 100 # Default prompt percentage length in tokens
        
        prompts_for_generation = []
        ground_truths = []
        
        # We will iterate through a sample of the dataset (e.g., first 100 items)
        for original_full_text in full_original_contents:
            if not original_full_text:
                continue # Skip empty texts

            # 1. Tokenize the original full text
            # Ensure add_special_tokens=True/False is appropriate for the context.
            # For prompts to a base model, usually False to avoid adding BOS mid-text.
            full_token_ids = self.tokenizer.encode(original_full_text, add_special_tokens=False) 
            input_token_length = prompt_length_tokens 
            if len(full_token_ids) < prompt_length_tokens:
                # take 30% of the full text if it's shorter than the prompt length
                input_token_length = int(len(full_token_ids) * 0.3)
            
            # 2. Slice the token IDs for prompt and ground truth
            prompt_token_ids = full_token_ids[:input_token_length]
            ground_truth_token_ids = full_token_ids[input_token_length : input_token_length + max_new_tokens]
            
            # 3. Decode token IDs back to strings for output
            prompt_text = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True) # Skip special tokens on decode
            ground_truth_text = self.tokenizer.decode(ground_truth_token_ids, skip_special_tokens=True) # Skip special tokens on decode
            
            prompts_for_generation.append(prompt_text)
            ground_truths.append(ground_truth_text)
            
        # Generate completions using the prepared prompts
        if not prompts_for_generation:
            logging.warning("No valid prompts generated for text generation.")
            return []

        # --- FIX: Pass crucial generation parameters to the pipeline call ---
        generation_params = {
            "max_new_tokens": max_new_tokens,
            "do_sample": kwargs.get("do_sample", True), # Enable sampling
            "temperature": kwargs.get("temperature", 0.7), # Default if not provided
            "top_k": kwargs.get("top_k", 50), # Default if not provided
            "top_p": kwargs.get("top_p", 0.95), # Default if not provided
            "no_repeat_ngram_size": kwargs.get("no_repeat_ngram_size", 3), # Prevent repeating 3-token n-grams
            "repetition_penalty": kwargs.get("repetition_penalty", 1.2), # Penalize repetition
            "return_full_text": False, # Ensure only generated part is returned
            "batch_size": 128 # Can be passed to pipeline for inference batching
        }

        generated_results = self.model(
            prompts_for_generation,
            **generation_params # Pass all the collected generation parameters
        )
        # --- END FIX ---
        
        completions = [r[0]["generated_text"] for r in generated_results]
        
        results_with_gt = list(zip(prompts_for_generation, completions, ground_truths))
        
        return results_with_gt