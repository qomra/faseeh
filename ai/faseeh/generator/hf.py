
import tqdm
import torch
import logging
from datasets import Dataset
from transformers import pipeline
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer

class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation"""
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def formatting_prompts_func(conv):
    """
    Format conversations using Llama3 template format.
    
    Args:
        example: Dictionary containing conversation data
        
    Returns:
        List of formatted conversation strings
    """
    system_prompt = "أنت مساعد مفيد ومحترم. تجيب دائماً بشكل مباشر ودقيق."  # Customize this    
    
    # Format each conversation turn using Llama3 template
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
        # # copy the dataset
        # dataset = dataset.map(lambda x: x)
        # # remove the last message from each conversation
        # dataset = dataset.map(lambda x: {"conversation": x["conversation"][:-1]})
        # # run the pipeline on the dataset
        # dataset_conversations = dataset["conversation"]
        # Format all prompts
        formatted_prompts = []
        for conv in dataset["conversation"]:
            # Format prompt but exclude the last message if it exists
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
        
    def generate(self, dataset: Dataset,max_new_tokens=200):
        logging.info("Generating completions...")
        results = []
        # get first 100 characters of each dataitem[content]
        if "content" in dataset.column_names:
            content = dataset["content"]
            content = Dataset.from_dict({
                "content": [c[:100] for c in content]
            })["content"]
        else:
            content = dataset["prompt"]
        
        
        # get first 100 items
        content = content[:100]
    
        results = self.model(
            content,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            batch_size=128)
        # zip input and output
        if "completion" in dataset.column_names:
            # zip with completion
            results = list(zip(content, [r[0]["generated_text"] for r in results], dataset["completion"]))
        else:
            results = list(zip(content, [r[0]["generated_text"] for r in results]))
        
        return results