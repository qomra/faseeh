
import tqdm
import torch
import logging
from datasets import Dataset
from transformers import pipeline

from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(conversation: list):
    prompt = ""
    for i, message in enumerate(conversation[:-1]):
        prompt += f"<|start_of_header|>{message['role']}<|end_of_header|>{message['content']}<|eot_id|>"
    prompt += f"<|start_of_header|>{conversation[-1]['role']}<|end_of_header|>"
    return prompt

class HuggingFaceWrapper:
    def __init__(self, model_name: str, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        # add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            tokenizer=self.tokenizer
        )
        self.model.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # set padding side to left
        self.model.tokenizer.padding_side = "left"


    def generate(self, dataset: Dataset, 
                 max_new_tokens=200, 
                 temprature=0.7,
                 top_k=50,
                 top_p=0.9):
        logging.info("Generating completions...")
        results = []
        # copy the dataset
        dataset = dataset.map(lambda x: x)
        # remove the last message from each conversation
        dataset = dataset.map(lambda x: {"conversation": x["conversation"][:-1]})
        # run the pipeline on the dataset
        dataset_conversations = dataset["conversation"]

        results = self.model(
            dataset_conversations,
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False,
            batch_size=128,
            max_new_tokens=max_new_tokens, 
            temperature=temprature,
            top_k=top_k,
            top_p=top_p)



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