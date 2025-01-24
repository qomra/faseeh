
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
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # set padding side to left
        self.model.tokenizer.padding_side = "left"


    def generate(self, dataset: Dataset):
        logging.info("Generating completions...")
        results = []
        # copy the dataset
        dataset = dataset.map(lambda x: x)
        # remove the last message from each conversation
        dataset = dataset.map(lambda x: {"conversation": x["conversation"][:-1]})
        # run the pipeline on the dataset
        dataset_conversations = dataset["conversation"]
        # sample 100 items
        dataset_conversations = dataset_conversations
        results = self.model(
            dataset_conversations,max_new_tokens=1000,
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False,
            batch_size=128)



        return results
