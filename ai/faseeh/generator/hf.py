
import tqdm
import logging
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(conversation: list):
    prompt = ""
    for i, message in enumerate(conversation[:-1]):
        prompt += f"<|start_of_header|>{message['role']}<|end_of_header|>{message['content']}<|eot_id|>"

    return prompt

class HuggingFaceWrapper:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, dataset: Dataset):
        logging.info("Generating completions...")
        results = []
        for data in tqdm.tqdm(dataset):
            conversation = data['conversation']
            prompt = format_prompt(conversation)
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(inputs, max_length=50)
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(completion)
        return results
