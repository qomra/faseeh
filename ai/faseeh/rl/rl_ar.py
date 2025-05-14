import logging
import torch
import re
import json
import os
import gc
from transformers import AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import pyarabic.araby as araby
from peft import LoraConfig

def format_dataset(dataset):
    """
    Format the dataset for training with thinking steps and structured output in Arabic.
    """
    def format_example(x):
        # Prompt the model to think in Arabic and provide structured output in Arabic
        return {
            'prompt': [
                {'role': 'system', 'content': 'أنت محلل لغوي عربي مساعد. قم أولا بتحليل الكلمة المطلوبة، وحدد أنماطها الصرفية، ثم حدد جذرها. اعرض تفكيرك ثم الجذر في تنسيق منظم.'},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': x['answer']
        }
    
    return dataset.map(format_example)

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Evaluate the correctness of the root identification.
    """
    responses = [completion[0]['content'] for completion in completions]
    reward = [combined_lexical_reward(r, a) for r, a in zip(responses, answer)]
    return reward

def combined_lexical_reward(response, answer):
    """
    Calculate lexical similarity between the response and the answer.
    """
    # Normalize Arabic text
    response_norm = araby.normalize_hamza(response)
    answer_norm = araby.normalize_hamza(answer)
    
    # Check exact match
    if answer_norm in response_norm:
        return 1.0
    
    # Extract potential root mentions using regex patterns (Arabic-only patterns)
    root_patterns = [
        r'الجذر[: ]+([""«»\'\'ـ\w]+)',
        r'جذر الكلمة[: ]+([""«»\'\'ـ\w]+)',
        r'الجذر اللغوي[: ]+([""«»\'\'ـ\w]+)',
        r'الجذر الثلاثي[: ]+([""«»\'\'ـ\w]+)',
        r'جذرها[: ]+([""«»\'\'ـ\w]+)'
    ]
    
    for pattern in root_patterns:
        match = re.search(pattern, response_norm)
        if match and match.group(1) and araby.normalize_hamza(match.group(1)) == answer_norm:
            return 1.0
    
    # Try to extract from JSON-like structures
    try:
        # Find JSON-like patterns and parse them
        json_match = re.search(r'{.*}', response_norm)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            if 'جذر' in data and araby.normalize_hamza(data['جذر']) == answer_norm:
                return 1.0
            elif 'الجذر' in data and araby.normalize_hamza(data['الجذر']) == answer_norm:
                return 1.0
    except:
        pass
    
    # Try to extract from list-like structures
    list_match = re.search(r'\[(["\'"]?[\w\u0600-\u06FF، ]+["\'"]?(?:,\s*["\'"]?[\w\u0600-\u06FF، ]+["\'"]?)*)\]', response_norm)
    if list_match:
        items = re.findall(r'["\'"]?([\w\u0600-\u06FF، ]+)["\'"]?', list_match.group(1))
        for item in items:
            if araby.normalize_hamza(item.strip()) == answer_norm:
                return 1.0
    
    # Partial match with penalty
    return 0.2 if any(token == answer_norm for token in response_norm.split()) else 0.0

def structure_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward well-structured responses with analysis followed by a clear root output.
    """
    rewards = []
    
    for completion in completions:
        response = completion[0]['content']
        
        # Pattern to detect structured response with analysis and root indication (Arabic-only terms)
        has_analysis = bool(re.search(r'(تحليل|أصل الكلمة|أفكر|دعونا نحلل|الوزن|الصيغة|الجذر اللغوي|التحليل الصرفي)', response, re.IGNORECASE))
        has_structured_root = bool(re.search(r'(الجذر|جذر الكلمة|الجذر الثلاثي|الجذر اللغوي|جذرها)[: ]+[""«»\'\'ـ\w]+', response, re.IGNORECASE) or 
                                 re.search(r'["\[{][\w\u0600-\u06FF، ,]+["\]}]', response))
        
        # Calculate reward based on response structure
        structure_score = 0.0
        if has_analysis:
            structure_score += 0.5
        if has_structured_root:
            structure_score += 0.5
            
        rewards.append(structure_score)
    
    return rewards

def consistency_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Evaluate the consistency between analysis and conclusion (root).
    """
    rewards = []
    
    for completion, correct_answer in zip(completions, answer):
        response = completion[0]['content']
        
        # Check if the correct root is mentioned in analysis
        root_in_analysis = 0.0
        if correct_answer in response:
            mentions = response.count(correct_answer)
            if mentions > 1:  # Root is mentioned multiple times (analysis + conclusion)
                root_in_analysis = 1.0
            elif mentions == 1:  # Root is only mentioned once
                root_in_analysis = 0.5
        
        rewards.append(root_in_analysis)
    
    return rewards

def extract_root_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward responses that clearly extract the final root(s) in a structured format.
    """
    rewards = []
    
    for completion, correct_answer in zip(completions, answer):
        response = completion[0]['content']
        
        # Look for structured output formats like lists, JSON, or clearly marked root (Arabic patterns)
        json_pattern = r'{.*"جذر"s*:s*"([^"]+)".*}'
        alt_json_pattern = r'{.*"الجذر"s*:s*"([^"]+)".*}'
        list_pattern = r'\[(["\'"]?[\w\u0600-\u06FF، ]+["\'"]?(?:,\s*["\'"]?[\w\u0600-\u06FF، ]+["\'"]?)*)\]'
        root_label_pattern = r'(الجذر|جذر الكلمة|الجذر الثلاثي|الجذر اللغوي|جذرها)[: ]+([""«»\'\'ـ\w]+)'
        
        has_structured_output = (
            bool(re.search(json_pattern, response)) or
            bool(re.search(alt_json_pattern, response)) or
            bool(re.search(list_pattern, response)) or
            bool(re.search(root_label_pattern, response))
        )
        
        rewards.append(1.0 if has_structured_output else 0.0)
    
    return rewards


class FaseehGRPOTrainer:
    def __init__(self,
                 tokenizer,
                 base_model,
                 output_dir,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        
        # Set environment variable to avoid memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Configure GRPO training parameters
        self.training_args = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=1,  # Small batch size to avoid OOM
            gradient_accumulation_steps=2,  # Increase steps to maintain effective batch size
            optim="adamw_torch",
            save_steps=100,
            save_total_limit=2,
            logging_steps=10,
            learning_rate=1e-5,
            warmup_steps=1,
            lr_scheduler_type="linear",
            bf16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            gradient_checkpointing=False,  # Disable to avoid hanging
        )
        
        # Set model initialization kwargs to pass to GRPOTrainer
        self.model_init_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "load_in_8bit": True,  # Enable 8-bit quantization in model_init_kwargs
        }
        self.training_args.model_init_kwargs = self.model_init_kwargs
        
        # Store base model path for later use
        self.base_model = base_model

    def train(self, dataset):
        logging.info("تدريب نموذج GRPO...")
        
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        # Log GPU information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logging.info(f"التدريب باستخدام {gpu_count} وحدة معالجة رسومات")
            for i in range(gpu_count):
                logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Format dataset for GRPO
        dataset = format_dataset(dataset)
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Initialize the GRPOTrainer with built-in PEFT support
        trainer = GRPOTrainer(
            model=self.base_model,  # Pass model path, not instance
            reward_funcs=[
                correctness_reward_func,
                structure_reward_func,
                consistency_reward_func,
                extract_root_reward_func
            ],
            args=self.training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            peft_config=lora_config  # Use built-in PEFT support
        )
        
        # Train the model
        trainer.train()
        
        # Model will be saved automatically based on save_steps and save_total_limit