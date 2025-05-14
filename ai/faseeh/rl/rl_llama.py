import logging
import torch
import re
import json
import os
import gc
from trl import GRPOTrainer, GRPOConfig
import pyarabic.araby as araby
from peft import LoraConfig

def format_dataset(dataset):
    """
    Format the dataset for training with thinking steps and structured output in Arabic.
    """
    def format_example(x):
        # Prompt the model to think in Arabic and provide structured output in Arabic
        system_prompt = """<|start_header_id|>system<|end_header_id|>
أنت محلل لغوي عربي. قم بتحليل الكلمة المطلوبة في السؤال بالتفصيل على النحو التالي:
1. حدّد الحروف الأصلية في الكلمة.
2. بيّن الوزن الصرفي للكلمة.
3. أعط مثالاً أو مثالين لكلمات أخرى من نفس الجذر.
اذكر التحليل بين علامتي <analyzing> و </analyzing>،
ثم حدد جذرها بوضوح في النهاية بهذه الصيغة:
<answer>
{root: "جذر الكلمة"}
</answer>
<|eot_id|>
"""

        return {
            'prompt': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': "<|start_header_id|>user<|end_header_id|>"+x['question']+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"}
            ],
            'answer': x['answer']
        }
    
    return dataset.map(format_example)

# Reward Functions

def extract_answer_root(text: str) -> str:
    """
    Extract the root from the model's answer section.
    Looks for JSON-like structure with a root key in the answer section.
    Updated for the new format using <analyzing> and <answer> tags.
    """
    try:
        # First try to extract the answer section with the new format
        answer_pattern = r'<answer>(.*?)(?:</answer>|$)'
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        if not answer_match:
            return ""
        
        answer_text = answer_match.group(1).strip()
        
        # Try to extract JSON-like structure
        root_pattern = r'{.*?root.*?:.*?"(.*?)".*?}'
        root_match = re.search(root_pattern, answer_text, re.DOTALL)
        if root_match:
            return root_match.group(1).strip()
        
        # Fallback: try to find any Arabic word after "root" or "جذر"
        fallback_pattern = r'(?:root|جذر).*?["\s:]+([ء-ي]{2,})'
        fallback_match = re.search(fallback_pattern, answer_text, re.DOTALL)
        if fallback_match:
            return fallback_match.group(1).strip()
        
        return ""
    except Exception as e:
        print(f"Error extracting root: {e}")
        return ""

def structure_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion follows the required structure.
    Looks for <analyzing> and <answer> sections in the output.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of float rewards (0.7 for correct structure, 0.0 otherwise)
    """
    pattern = r'<analyzing>.*?</analyzing>.*?<answer>.*?</answer>'
    responses = [completion[0]['content'] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.7 if match else 0.0 for match in matches]

def analysis_quality_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks the quality of the analysis section.
    Looks for morphological pattern identification in the <analyzing> section.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of float rewards
    """
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for response in responses:
        reward = 0.0
        
        # Extract analysis section with new format
        analysis_pattern = r'<analyzing>(.*?)</analyzing>'
        analysis_match = re.search(analysis_pattern, response, re.DOTALL)
        
        if analysis_match:
            analysis_text = analysis_match.group(1).strip()
            
            # Check for discussion of morphological patterns (وزن)
            if 'وزن' in analysis_text:
                reward += 0.3
                
            # Check for detailed analysis with multiple steps
            if len(analysis_text.split('\n')) > 3:
                reward += 0.2
                
            # Check if analysis mentions prefixes/suffixes
            if any(term in analysis_text for term in ['سابقة', 'لاحقة', 'زائدة', 'prefix', 'suffix']):
                reward += 0.2
        
        rewards.append(reward)
    
    return rewards

def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    
    # Debug print
    if responses:
        print(f"SAMPLE RESPONSE:\n{responses[0][:300]}")
    
    extracted_roots = [extract_answer_root(r) for r in responses]
    print(f"EXTRACTED ROOTS: {extracted_roots}")
    print(f"EXPECTED ANSWERS: {answer}")
    
    # Normalize Arabic text for comparison (remove diacritics and normalize letters)
    normalized_extracted = [araby.strip_tashkeel(root) for root in extracted_roots]
    normalized_answers = [araby.strip_tashkeel(ans) for ans in answer]
    
    # Print debug information for the first example
    if len(prompts) > 0 and len(completions) > 0 and len(answer) > 0:
        q = prompts[0][-1]['content']
        # print('-'*20)
        # print(f"Question:\n{q}")
        # print(f"\nExpected Root:\n{answer[0]}")
        # print(f"\nResponse:\n{responses[0][:300]}...")  # First 300 chars
        # print(f"\nExtracted Root:\n{extracted_roots[0]}")
    
    return [2.0 if norm_root == norm_ans else 0.0 
            for norm_root, norm_ans in zip(normalized_extracted, normalized_answers)]

def consistency_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the root in the answer section is consistent
    with Arabic morphological patterns.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of float rewards
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_roots = [extract_answer_root(r) for r in responses]
    
    rewards = []
    for root in extracted_roots:
        reward = 0.0
        
        # Clean the root
        root = araby.strip_tashkeel(root)
        
        # Check if it's a valid Arabic root (typically 3-4 letters)
        if root and len(root) >= 2 and len(root) <= 4:
            reward += 0.3
            
            # Check if it contains only Arabic letters
            if all(char in araby.LETTERS for char in root):
                reward += 0.2
                
        rewards.append(reward)
    
    return rewards

def extract_root_reward_func(completions, **kwargs) -> list[float]:
    """
    Enhanced reward function that checks if the model properly extracts the root
    in a JSON-like structure in the answer section with better pattern matching
    and graduated reward scoring.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of float rewards with more nuanced scoring
    """
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for response in responses:
        reward = 0.0
        
        # Extract answer section with new format
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        
        if not answer_match:
            # No answer section found
            rewards.append(0.0)
            continue
            
        answer_text = answer_match.group(1).strip()
        
        if not answer_text:
            # Empty answer section
            rewards.append(0.1)  # Small reward for having the section at least
            continue
            
        # Check for proper JSON-like format with root key
        # More flexible pattern that handles whitespace and optional quotes around keys
        json_pattern = r'{.*?(?:"root"|root|"جذر"|جذر).*?:.*?"([ء-ي\s]+)".*?}'
        json_match = re.search(json_pattern, answer_text, re.DOTALL)
        
        if json_match:
            # Extract the actual root value
            root_value = json_match.group(1).strip()
            
            # Check if it's a valid Arabic root (typically 2-4 letters)
            if root_value and 2 <= len(araby.strip_tashkeel(root_value)) <= 4:
                reward += 0.8
                # Bonus for properly formatted root
                if all(char in araby.LETTERS + 'ءئؤإأ' for char in araby.strip_tashkeel(root_value)):
                    reward += 0.2
            else:
                # JSON format correct but invalid root value
                reward += 0.5
        elif '{' in answer_text and '}' in answer_text:
            # Attempted JSON but incorrect format
            if any(term in answer_text for term in ['"root"', 'root', '"جذر"', 'جذر']):
                reward += 0.4  # Attempted root key in JSON-like structure
            else:
                reward += 0.2  # Just JSON-like structure
        elif any(term in answer_text for term in ['"root"', 'root', '"جذر"', 'جذر']):
            # Mentioned root key but not in JSON format
            reward += 0.3
            
        rewards.append(reward)
    
    return rewards

def advanced_analysis_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function to encourage advanced Arabic morphological analysis.
    Checks the analyzing section for explicit discussion of:
      - Root letters
      - Morphological pattern (وزن)
      - Prefixes/Suffixes (سوابق/لواحق)
      - Alternative conjugations or derivatives
    
    Args:
        completions: List of model completions
    
    Returns:
        List of float rewards (range: 0.0 - 1.0)
    """
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for response in responses:
        reward = 0.0

        analysis_pattern = r'<analyzing>(.*?)</analyzing>'
        analysis_match = re.search(analysis_pattern, response, re.DOTALL)

        if analysis_match:
            analysis_text = analysis_match.group(1).strip()

            # Check for explicit root letter identification (حروف الجذر)
            if re.search(r'(حروف الجذر|الحروف الجذرية|الجذر)', analysis_text):
                reward += 0.3
            
            # Check for explicit root letter identification (حروف الجذر)
            if re.search(r'(ثلاثي|رباعي|خماسي)', analysis_text):
                reward += 0.3

            # Check for morphological pattern (وزن)
            if re.search(r'(وزن|الوزن الصرفي)', analysis_text):
                reward += 0.25

            # Check for discussion of prefixes/suffixes explicitly
            if re.search(r'(سابقة|لاحقة|زوائد|أحرف زائدة|سوابق|لواحق|الحروف الأصلية|حروف الجذر|الأحرف الأساسية)', analysis_text):
                reward += 0.25

            # Check for mentioning alternative conjugations or related derivatives
            if re.search(r'(تصريفات أخرى|مشتقات|صيغ أخرى|مثل)', analysis_text):
                reward += 0.2

        rewards.append(reward)

    return rewards


class FaseehGRPOTrainer:
    def __init__(self,
                 tokenizer,
                 base_model,
                 output_dir,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.experiment_name = kwargs.get("experiment_id", "grpo_experiment")
        
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
        
        # Create callbacks
        from .callback import SimpleJsonlLogger, DebugLoggingCallback
        logging_callback = SimpleJsonlLogger(self.output_dir)
        debug_callback = DebugLoggingCallback(self.output_dir)
        # Initialize the GRPOTrainer with built-in PEFT support
        trainer = GRPOTrainer(
            model=self.base_model,  # Pass model path, not instance
            reward_funcs=[
                correctness_reward_func,
                advanced_analysis_reward_func,
                structure_reward_func,
                consistency_reward_func,
                #extract_root_reward_func,
                #analysis_quality_reward_func
            ],
            args=self.training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            peft_config=lora_config,  # Use built-in PEFT support,
            callbacks=[logging_callback,debug_callback]
        )
        
        # Train the model
        trainer.train()
        
        # Model will be saved automatically based on save_steps and save_total_limit