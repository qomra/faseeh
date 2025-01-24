


import logging
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer,RLOOConfig, RLOOTrainer, apply_chat_template

class FaseehRewardTrainer:
    def __init__(self,
                 tokenizer,
                 base_model,
                 output_dir,
                 batch_size = 2,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.base_model_name = base_model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
        self.base_model.config.pad_token_id = tokenizer.pad_token_id
        self.training_args = RewardConfig(output_dir=output_dir, per_device_train_batch_size=batch_size)
  
    def train(self, dataset):
        logging.info("Training reward model...")
        trainer = RewardTrainer(
            args=self.training_args,
            model=self.base_model,
            processing_class=self.tokenizer,
            train_dataset=dataset,
        )
        trainer.train()

class FaseehRLTrainer:
    def __init__(self,
                 tokenizer,
                 reward_model,
                 ref_policy_model,
                 policy_model,
                 output_dir,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model, num_labels=1)
        self.reward_model.config.pad_token_id = tokenizer.pad_token_id
        self.ref_policy_model = AutoModelForCausalLM.from_pretrained(ref_policy_model)
        self.policy_model = AutoModelForCausalLM.from_pretrained(policy_model)
        self.training_args = RLOOConfig(output_dir=output_dir)
    
    def train(self, dataset):
        
        dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer})
        dataset = dataset.map(lambda x: self.tokenizer(x["prompt"]), remove_columns="prompt")
        trainer = RLOOTrainer(
        config=self.training_args,
        processing_class=self.tokenizer,
        policy=self.policy_model,
        ref_policy=self.ref_policy_model,
        reward_model=self.reward_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )
        trainer.train()
