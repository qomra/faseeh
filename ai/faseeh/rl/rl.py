

import torch
import logging
from typing import List, Dict, Any
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer,RLOOConfig, RLOOTrainer, apply_chat_template

class FaseehRewardTrainer:
    def __init__(self,
                 tokenizer,
                 base_model,
                 output_dir,
                 batch_size = 1,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.base_model_name = base_model
        model_kwargs = dict()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1, **model_kwargs)
        self.base_model.config.pad_token_id = tokenizer.pad_token_id
        
        self.training_args = RewardConfig(output_dir=output_dir, 
                                          per_device_train_batch_size=batch_size,
                                          num_train_epochs=1,
                                          save_steps=500,
                                          save_total_limit=1,
                                          logging_steps=10)
  
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

        model_kwargs = dict(
            attn_implementation="eager",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, 
        )
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model,
                                                                                **model_kwargs)
        self.ref_policy_model = AutoModelForCausalLM.from_pretrained(ref_policy_model,
                                                                      **model_kwargs)
        self.policy_model = AutoModelForCausalLM.from_pretrained(policy_model,
                                                                  **model_kwargs)
        self.training_args = RLOOConfig(output_dir=output_dir,per_device_train_batch_size=1,rloo_k=1)
    
    def train(self, dataset):
        
        dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer})
        dataset = dataset.map(lambda x: self.tokenizer(x["prompt"]), remove_columns="prompt")
        # split dataset into train and test
        dataset = dataset.train_test_split(test_size=0.2)
        logging.info(dataset)
        logging.info("Training RL model...")
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

class FaseehDPOTrainer:
    def __init__(self,
                 tokenizer,
                 base_model,
                 output_dir,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.training_args = DPOConfig(
            output_dir=output_dir,
            per_device_eval_batch_size=1,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            save_steps=100,
            save_total_limit=1,
            logging_steps=10,
            loss_type="hinge")
    
    def train(self, dataset):
        logging.info("Training DPO model...")
        # sample 100 examples from the dataset
        #dataset = dataset.select(range(1000))
        # losses = #["sigmoid","hinge","ipo","exo_pair",
                #"nca_pair","robust","bco_pair",
        losses=  ["sppo_hard"]
                #"aot","aot_pair","discopop","apo_zero","apo_down"]
        for loss in losses:
            logging.info(f"Training with loss: {loss}")
            self.training_args.loss_type = loss 
            output_dir = self.output_dir + f"_{loss}"  
            self.training_args.output_dir = output_dir
            trainer = DPOTrainer(
                model=self.model, args=self.training_args, train_dataset=dataset, processing_class=self.tokenizer
            )
            trainer.train()