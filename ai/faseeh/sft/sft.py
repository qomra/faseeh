
import torch
import logging
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM

from .model import LlamaForCausalLM
from .utils import (formatting_prompts_func,
                    get_instruction_template,
                    get_response_template)
from ..pretrain import load_pretrained_model

from transformers import LlamaConfig
from trl import SFTConfig, SFTTrainer,DataCollatorForCompletionOnlyLM


class FaseehSFTTrainer:
    def __init__(self,
                 sft_config,
                 tokenizer,
                 pretrained_model_kind,
                 pretrain_model_ckpt,
                 output_dir,
                 llama_config = None,
                 sample_size = -1,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.sft_config = SFTConfig(**sft_config)
        self.pretrain_model_ckpt = pretrain_model_ckpt
        self.pretrained_model_kind = pretrained_model_kind
        if self.pretrained_model_kind == "faseeh":
            self.llama_config = LlamaConfig(**llama_config)
        self.sample_size = sample_size
        
        
    
    def train(self, dataset):
        if self.pretrained_model_kind == "faseeh":    
            pre_trained_model = load_pretrained_model(self.pretrain_model_ckpt)
            sft_model = LlamaForCausalLM(self.llama_config)
            logging.info("Copying weights from pre-trained model")
            sft_model.copy_weights_from_pretrained(pre_trained_model)
            # delete pre_trained_model to free memory
            del pre_trained_model  # Remove the reference
            torch.cuda.empty_cache()  # Clear unused memory cache
        else:
            sft_model = AutoModelForCausalLM.from_pretrained(self.pretrain_model_ckpt)

        # Prepare data collator
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=self.tokenizer,
            instruction_template=get_instruction_template(self.tokenizer),
            response_template=get_response_template(self.tokenizer)
        )
        if self.sample_size > 0:
            sample_size = self.sample_size
            # if sample_size is float then it is a percentage
            if isinstance(self.sample_size, float):
                sample_size = int(len(dataset) * self.sample_size)
            logging.info(f"Sampling {sample_size} examples from the dataset")
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
        # Prepare SFT trainer
        trainer = SFTTrainer(
            sft_model,
            args=self.sft_config,
            train_dataset=dataset,
            formatting_func=formatting_prompts_func,
            processing_class=self.tokenizer,
            data_collator=data_collator
        )

        logging.info("Training the model...")
        # Train the model
        trainer.train()
    
        logging.info("Saving the model...")
        # Save the model
        sft_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)