
import logging
from typing import List, Dict, Any

from .model import LlamaForCausalLM
from .utils import (formatting_prompts_func,
                    get_instruction_template,
                    get_response_template)

from transformers import LlamaConfig
from trl import SFTConfig, SFTTrainer,DataCollatorForCompletionOnlyLM


class SFTTrainer:
    def __init__(self,
                 sft_config,
                 llama_config,
                 tokenizer,
                 pretrain_model,
                 output_dir):
        self.sft_config = SFTConfig(**sft_config)
        self.llama_config = LlamaConfig(**llama_config)
        self.tokenizer = tokenizer
        self.pretrain_model = pretrain_model
        self.output_dir = output_dir
    
    def train(self, dataset):

        sft_model = LlamaForCausalLM(self.llama_config)
        logging.info("Copying weights from pre-trained model")
        sft_model.copy_weights_from_pretrained(self.pretrain_model)

        # Prepare data collator
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=self.tokenizer,
            instruction_template=get_instruction_template(self.tokenizer),
            response_template=get_response_template(self.tokenizer)
        )

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