# Modified sft.py approach for memory optimization

import torch
import logging
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .model import LlamaForCausalLM
# from .utils import (formatting_prompts_func,
#                     get_instruction_template,
#                     get_response_template)
from .utils import allam_formatting_prompts_func, get_allam_instruction_template, get_allam_response_template
from ..pretrain import load_pretrained_model

from transformers import LlamaConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from .collators import FaseehDataCollatorForCompletionOnlyLM, formatting_prompts_func,get_template_func, get_response_template_func

from peft import LoraConfig, prepare_model_for_kbit_training


from transformers import DataCollatorForLanguageModeling
import torch
import logging
from transformers import EarlyStoppingCallback


class CustomCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        
        # Store the exact token IDs we need to find based on the debug output
        self.inst_end_token_ids = [23741, 62410, 63716]  # [/INST]
        self.tokenizer = tokenizer
        
        # Log the token IDs we're looking for
        logging.info(f"Looking for response token IDs: {self.inst_end_token_ids}")
    
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        
        labels = batch["labels"].clone()
        found_any = False
        
        for i in range(labels.shape[0]):
            # Find position of [/INST] in the sequence
            response_pos = -1
            
            for pos in range(len(batch["input_ids"][i]) - len(self.inst_end_token_ids)):
                if torch.all(batch["input_ids"][i][pos:pos+len(self.inst_end_token_ids)] == torch.tensor(self.inst_end_token_ids, device=batch["input_ids"].device)):
                    # Found [/INST] token - we want to start the response right after it
                    response_pos = pos + len(self.inst_end_token_ids)
                    break
            
            # If we found [/INST], mask everything before it
            if response_pos != -1:
                labels[i, :response_pos] = -100
                found_any = True
                #logging.debug(f"Found response at position {response_pos} in example {i}")
            else:
                # If we couldn't find [/INST], mask the entire example
                labels[i, :] = -100
                #logging.warning(f"Could not find response token in example {i}")
        
        # if found_any:
        #     logging.info("Successfully found response tokens in at least one example")
        # else:
        #     logging.error("Could not find response tokens in ANY examples - check formatting!")
            
        batch["labels"] = labels
        return batch


class FaseehSFTTrainer:
    def __init__(self,
                 sft_config,
                 tokenizer,
                 pretrained_model_kind,
                 pretrain_model_ckpt,
                 output_dir,
                 llama_config = None,
                 sample_size = -1,
                 lora_config = None,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.sft_config = SFTConfig(**sft_config)
        self.pretrain_model_ckpt = pretrain_model_ckpt
        self.pretrained_model_kind = pretrained_model_kind
        if self.pretrained_model_kind == "faseeh":
            self.llama_config = LlamaConfig(**llama_config)
        self.sample_size = sample_size
        self.lora_config = None
        if lora_config is not None:
            self.lora_config = LoraConfig(**lora_config)
    from transformers import EarlyStoppingCallback

    # Add this method to your FaseehSFTTrainer class:
    def _prepare_validation_dataset(self, dataset, format_template_func):
        """Prepare validation dataset if available"""
        eval_dataset = None
        
        # Check if dataset has test split
        if dataset.column_names and "test" in dataset.column_names:
            eval_dataset = dataset["test"]
            logging.info("Validation dataset found in 'test' split")

        if eval_dataset:
            # Apply same preprocessing as training dataset
            def tokenize_function(examples):
                texts = format_template_func(examples)
                return self.tokenizer(texts, truncation=True, padding=False, max_length=2048)
            
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
            logging.info("Validation dataset prepared for early stopping")
        
        return eval_dataset
    

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
            if self.lora_config is not None:
                logging.info("Loading pre-trained model with 4-bit quantization")
                # Configure 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                   # Load model with quantization
                sft_model = AutoModelForCausalLM.from_pretrained(
                    self.pretrain_model_ckpt,
                    quantization_config=quantization_config,
                    device_map="auto",  # Automatically manage model placement
                    torch_dtype=torch.bfloat16
                )
            else:
                   # Load model with quantization
                sft_model = AutoModelForCausalLM.from_pretrained(
                    self.pretrain_model_ckpt,
                    quantization_config=quantization_config,
                    device_map="auto",  # Automatically manage model placement
                    torch_dtype=torch.bfloat16
                )
                quantization_config = None
                logging.info(f"Loading pre-trained model: {self.pretrain_model_ckpt} with quantization config: {quantization_config}")
            
         

                sft_model.train()

                # Force all parameters to be trainable
                for param in sft_model.parameters():
                    param.requires_grad = True

                # Check trainable parameters
                trainable = sum(p.numel() for p in sft_model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in sft_model.parameters())
                logging.info(f"Trainable parameters: {trainable:,} / {total:,}")

                if trainable == 0:
                    raise ValueError("No trainable parameters!")
            
            # Prepare model for k-bit training
            sft_model = prepare_model_for_kbit_training(
                sft_model,
                use_gradient_checkpointing=True
            )
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(sft_model, "enable_input_require_grads"):
                sft_model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                sft_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
       

        format_template_func = allam_formatting_prompts_func if "ALLaM" in self.pretrain_model_ckpt else formatting_prompts_func
        get_template_func_ = get_allam_instruction_template if "ALLaM" in self.pretrain_model_ckpt else get_template_func
        get_response_template_func_ = get_allam_response_template if "ALLaM" in self.pretrain_model_ckpt else get_response_template_func
        print("format_template_func", format_template_func)
        print("get_template_func", get_template_func_)
        print("get_response_template_func", get_response_template_func_)
        
        # Prepare data collator
        if "ALLaM" in self.pretrain_model_ckpt:
            data_collator = CustomCompletionOnlyLM(
                tokenizer=self.tokenizer
            )
        else: 
            data_collator = FaseehDataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer,
                instruction_template=get_template_func_(self.tokenizer),
                response_template=get_response_template_func_(self.tokenizer)
            )
        
        train_dataset = dataset["train"]

        if self.sample_size > 0:
            sample_size = self.sample_size
            # if sample_size is float then it is a percentage
            if isinstance(self.sample_size, float):
                sample_size = int(len(train_dataset) * self.sample_size)
            logging.info(f"Sampling {sample_size} examples from the dataset")
            # select last sample_size examples
            train_dataset = train_dataset.select(range(len(train_dataset)-sample_size, len(train_dataset)))
        
        def preprocess_dataset(dataset, tokenizer, formatting_func):
            def tokenize_function(examples):
                texts = formatting_func(examples)
                return tokenizer(texts, truncation=True, padding=False, max_length=2048)
            
            return dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )

        # Add this after your data collator setup and before creating SFTTrainer:
        train_dataset = preprocess_dataset(train_dataset, self.tokenizer, format_template_func)
        eval_dataset = self._prepare_validation_dataset(dataset, format_template_func)

        callbacks = []
        if eval_dataset:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
            callbacks.append(early_stopping)
            logging.info("Early stopping enabled with validation dataset")
        else:
            # No eval dataset, so disable evaluation
            self.sft_config.eval_strategy = "no"
            self.sft_config.load_best_model_at_end = False
            logging.info("No validation dataset found, disabling evaluation")

        # With this:
        trainer = SFTTrainer(
            model=sft_model,
            args=self.sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # This could be None
            data_collator=data_collator,
            peft_config=self.lora_config,
            processing_class=self.tokenizer,
            callbacks=callbacks  # Add early stopping
        )

        logging.info("Training the model...")
        # Train the model
        trainer.train()
    
        logging.info(f"Saving the model in {self.output_dir}...")
        # Save the model
        trainer.save_model()