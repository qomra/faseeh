# faseeh/pretrain/pretrain_hf4.py

import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from itertools import chain
import os

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    set_seed,
    MixtralConfig,
    MixtralForCausalLM
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

logger = logging.getLogger(__name__)

@dataclass
class PretrainerConfig:
    # ... (PretrainerConfig definition remains unchanged) ...
    output_dir: str = field(default="./pretrainer_output")
    block_size: Optional[int] = field(default=8192)
    seed: int = field(default=42)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    num_train_epochs: float = field(default=1)
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=2000)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=1000)
    save_total_limit: Optional[int] = field(default=2)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    torch_dtype: Optional[str] = field(default="bfloat16") # Keep this in PretrainerConfig
    use_flash_attention_2: bool = field(default=True)
    resume_from_checkpoint: bool = field(default=True)
    report_to: Optional[List[str]] = field(default_factory=lambda: ["tensorboard"])

    # Model Parameters for MixtralConfig
    model_type: str = field(default="mixtral")
    vocab_size: int = field(default=32000)
    hidden_size: int = field(default=1024)
    intermediate_size: int = field(default=4096)
    num_hidden_layers: int = field(default=16)
    num_attention_heads: int = field(default=16)
    num_key_value_heads: int = field(default=4)
    num_local_experts: int = field(default=8)
    num_experts_per_tok: int = field(default=1)
    router_aux_loss_coef: float = field(default=0.001)
    rms_norm_eps: float = field(default=1e-5)
    rope_theta: float = field(default=500000.0)
    max_position_embeddings: int = field(default=8192)
    sliding_window: Optional[int] = field(default=None)
    attention_bias: bool = field(default=False)
    hidden_act: str = field(default="silu")
    initializer_range: float = field(default=0.02)
    use_cache: bool = field(default=True)


class Pretrainer:
    def __init__(
        self,
        tokenizer,
        output_path: str,
        config: Optional[PretrainerConfig] = None
    ):
        self.tokenizer = tokenizer
        self.output_path = output_path
        self.config = config if config is not None else PretrainerConfig()

        self.config.output_dir = output_path

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        self._initialize_components()

    def _initialize_components(self):
        model_config = MixtralConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            num_local_experts=self.config.num_local_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            router_aux_loss_coef=self.config.router_aux_loss_coef,
            
            rms_norm_eps=self.config.rms_norm_eps,
            rope_theta=self.config.rope_theta,
            max_position_embeddings=self.config.max_position_embeddings,
            sliding_window=self.config.block_size,
            
            attention_bias=self.config.attention_bias,
            hidden_act=self.config.hidden_act,
            initializer_range=self.config.initializer_range,
            use_cache=self.config.use_cache,
            
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            torch_dtype="bfloat16" if self.config.bf16 else ("float16" if self.config.fp16 else "float32"),
        )

        if self.config.use_flash_attention_2:
            model_config.attn_implementation = "flash_attention_2"
        else:
            model_config.attn_implementation = "sdpa"

        model_config._name_or_path = self.output_path

        logger.info(f"Creating new MixtralForCausalLM model with config: {model_config}")
        self.model = AutoModelForCausalLM.from_config(
            model_config,
            torch_dtype=torch.bfloat16 if self.config.bf16 else (torch.float16 if self.config.fp16 else torch.float32)
        )

        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) != embedding_size:
            logger.info(f"Resizing model embeddings from {embedding_size} to {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
             self.model.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            logger.warning("Tokenizer has no pad_token_id or eos_token_id. Model's pad_token_id may be unset.")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model total parameters: {total_params / 1e6:.2f} M")
        logger.info(f"Model trainable parameters: {trainable_params / 1e6:.2f} M")

    def _get_block_size(self):
        if self.config.block_size > self.model.config.max_position_embeddings:
            logger.warning(
                f"The block_size passed ({self.config.block_size}) is larger than the maximum length for the model"
                f"({self.model.config.max_position_embeddings}). Using block_size={self.model.config.max_position_embeddings}."
            )
            block_size = self.model.config.max_position_embeddings
        else:
            block_size = self.config.block_size
        logger.info(f"Using block size: {block_size}")
        return block_size

    def _preprocess_dataset(self, dataset: Dataset):
        block_size = self._get_block_size()

        # FIX: Add 'tokenizer' and 'block_size' to the signature of the nested function
        def tokenize_and_group_texts(examples, tokenizer, block_size): # <-- ADDED tokenizer, block_size
            concatenated_content = [
                f"{ex.get('title', '')}\n\n{ex.get('content', '')}".strip()
                for ex in examples
            ]

            # Use the 'tokenizer' argument passed to the function
            tokenized_examples = tokenizer(
                concatenated_content,
                add_special_tokens=False,
                return_attention_mask=False,
                return_special_tokens_mask=False,
            )

            all_blocks_from_batch = []
            concatenated_tokens_from_batch = list(chain(*tokenized_examples["input_ids"]))

            total_length = (len(concatenated_tokens_from_batch) // block_size) * block_size
            
            for i in range(0, total_length, block_size):
                all_blocks_from_batch.append(concatenated_tokens_from_batch[i : i + block_size])
            
            result = {
                "input_ids": all_blocks_from_batch,
                "labels": all_blocks_from_batch
            }
            return result

        processed_dataset = dataset.map(
            tokenize_and_group_texts,
            batched=True,
            batch_size=500, # Adjusted batch_size for preprocessing based on smaller units (volumes)
            remove_columns=dataset.column_names,
            desc=f"Tokenizing and grouping into {block_size} blocks",
            # `tokenizer` here refers to `self.tokenizer` from the outer scope
            fn_kwargs={"tokenizer": self.tokenizer, "block_size": block_size}, # <-- Use self.tokenizer here
            num_proc=os.cpu_count(),
            load_from_cache_file=True
        )
        logger.info(f"Preprocessed dataset has {len(processed_dataset)} examples.")
        return processed_dataset
    
    def train(self, dataset: Dataset):
        set_seed(self.config.seed)

        if "input_ids" not in dataset.column_names or "labels" not in dataset.column_names:
            logger.info("Dataset not pre-tokenized with 'input_ids' and 'labels'. Preprocessing now.")
            train_dataset = self._preprocess_dataset(dataset)
        else:
            logger.info("Dataset already contains 'input_ids' and 'labels'. Skipping preprocessing step in Pretrainer.")
            train_dataset = dataset
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=False,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            # FIX: Remove use_flash_attention_2 from TrainingArguments. It's not a valid argument here.
            # use_flash_attention_2=self.config.use_flash_attention_2, # REMOVED THIS LINE
            resume_from_checkpoint=self.config.resume_from_checkpoint,
            report_to=self.config.report_to,
            dataloader_num_workers=os.cpu_count() // 2,
        )

        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        last_checkpoint = None
        if training_args.resume_from_checkpoint:
            logger.info("Checking for existing checkpoint...")
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None:
                logger.info(f"Found checkpoint at {last_checkpoint}")
            else:
                logger.info("No checkpoint found")

        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return metrics