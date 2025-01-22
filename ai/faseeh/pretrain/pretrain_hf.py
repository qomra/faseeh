import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from itertools import chain

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

logger = logging.getLogger(__name__)

@dataclass
class PretrainerConfig:
    """Configuration for the Pretrainer class."""
    output_dir: str = field(default="./pretrainer_output")
    block_size: Optional[int] = field(default=None)
    seed: int = field(default=42)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1)
    num_train_epochs: float = field(default=1.0)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.0)
    warmup_steps: int = field(default=0)
    logging_steps: int = field(default=500)
    save_steps: int = field(default=1000)
    save_total_limit: Optional[int] = field(default=2)
    fp16: bool = field(default=False)
    torch_dtype: Optional[str] = field(default=None)
    use_flash_attention_2: bool = field(default=False)
    resume_from_checkpoint: bool = field(default=True)
    

class CustomDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Get max length in this batch
        batch_max_len = max(len(feature["input_ids"]) for feature in features)
        
        # Prepare batch
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        # Pad sequences to max length in batch
        for feature in features:
            padding_length = batch_max_len - len(feature["input_ids"])
            
            # Pad input_ids and labels
            padded_input_ids = feature["input_ids"] + [self.pad_token_id] * padding_length
            padded_labels = feature["labels"] + [-100] * padding_length  # -100 is ignored in loss calculation
            attention_mask = feature["attention_mask"] + [0] * padding_length
            
            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(padded_labels)
        
        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch

class Pretrainer:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer,
        output_path: str,
        config: Optional[PretrainerConfig] = None
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.output_path = output_path
        self.config = config if config is not None else PretrainerConfig()
        # update output path
        self.config.output_dir = output_path
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
        # Initialize model and other components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize the model and other necessary components."""
        # Load config
        config_kwargs = {"trust_remote_code": True}
        if self.model_name_or_path:
            self.model_config = AutoConfig.from_pretrained(self.model_name_or_path, **config_kwargs)
        else:
            self.model_config = CONFIG_MAPPING[self.config.model_type]()
            logger.warning("Training new model from scratch")

        # Load model
        torch_dtype = (
            self.config.torch_dtype
            if self.config.torch_dtype in ["auto", None]
            else getattr(torch, self.config.torch_dtype)
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            config=self.model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            use_flash_attention_2=self.config.use_flash_attention_2
        )
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Resize embeddings if necessary
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        logging.info(f"Embedding size: {embedding_size} | Vocab size: {len(self.tokenizer)}")
        if len(self.tokenizer) != embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _get_block_size(self):
        """Determine the appropriate block size for training."""
        if self.config.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 8192:
                logger.warning(
                    "The tokenizer picked seems to have a very large `model_max_length` "
                    f"({self.tokenizer.model_max_length}). Using block_size=8192."
                )
                block_size = 8192
        else:
            if self.config.block_size > self.tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({self.config.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.config.block_size, self.tokenizer.model_max_length)
        logger.info(f"Using block size: {block_size}")
        return block_size

    def _preprocess_dataset(self, dataset: Dataset):
        """Preprocess the dataset for training."""
        block_size = self._get_block_size()
        
        def tokenize_function(examples):
            outputs = self.tokenizer(
                [str(item) for item in examples["content"]],
                truncation=True,
                max_length=block_size,
                return_attention_mask=True
            )
            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
                "labels": outputs["input_ids"]
            }

        processed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
            batch_size=100
        )

        return processed_dataset

    def train(self, dataset: Dataset):
        """Train the model on the provided dataset."""
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
        # Preprocess dataset
        train_dataset = self._preprocess_dataset(dataset)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
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
            resume_from_checkpoint=self.config.resume_from_checkpoint
        )

        # Initialize trainer with custom collator
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=CustomDataCollator(self.tokenizer)
        )

        # Check for existing checkpoint
        last_checkpoint = None
        if training_args.resume_from_checkpoint:
            logging.info("Checking for existing checkpoint...")
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None:
                logging.info(f"Found checkpoint at {last_checkpoint}")
            else:
                logging.info("No checkpoint found")
        
        # Start training
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
        
        # Log and save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        return metrics