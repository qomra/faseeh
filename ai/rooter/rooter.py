import os
import json
import torch
import random
import argparse
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
    TaskType
)
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Train or run inference with Arabic root analysis model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='Mode to run the script (train or inference)')
    parser.add_argument('--model_name', type=str, default='/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/meta-llama/Llama-3.1-8B-Instruct/',
                        help='Base model to use')
    parser.add_argument('--dataset_path', type=str, default='root_analysis_dataset.json',
                        help='Path to the dataset JSON file')
    parser.add_argument('--output_dir', type=str, default='models/rooter',
                        help='Directory to save the model')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--max_seq_length', type=int, default=8000, help='Maximum sequence length')
    parser.add_argument('--use_flash_attn', action='store_true', help='Use Flash Attention if available')

    return parser.parse_args()

def load_model_and_tokenizer(model_name,training=True):
    """Load model and tokenizer for either training or inference."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if training:
        # For training, use 8-bit quantization for better speed
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True
        )
        

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
        
        # Prepare for LoRA training
        model = prepare_model_for_kbit_training(model)
    else:
        # For inference, load in 8-bit to save memory but maintain quality
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            load_in_8bit=True
        )
    
    return model, tokenizer

def prepare_lora_config(args):
    """Prepare LoRA configuration."""
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def preprocess_dataset(data, tokenizer, max_seq_length):
    """Preprocess dataset efficiently for training."""
    
    processed_texts = []
    input_lengths = []
    
    # Process all examples first
    for item in tqdm(data, desc="Preprocessing data"):
        input_text = item["input"]
        output_text = item["output"]
        full_text = input_text + output_text
        
        # Calculate input length for masking
        input_tokens = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
        input_length = len(input_tokens["input_ids"][0])
        
        processed_texts.append(full_text)
        input_lengths.append(input_length)
    
    # Batch tokenize all texts
    print(f"Tokenizing {len(processed_texts)} examples...")
    tokenized = tokenizer(
        processed_texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    # Mask the input parts for all examples at once
    for i, input_length in enumerate(input_lengths):
        tokenized["labels"][i, :input_length] = -100
    
    return tokenized

def prepare_dataset(dataset_path, tokenizer, max_seq_length, train_ratio=0.8):
    """Prepare dataset for training and evaluation."""
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Generate prompt-completion pairs for training
    examples = []
    for item in data:
        # Format with correct LLaMA 3.1 template
        input_text = f"<|start_header_id|>system<|end_header_id|>أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية.<|eot_id|><|start_header_id|>user<|end_header_id|>{item['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        output_text = f"<analyze>\n{item['analysis']}\n</analyze>"
        
        examples.append({
            "input": input_text,
            "output": output_text,
        })
    
    # Shuffle and split the dataset
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")
    
    # Tokenize datasets
    train_tokenized = preprocess_dataset(train_examples, tokenizer, max_seq_length)
    eval_tokenized = preprocess_dataset(eval_examples, tokenizer, max_seq_length)
    
    # Convert to datasets
    train_dataset = Dataset.from_dict(train_tokenized)
    eval_dataset = Dataset.from_dict(eval_tokenized)
    
    return train_dataset, eval_dataset

def train_model(args):
    """Train the model with LoRA."""
    set_seeds(42)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, training=True)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(
        args.dataset_path, tokenizer, args.max_seq_length
    )
    
    # Configure LoRA
    lora_config = prepare_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=100,  # Less frequent evaluation
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=100,  # Less frequent saving
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        fp16=True,
        bf16=False,
        load_best_model_at_end=True,
        report_to="tensorboard",
        save_total_limit=3,  # Keep fewer checkpoints
        optim="paged_adamw_8bit",
        gradient_checkpointing=False,  # Disable gradient checkpointing for speed
        max_grad_norm=1.0,  # Clip gradients
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create and start the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")

def run_inference(args):
    """Run inference with the trained model."""
    print("Loading model for inference...")
    
    # Load base model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer(args.model_name, training=False)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.output_dir)
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded. You can start entering questions (type 'exit' to quit):")
    
    while True:
        # Get user input
        question = input("\nEnter a question about an Arabic word: ")
        if question.lower() == 'exit':
            break
        
        # Create prompt with correct LLaMA 3.1 template
        prompt = f"<|start_header_id|>system<|end_header_id|>أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية.<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n"
        
        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        
        # Generate with streaming
        print("\nGenerating response...\n")
        
        # Start the generation
        generated_text = ""
        
        with torch.no_grad():
            for new_tokens in model.generate(
                input_ids=input_ids,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                streaming=True,
            ):
                # Get the new token
                token = new_tokens[0, -1].unsqueeze(0)
                
                # Convert token to text and print
                new_text = tokenizer.decode(token)
                generated_text += new_text
                print(new_text, end="", flush=True)
                
                # Check if we've reached the end tag
                if "</analyze>" in generated_text:
                    break
        
        print("\n\nGeneration completed.\n")

def main():
    args = parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'inference':
        run_inference(args)
    else:
        print(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()