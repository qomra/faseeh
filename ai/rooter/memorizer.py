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
    parser = argparse.ArgumentParser(description='Train or run inference with Lisan Al-Arab memorization model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='Mode to run the script (train or inference)')
    parser.add_argument('--model_name', type=str, default='/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/meta-llama/Llama-3.1-8B-Instruct/',
                        help='Base model to use')
    parser.add_argument('--dataset_path', type=str, default='memorization_dataset_short.json',
                        help='Path to the training dataset JSON file')
    parser.add_argument('--output_dir', type=str, default='models/memorizer',
                        help='Directory to save the model')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--max_seq_length', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--use_flash_attn', action='store_true', help='Use Flash Attention if available')
    return parser.parse_args()

def load_model_and_tokenizer(model_name, training=True, use_flash_attn=False, max_seq_length=4096):
    """Load model and tokenizer for either training or inference."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if training:
        # For training, use 8-bit quantization to save memory while maintaining better performance than 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True
        )
        
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        
        # Add Flash Attention if requested and supported
        if use_flash_attn:
            try:
                # Try to check if flash attention is available
                import importlib.util
                has_flash_attn = importlib.util.find_spec("flash_attn") is not None
                
                if has_flash_attn:
                    print("Using Flash Attention 2.0 for faster training")
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                else:
                    print("Flash Attention not available, using default attention")
            except Exception as e:
                print(f"Flash Attention check failed: {e}, using default attention")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
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
    """Preprocess and tokenize the dataset more efficiently."""
    
    # First, process all the texts to create input-output pairs
    processed_texts = []
    for item in tqdm(data, desc="Preprocessing data"):
        inp = item["input"]
        
        # Get the output and truncate to conserve memory
        out = item["output"].split("\n</lisan alarab>")[0]
        # Tokenize to find a rough estimate of token count (faster than full tokenization)
        out_approx_tokens = len(out.split())
        
        # Truncate to approximately 500 tokens (~500 words as a rough estimate)
        if out_approx_tokens > 500:
            out = " ".join(out.split()[:500])
        
        out = out + "\n</lisan alarab>"
        full_text = inp + out
        
        # Get the input length in tokens for masking later
        input_tokens = tokenizer(inp, return_tensors="pt", add_special_tokens=False)
        input_length = len(input_tokens["input_ids"][0])
        
        processed_texts.append({
            "text": full_text,
            "input_length": input_length
        })
    
    # Now batch tokenize all texts
    texts = [item["text"] for item in processed_texts]
    input_lengths = [item["input_length"] for item in processed_texts]
    
    print(f"Tokenizing {len(texts)} examples...")
    tokenized = tokenizer(
        texts,
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

def prepare_dataset(train_path, tokenizer, max_seq_length):
    """Prepare dataset for training and evaluation using more efficient processing."""
    print(f"Loading training dataset from: {train_path}")
    
    # Load datasets
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Split into training and validation
    random.shuffle(data)
    val_size = int(0.1 * len(data))
    val_data = data[:val_size]
    train_data = data[val_size:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Process training data
    train_tokenized = preprocess_dataset(train_data, tokenizer, max_seq_length)
    train_dataset = Dataset.from_dict(train_tokenized)
    
    # Process validation data
    val_tokenized = preprocess_dataset(val_data, tokenizer, max_seq_length)
    val_dataset = Dataset.from_dict(val_tokenized)
    
    return train_dataset, val_dataset

def train_model(args):
    """Train the model with LoRA."""
    set_seeds(42)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, 
        training=True,
        use_flash_attn=args.use_flash_attn, 
        max_seq_length=args.max_seq_length
    )
    
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
        eval_steps=5,  # Less frequent evaluation to speed up training
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="steps",
        save_steps=5,  # Less frequent saving to reduce I/O overhead
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
        save_total_limit=3,  # Keep fewer checkpoints to save disk space
        optim="paged_adamw_8bit",
        gradient_checkpointing=False,  # Disable gradient checkpointing for speed
        max_grad_norm=1.0,  # Clip gradients to prevent instability
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
    
    print("Model loaded. You can start entering questions and analyses (type 'exit' to quit):")
    
    while True:
        # Get user input
        question = input("\nEnter a question about an Arabic word: ")
        if question.lower() == 'exit':
            break
        
        analysis = input("\nEnter the root analysis (or press Enter to use a placeholder): ")
        if not analysis.strip():
            analysis = "هذا تحليل للجذر. سيتم استخدام هذا الجذر للبحث في لسان العرب.\n{\"root\": \"أبأ\"}"
        
        # Create prompt with the LLaMA 3.1 template
        prompt = f"<|start_header_id|>system<|end_header_id|>أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية ومعانيها من معجم لسان العرب.<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{analysis}\n</analyze>\n\n<lisan alarab>\n"
        
        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        
        # Generate with streaming
        print("\nGenerating response...\n")
        
        # Start the generation
        generated_text = ""
        
        with torch.no_grad():
            # for new_tokens in model.generate(
            #     input_ids=input_ids,
            #     max_new_tokens=1024,
            #     temperature=0.7,
            #     top_p=0.9,
            #     repetition_penalty=1.1,
            #     do_sample=True,
            #     pad_token_id=tokenizer.eos_token_id,
            #     streaming=True,
            # ):
            #     # Get the new token
            #     token = new_tokens[0, -1].unsqueeze(0)
                
            #     # Convert token to text and print
            #     new_text = tokenizer.decode(token)
            #     generated_text += new_text
            #     print(new_text, end="", flush=True)
                
            #     # Check if we've reached the end tag
            #     if "</lisan alarab>" in generated_text:
            #         break
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:])
            print(generated_text)
        
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