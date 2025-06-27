import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Run end-to-end Arabic root lookup using multiple LoRA models')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-8B-Instruct',
                        help='Base model to use')
    parser.add_argument('--root_analyzer_path', type=str, default='models/rooter',
                        help='Path to the root analyzer LoRA model')
    parser.add_argument('--memorizer_path', type=str, default='models/memorizer',
                        help='Path to the memorizer LoRA model')
    parser.add_argument('--max_length', type=int, default=4096,
                        help='Maximum context length')
    parser.add_argument('--load_8bit', action='store_true',
                        help='Load model in 8-bit precision')
    parser.add_argument('--load_4bit', action='store_true',
                        help='Load model in 4-bit precision')
    return parser.parse_args()

def load_base_model(args):
    """Load the base model with appropriate quantization settings"""
    print(f"Loading base model: {args.model_name}")
    
    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure model loading based on precision arguments
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    
    if args.load_8bit:
        model_kwargs["load_in_8bit"] = True
    elif args.load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        model_kwargs["quantization_config"] = quantization_config
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    return base_model, tokenizer

def run_root_analyzer(question, model, tokenizer, max_length=4096):
    """Run the root analyzer model to get the analysis"""
    system_prompt = "أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية."
    prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate the analysis
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Get the model's output (skip the prompt)
    prompt_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    # Extract the content between <analyze> tags
    analysis = ""
    if "<analyze>" in response and "</analyze>" in response:
        start_idx = response.find("<analyze>") + len("<analyze>")
        end_idx = response.find("</analyze>")
        analysis = response[start_idx:end_idx].strip()
    else:
        analysis = response.strip()
    
    return analysis

def run_memorizer(question, analysis, model, tokenizer, max_length=4096):
    """Run the memorizer model to retrieve content from Lisan Al-Arab"""
    system_prompt = "أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية ومعانيها من معجم لسان العرب."
    prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{analysis}\n</analyze>\n\n<lisan alarab>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate the content
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=2048,  # Longer to fit dictionary entries
            temperature=0.3,  # Lower temperature for more faithful recall
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Get the model's output (skip the prompt)
    prompt_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    # Extract the content until </lisan alarab>
    lisan_content = ""
    if "</lisan alarab>" in response:
        end_idx = response.find("</lisan alarab>")
        lisan_content = response[:end_idx].strip()
    else:
        lisan_content = response.strip()
    
    return lisan_content

def generate_answer(question, analysis, lisan_content, model, tokenizer, max_length=4096):
    """Generate a final answer using the base model without LoRA adapters"""
    system_prompt = "أنت خبير في علم اللغة العربية متخصص في تفسير معاني الكلمات العربية."
    
    # Create a prompt that includes the analysis and Lisan Al-Arab content
    prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{analysis}\n</analyze>\n\n<lisan alarab>\n{lisan_content}\n</lisan alarab>\n\n<answer>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Get the model's output (skip the prompt)
    prompt_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    # Extract until </answer> if present
    answer = ""
    if "</answer>" in response:
        end_idx = response.find("</answer>")
        answer = response[:end_idx].strip()
    else:
        answer = response.strip()
    
    return answer

def main():
    args = parse_args()
    
    # Load base model and tokenizer
    print("Loading models...")
    base_model, tokenizer = load_base_model(args)
    
    # Keep a copy of the base model for answer generation
    base_model_copy = base_model
    
    # Load LoRA models
    print(f"Loading root analyzer from {args.root_analyzer_path}")
    analyzer_model = PeftModel.from_pretrained(base_model, args.root_analyzer_path)
    
    print(f"Loading memorizer from {args.memorizer_path}")
    memorizer_model = PeftModel.from_pretrained(base_model_copy, args.memorizer_path)
    
    print("Models loaded successfully. Enter a question to begin (type 'exit' to quit):")
    
    while True:
        question = input("\nأدخل سؤالك عن كلمة عربية: ")
        if question.lower() == 'exit':
            break
        
        print("\nجاري تحليل الكلمة والبحث عن جذرها...")
        
        # Step 1: Run root analyzer to get analysis
        analysis = run_root_analyzer(question, analyzer_model, tokenizer, args.max_length)
        print(f"\n<analyze>\n{analysis}\n</analyze>")
        
        # Step 2: Run memorizer to get content from Lisan Al-Arab
        print("\nجاري البحث في لسان العرب...")
        lisan_content = run_memorizer(question, analysis, memorizer_model, tokenizer, args.max_length)
        print(f"\n<lisan alarab>\n{lisan_content}\n</lisan alarab>")
        
        # Step 3: Generate final answer using base model (without LoRA)
        print("\nجاري إعداد الإجابة النهائية...")
        
        # Load a fresh copy of the base model (to ensure no LoRA adapters are attached)
        answer_model, _ = load_base_model(args)
        
        answer = generate_answer(question, analysis, lisan_content, answer_model, tokenizer, args.max_length)
        print(f"\n<answer>\n{answer}\n</answer>")

if __name__ == "__main__":
    main()