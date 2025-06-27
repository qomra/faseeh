import json
import os
import random
import argparse
import pyarabic.araby as araby
from tqdm import tqdm

def strip_tashkeel(text):
    """Strip tashkeel from Arabic text"""
    return araby.strip_tashkeel(text)

def main():
    parser = argparse.ArgumentParser(description="Prepare memorization dataset for Lisan Al-Arab")
    parser.add_argument("--lisan_alarab", type=str, default="lisan_alarab.json",
                        help="Path to the Lisan Al-Arab JSON file")
    parser.add_argument("--analyses", type=str, default="root_analysis_dataset.json",
                        help="Path to the synthesized analyses JSON file")
    parser.add_argument("--output", type=str, default="memorization_dataset.json",
                        help="Output path for the created dataset")
    args = parser.parse_args()
    
    # Load Lisan Al-Arab dictionary
    print(f"Loading Lisan Al-Arab from {args.lisan_alarab}...")
    with open(args.lisan_alarab, 'r', encoding='utf-8') as f:
        lisan_alarab = json.load(f)
    
    # Strip tashkeel from all roots in Lisan Al-Arab
    print("Stripping tashkeel from Lisan Al-Arab roots...")
    clean_lisan_alarab = {}
    for root, content in lisan_alarab.items():
        clean_root = strip_tashkeel(root)
        clean_lisan_alarab[clean_root] = content
    
    # Load the synthesized analyses
    print(f"Loading analyses from {args.analyses}...")
    with open(args.analyses, 'r', encoding='utf-8') as f:
        analyses = json.load(f)
    
    # Create dataset examples
    print("Creating memorization dataset examples...")
    examples = []
    
    for item in tqdm(analyses):
        question = item["question"]
        analysis = item["analysis"]
        
        # Get the root, either from extracted_root or answer field
        if "extracted_root" in item and item["extracted_root"]:
            root = strip_tashkeel(item["extracted_root"])
        elif "answer" in item and item["answer"]:
            root = strip_tashkeel(item["answer"])
        else:
            # Skip items without a clear root
            continue
        
        # Check if the root exists in Lisan Al-Arab
        if root in clean_lisan_alarab:
            content = clean_lisan_alarab[root]
            has_root = True
        else:
            content = ""
            has_root = False
        
        # Create the input with the question and analysis
        input_text = f"<|start_header_id|>system<|end_header_id|>أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية ومعانيها من معجم لسان العرب.<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{analysis}\n</analyze>\n\n<lisan alarab>\n"
        
        # Create the output based on whether the root exists - stopping at </lisan alarab>
        if has_root:
            output_text = f"{content}\n</lisan alarab>"
        else:
            output_text = f"لم يتم العثور على جذر '{root}' في معجم لسان العرب.\n</lisan alarab>"
        
        # Create example
        example = {
            "input": input_text,
            "output": output_text,
            "root": root,
            "has_root": has_root,
            "original_question": question
        }
        examples.append(example)
    
    
    # Add examples for non-existent roots as well
    non_existent_roots = [
        "زخط", "ضغث", "طعس", "صفق", "قظع", "ظكل", "جنفش", "قسطم", 
        "خصلد", "ذبعر", "طخزق", "ضمنج", "كظفر", "غذمل", "شنجب"
    ]
    
    for root in non_existent_roots:
        if root not in clean_lisan_alarab:
            question = f"ما معنى جذر '{root}' في معجم لسان العرب؟"
            analysis = f"عند تحليل السؤال، نجد أنه يتعلق بالجذر '{root}'.\n\nالجذر يتكون من الحروف: {', '.join(root)}.\n\nسنبحث عن هذا الجذر في معجم لسان العرب.\n\n{{'root': '{root}'}}"
            
            input_text = f"<|start_header_id|>system<|end_header_id|>أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية ومعانيها من معجم لسان العرب.<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{analysis}\n</analyze>\n\n<lisan alarab>\n"
            output_text = f"لم يتم العثور على جذر '{root}' في معجم لسان العرب.\n</lisan alarab>"
            
            example = {
                "input": input_text,
                "output": output_text,
                "root": root,
                "has_root": False,
                "original_question": question
            }
            examples.append(example)
    
    # # Shuffle and split the dataset (80% train, 20% validation)
    # random.seed(42)  # For reproducibility
    # random.shuffle(examples)
    # split_idx = int(len(examples) * 0.8)
    # train_examples = examples[:split_idx]
    # val_examples = examples[split_idx:]
    
    # Save the datasets
    # train_output = args.output.replace('.json', '_train.json')
    # val_output = args.output.replace('.json', '_val.json')
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    

if __name__ == "__main__":
    main()