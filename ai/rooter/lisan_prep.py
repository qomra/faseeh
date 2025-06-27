import json
import os
import random
<<<<<<< HEAD
import argparse
import pyarabic.araby as araby
from tqdm import tqdm
=======
import pyarabic.araby as araby
from tqdm import tqdm
import re
>>>>>>> 12f36a274bf141e6426259d93b106b0cab363fb3

def strip_tashkeel(text):
    """Strip tashkeel from Arabic text"""
    return araby.strip_tashkeel(text)

<<<<<<< HEAD
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
=======
def extract_root_from_json(analysis_text):
    """Extract the root from the JSON part at the end of the analysis"""
    # Try to find the JSON part at the end of the analysis
    json_match = re.search(r'({.*})(?:\s*)$', analysis_text)
    if json_match:
        try:
            analysis_json = json.loads(json_match.group(1))
            if 'root' in analysis_json:
                return analysis_json['root']
        except json.JSONDecodeError:
            pass
    return None

def create_analysis_variations(root):
    """Create different analysis variations for a given root"""
    variations = []
    
    # Variation 1: Super simple
    variations.append(f'الجذر هو: "{root}"\n\n{{"root": "{root}"}}')
    
    # Variation 2: With letters breakdown
    variations.append(f'الجذر هو: "{root}"\nالحروف المكونة للجذر: {" - ".join(root)}\n\n{{"root": "{root}"}}')
    
    # Variation 3: More linguistic context
    variations.append(f'بعد تحليل الكلمة المطلوبة، توصلت إلى أن جذرها هو "{root}"\n\n{{"root": "{root}"}}')
    
    # Variation 4: Structured analysis
    variations.append(f'تحليل الكلمة:\n- الجذر: {root}\n- عدد الحروف: {len(root)}\n\n{{"root": "{root}"}}')
    
    # Variation 5: With fake morphological analysis
    word_forms = ["فَعَلَ", "فَعِلَ", "فَعُلَ", "فَعَّلَ", "أَفْعَلَ", "تَفَعَّلَ", "اِنْفَعَلَ", "اِفْتَعَلَ", "اِفْعَلَّ"]
    rand_form = random.choice(word_forms)
    variations.append(f'الجذر: {root}\nالوزن: {rand_form}\nنوع الفعل: {random.choice(["لازم", "متعدي"])}\n\n{{"root": "{root}"}}')
    
    # Variation 6: Detailed technical
    variations.append(f'الجذر الأصلي للكلمة هو "{root}"\nنوع الجذر: {random.choice(["ثلاثي", "رباعي", "خماسي"])}\nأصل الاشتقاق: {random.choice(["عربي أصيل", "دخيل", "مولّد"])}\n\n{{"root": "{root}"}}')
    
    # Variation 7: With fake confidence score
    conf = random.uniform(0.85, 0.99)
    variations.append(f'الجذر المستخرج: {root}\nنسبة الثقة: {conf:.2f}\n\n{{"root": "{root}", "confidence": {conf:.2f}}}')
    
    # Variation 8: Different JSON format
    variations.append(f'تم تحليل الكلمة وتبين أن جذرها هو "{root}"\n\n{{"analysis": {{"root": "{root}", "source": "لسان العرب"}}}}')
    
    return variations

def create_question_variations(root):
    """Create different question formulations for a given root"""
    templates = [
        "ما معنى كلمة {root} في معجم لسان العرب؟",
        "ابحث عن جذر {root} في لسان العرب",
        "ما هو تفسير جذر {root} في المعاجم العربية؟",
        "اشرح لي معنى {root} كما ورد في لسان العرب",
        "ما هو أصل كلمة {root} وما معناها؟",
        "أريد معرفة معنى جذر {root}",
        "هل يمكنك البحث عن معنى {root} في لسان العرب؟",
        "ما هي دلالات جذر {root} في المعجم؟",
        "استخرج معنى {root} من لسان العرب",
        "ما معنى {root} لغةً؟"
    ]
    
    return [template.format(root=root) for template in templates]

def main():
    # Fixed paths
    lisan_alarab_path = "lisan_alarab.json"
    analyses_path = "root_analysis_dataset_short.json"
    output_path = "memorization_dataset_short.json"
    
    # Configuration
    duplicate_factor = 3  # Number of duplicates for each root
    max_content_length = 2000  # Maximum length of Lisan Al-Arab content (0 for no limit)
    
    # Load Lisan Al-Arab dictionary
    print(f"Loading Lisan Al-Arab from {lisan_alarab_path}...")
    with open(lisan_alarab_path, 'r', encoding='utf-8') as f:
        lisan_alarab = json.load(f)
    
    # Strip tashkeel from all roots in Lisan Al-Arab
    print("Processing Lisan Al-Arab roots...")
    clean_lisan_alarab = {}
    for root, content in tqdm(lisan_alarab.items()):
>>>>>>> 12f36a274bf141e6426259d93b106b0cab363fb3
        clean_root = strip_tashkeel(root)
        clean_lisan_alarab[clean_root] = content
    
    # Load the synthesized analyses
<<<<<<< HEAD
    print(f"Loading analyses from {args.analyses}...")
    with open(args.analyses, 'r', encoding='utf-8') as f:
=======
    print(f"Loading analyses from {analyses_path}...")
    with open(analyses_path, 'r', encoding='utf-8') as f:
>>>>>>> 12f36a274bf141e6426259d93b106b0cab363fb3
        analyses = json.load(f)
    
    # Create dataset examples
    print("Creating memorization dataset examples...")
    examples = []
    
<<<<<<< HEAD
    for item in tqdm(analyses):
        question = item["question"]
        analysis = item["analysis"]
        
        # Get the root, either from extracted_root or answer field
        if "extracted_root" in item and item["extracted_root"]:
=======
    # Track unique roots to avoid duplicates
    unique_roots = set()
    
    for item in tqdm(analyses, desc="Processing analysis items"):
        original_question = item["question"]
        original_analysis = item["analysis"]
        
        # Get the root, prioritizing extraction from JSON
        json_root = extract_root_from_json(original_analysis)
        if json_root:
            root = strip_tashkeel(json_root)
        elif "extracted_root" in item and item["extracted_root"]:
>>>>>>> 12f36a274bf141e6426259d93b106b0cab363fb3
            root = strip_tashkeel(item["extracted_root"])
        elif "answer" in item and item["answer"]:
            root = strip_tashkeel(item["answer"])
        else:
            # Skip items without a clear root
            continue
        
        # Check if the root exists in Lisan Al-Arab
        if root in clean_lisan_alarab:
            content = clean_lisan_alarab[root]
<<<<<<< HEAD
            has_root = True
=======
            
            # Limit content length if specified
            if max_content_length > 0 and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
                
            has_root = True
            unique_roots.add(root)
>>>>>>> 12f36a274bf141e6426259d93b106b0cab363fb3
        else:
            content = ""
            has_root = False
        
<<<<<<< HEAD
        # Create the input with the question and analysis
        input_text = f"<|start_header_id|>system<|end_header_id|>أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية ومعانيها من معجم لسان العرب.<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{analysis}\n</analyze>\n\n<lisan alarab>\n"
        
        # Create the output based on whether the root exists - stopping at </lisan alarab>
=======
        # Create examples with the original analysis first
        input_text = f"<|start_header_id|>system<|end_header_id|>أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية ومعانيها من معجم لسان العرب.<|eot_id|><|start_header_id|>user<|end_header_id|>{original_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{original_analysis}\n</analyze>\n\n<lisan alarab>\n"
        
>>>>>>> 12f36a274bf141e6426259d93b106b0cab363fb3
        if has_root:
            output_text = f"{content}\n</lisan alarab>"
        else:
            output_text = f"لم يتم العثور على جذر '{root}' في معجم لسان العرب.\n</lisan alarab>"
        
<<<<<<< HEAD
        # Create example
=======
>>>>>>> 12f36a274bf141e6426259d93b106b0cab363fb3
        example = {
            "input": input_text,
            "output": output_text,
            "root": root,
            "has_root": has_root,
<<<<<<< HEAD
            "original_question": question
        }
        examples.append(example)
    
    
    # Add examples for non-existent roots as well
=======
            "original_question": original_question
        }
        
        # Add examples with the original analysis
        if has_root:
            examples.append(example.copy())
    
    # Create additional examples with variations for all unique roots
    print(f"Creating variations for {len(unique_roots)} unique roots...")
    
    for root in tqdm(unique_roots, desc="Creating root variations"):
        if root in clean_lisan_alarab:
            content = clean_lisan_alarab[root]
            
            # Limit content length if specified
            if max_content_length > 0 and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            # Get question variations
            question_variations = create_question_variations(root)
            
            # Get analysis variations
            analysis_variations = create_analysis_variations(root)
            
            # Create examples with different combinations
            for _ in range(duplicate_factor):
                question = random.choice(question_variations)
                analysis = random.choice(analysis_variations)
                
                input_text = f"<|start_header_id|>system<|end_header_id|>أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية ومعانيها من معجم لسان العرب.<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{analysis}\n</analyze>\n\n<lisan alarab>\n"
                output_text = f"{content}\n</lisan alarab>"
                
                example = {
                    "input": input_text,
                    "output": output_text,
                    "root": root,
                    "has_root": True,
                    "original_question": question
                }
                examples.append(example)
    
    # Add examples for non-existent roots
>>>>>>> 12f36a274bf141e6426259d93b106b0cab363fb3
    non_existent_roots = [
        "زخط", "ضغث", "طعس", "صفق", "قظع", "ظكل", "جنفش", "قسطم", 
        "خصلد", "ذبعر", "طخزق", "ضمنج", "كظفر", "غذمل", "شنجب"
    ]
    
<<<<<<< HEAD
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
    
    
=======
    print("Adding examples for non-existent roots...")
    for root in non_existent_roots:
        if root not in clean_lisan_alarab:
            # For each non-existent root, create a few examples with different analyses
            for _ in range(2):  # Create fewer examples for non-existent roots
                question = random.choice(create_question_variations(root))
                analysis = random.choice(create_analysis_variations(root))
                
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
    
    # Shuffle the examples for better training
    random.seed(42)  # For reproducibility
    random.shuffle(examples)
    
    print(f"Final dataset size: {len(examples)} examples")
    
    # Save the dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {output_path}")
>>>>>>> 12f36a274bf141e6426259d93b106b0cab363fb3

if __name__ == "__main__":
    main()