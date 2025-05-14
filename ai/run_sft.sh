
project=oryx-2.0-allam
python run.py projects/$project/sft_lora.yml 
python run.py projects/$project/generate_sft_lora.yml 
python projects/$project/data/extract.py projects/$project/data/generation_sft_lora.jsonl projects/$project/data/generation_sft_lora.rtl.txt
