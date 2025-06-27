from pathlib import Path
import json, re, unicodedata
import json
from tqdm import tqdm
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LEXICON_PATH = f"{SCRIPT_DIR}/dataset.json"  # Arabic lexicon file
CACHE_FILE   = "rhyme2words.json"            # pickled/JSON cache

def strip_diacritics(text):
    return ''.join(ch for ch in unicodedata.normalize('NFD', text)
                   if unicodedata.category(ch) != 'Mn')

def normalise(word):
    word = strip_diacritics(word)
    return word

def build_rhyme_index():
    rhyme2words = {1: {}, 2: {}, 3: {}}
    """
    Get all words in {
        "لسان العرب" : {
            "شخز" : [
            "الشَّخْسِ",
            "الشَّخْزُ",
            ],
            "هندم" : [
            "الهِندامُ"
            ],
            "شهرق" : [
            "شَهْرقُ",
            "الشَّهْرَقا",
            "الشَّهْرقُ"
            ],
            .
            .
        }
    """
    print("Building rhyme index from:", LEXICON_PATH)
    if not Path(LEXICON_PATH).exists():
        raise FileNotFoundError(f"Lexicon file not found: {LEXICON_PATH}")
    if Path(CACHE_FILE).exists():
        print("Cache file already exists, skipping index build:", CACHE_FILE)
        return  
    with open(LEXICON_PATH, 'r', encoding='utf8') as f:
        lexicon = json.load(f)["لسان العرب"]
    
    print("Processing words...")
    for root, words in tqdm(lexicon.items(), desc="Indexing words"):
        for w in words:
            w = normalise(w.strip())
            if not w: 
                continue
            for k in (1, 2, 3):
                if len(w) >= k:
                    key = w[-k:]
                    if key not in rhyme2words[k]:
                        rhyme2words[k][key] = []
                    rhyme2words[k][key].append(w)
    # remove duplicates
    for k in (1, 2, 3):
        for key in rhyme2words[k]:
            rhyme2words[k][key] = list(set(rhyme2words[k][key]))
    # Write the rhyme2words index to a JSON file
    with open(CACHE_FILE, 'w', encoding='utf8') as f:
        json.dump(rhyme2words, f, ensure_ascii=False, indent=4)
    print("✅ rhyme index cached:", CACHE_FILE)

# run once:
build_rhyme_index()
