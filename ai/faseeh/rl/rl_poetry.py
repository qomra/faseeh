import torch
import time
import threading
import socket
import signal
import requests
import regex as re
import re
import pyarabic.araby as araby
import os
import numpy as np
import math
import logging
import json
import gc
import anthropic
from urllib3.util.retry import Retry
from typing import Optional, List, Dict, Any
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM
from requests.adapters import HTTPAdapter
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from peft import LoraConfig
from peft import get_peft_model
from Levenshtein import distance as lev
from contextlib import contextmanager

# This script direcory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# rhyme2words cache file
RHYME_FILE = os.path.join(SCRIPT_DIR, "rhyme2words.json")
RHYME_INDEX = None
# Anthropic API key from environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timing out operations"""
    def signal_handler(signum, frame):
        raise TimeoutException("Operation timed out")
    
    # Set the signal handler and a alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler and cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

class RobustAnthropicClient:
    def __init__(self, api_key: str, max_retries: int = 3, timeout_seconds: int = 30):
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_seconds  # Built-in timeout
        )
        
    def complete_with_fallback(self, prompt: str, model_name: str = "claude-sonnet-4-20250514") -> str:
        """Complete prompt with robust error handling and fallbacks"""
        
        for attempt in range(self.max_retries):
            try:
                # Check internet connectivity first
                if not self._check_internet_connection():
                    logging.warning(f"No internet connection detected (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        time.sleep(5)  # Wait before retry
                        continue
                    else:
                        logging.error("No internet connection after all retries")
                        return ""
                
                # Try the API call with timeout
                with timeout(self.timeout_seconds):
                    message = self.client.messages.create(
                        model=model_name,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return message.content[0].text
                    
            except TimeoutException:
                logging.warning(f"API call timed out after {self.timeout_seconds}s (attempt {attempt + 1})")
                
            except anthropic.RateLimitError as e:
                wait_time = 60  # Wait 1 minute for rate limits
                logging.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}): {e}")
                time.sleep(wait_time)
                
            except anthropic.APIConnectionError as e:
                logging.warning(f"Connection error (attempt {attempt + 1}): {e}")
                time.sleep(5)
                
            except anthropic.APIError as e:
                logging.warning(f"API error (attempt {attempt + 1}): {e}")
                time.sleep(2)
                
            except Exception as e:
                logging.warning(f"Unexpected error (attempt {attempt + 1}): {e}")
                time.sleep(2)
        
        logging.error(f"All {self.max_retries} attempts failed for API call")
        return ""
    
    def _check_internet_connection(self, host="8.8.8.8", port=53, timeout=3):
        """Check if internet connection is available"""
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False

# Initialize robust client
robust_client = RobustAnthropicClient(ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

def anthropic_complete(prompt: str, model_name: str = "claude-sonnet-4-20250514") -> str:
    """Complete prompt using robust Anthropic client"""
    if not robust_client:
        logging.error("Anthropic client not initialized")
        return ""
    
    return robust_client.complete_with_fallback(prompt, model_name)

def claude_qafiya_weighted_reward_func_robust(prompts, completions, poem, **kwargs) -> list[float]:
    """Claude scoring with heavy emphasis on قافية (rhyme) - with robust error handling"""
    poems = [completion[0]['content'] for completion in completions]
    
    batch_size = 3
    all_rewards = []
    
    logging.info(f"🎯 Claude قافية-weighted scoring: {len(poems)} poems in batches of {batch_size}")
    
    for i in range(0, len(poems), batch_size):
        batch_poems = poems[i:i + batch_size]
        batch_targets = poem[i:i + batch_size]
        
        logging.info(f"📝 Scoring batch {i//batch_size + 1} with قافية emphasis")
        
        # Try Claude scoring with fallback
        batch_scores = batch_score_poems_with_weighted_qafiya_robust(batch_poems, batch_targets)
        
        # If Claude scoring completely failed, use local fallback
        if not batch_scores or all(score == 0.0 for score in batch_scores):
            logging.warning(f"Claude scoring failed for batch {i//batch_size + 1}, using local fallback")
            batch_scores = local_fallback_scoring(batch_poems, batch_targets)
        
        all_rewards.extend(batch_scores)
        
        # Only sleep if we got valid scores (indicating successful API calls)
        if batch_scores and any(score > 0 for score in batch_scores):
            time.sleep(1.5)
        
        avg_score = sum(batch_scores) / len(batch_scores) if batch_scores else 0
        logging.info(f"✅ Batch {i//batch_size + 1} قافية-weighted avg: {avg_score:.3f}")
    
    if all_rewards:
        final_avg = sum(all_rewards) / len(all_rewards)
        logging.info(f"🏆 Overall qافية-weighted scoring complete. Average: {final_avg:.3f}")
    
    return all_rewards

def batch_score_poems_with_weighted_qafiya_robust(poems_batch, targets_batch):
    """Score multiple poems with emphasis on قافية (rhyme) - robust version"""
    
    batch_prompt = """أنت خبير في الشعر العربي والقافية. قيم كل قصيدة مقارنة بالمستهدفة من ناحية:

1. **دقة القافية** (الأهم): هل تنتهي الأبيات بنفس القافية المطلوبة؟ (0.0 إلى 1.0)
2. **اتساق الوزن**: هل الأبيات متسقة في الوزن والإيقاع؟ (0.0 إلى 1.0)
3. **جودة اللغة**: هل اللغة فصيحة وشاعرية؟ (0.0 إلى 1.0)
4. **المعنى**: هل المعنى واضح ومناسب للسياق؟ (0.0 إلى 1.0)
5. **الالتزام بالطلب**: هل تلبي القصيدة متطلبات الطلب؟ (0.0 إلى 1.0)

**ملاحظة: القافية هي الأهم في الشعر العربي، ركز عليها بشكل خاص.**

"""

    # Add all poem pairs to the batch
    for i, (generated, target) in enumerate(zip(poems_batch, targets_batch)):
        batch_prompt += f"""
**قصيدة {i+1}:**
المستهدفة: ```{target}```
المولدة: ```{generated}```
"""

    batch_prompt += f"""
قيم كل قصيدة وأعط نقاطاً من 0.0 إلى 1.0 لكل معيار.

أعط النتائج بهذا التنسيق بالضبط لكل قصيدة:
```json
{{
  "poem_1": {{
    "قافية": 0.8,
    "وزن": 0.7,
    "لغة": 0.9,
    "معنى": 0.8,
    "التزام": 0.7,
    "تعليق": "تعليق مختصر"
  }},
  "poem_2": {{
    "قافية": 0.6,
    "وزن": 0.5,
    "لغة": 0.7,
    "معنى": 0.6,
    "التزام": 0.5,
    "تعليق": "تعليق مختصر"
  }}{f',' if len(poems_batch) > 2 else ''}
  {"... وهكذا لباقي القصائد" if len(poems_batch) > 2 else ""}
}}
```"""

    try:
        # Use robust completion
        response = anthropic_complete(batch_prompt)
        
        if not response:  # Empty response indicates failure
            logging.warning("Empty response from Claude API")
            return [0.0] * len(poems_batch)
        
        # Extract JSON from response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)
            
            # Calculate weighted scores with emphasis on قافية
            weighted_scores = []
            detailed_results = []
            
            for i in range(len(poems_batch)):
                poem_key = f"poem_{i+1}"
                if poem_key in result:
                    poem_data = result[poem_key]
                    
                    # Extract individual scores
                    qafiya = float(poem_data.get("قافية", 0.0))
                    wazan = float(poem_data.get("وزن", 0.0))
                    lugha = float(poem_data.get("لغة", 0.0))
                    maana = float(poem_data.get("معنى", 0.0))
                    iltizam = float(poem_data.get("التزام", 0.0))
                    
                    # WEIGHTED CALCULATION - EMPHASIZE QAFIYA
                    weighted_score = (
                        qafiya * 0.50 +    # 50% weight for rhyme
                        wazan * 0.20 +     # 20% weight for meter
                        lugha * 0.15 +     # 15% weight for language
                        maana * 0.10 +     # 10% weight for meaning
                        iltizam * 0.05     # 5% weight for compliance
                    )
                    
                    weighted_scores.append(weighted_score)
                    
                    # Store detailed breakdown for logging
                    detailed_results.append({
                        'poem_index': i+1,
                        'قافية': qafiya,
                        'وزن': wazan,
                        'لغة': lugha,
                        'معنى': maana,
                        'التزام': iltizam,
                        'weighted_score': weighted_score,
                        'تعليق': poem_data.get("تعليق", "")
                    })
                else:
                    weighted_scores.append(0.0)
                    detailed_results.append({
                        'poem_index': i+1,
                        'error': 'No data found for this poem'
                    })
            
            # Log detailed results with weighting info
            logging.info("📊 Weighted Claude Scoring Results (قافية=50%, وزن=20%, لغة=15%, معنى=10%, التزام=5%):")
            for result in detailed_results:
                if 'error' not in result:
                    logging.info(f"  قصيدة {result['poem_index']}: "
                               f"قافية={result['قافية']:.2f}(50%), وزن={result['وزن']:.2f}(20%), "
                               f"لغة={result['لغة']:.2f}(15%), معنى={result['معنى']:.2f}(10%), "
                               f"التزام={result['التزام']:.2f}(5%) → مرجح={result['weighted_score']:.3f}")
                    if result['تعليق']:
                        logging.info(f"    تعليق: {result['تعليق']}")
                else:
                    logging.warning(f"  قصيدة {result['poem_index']}: {result['error']}")
            
            if len(weighted_scores) == len(poems_batch):
                return weighted_scores
            
        # Fallback strategies
        logging.warning("JSON parsing failed, using local fallback...")
        return local_fallback_scoring(poems_batch, targets_batch)
        
    except Exception as e:
        logging.error(f"Error in weighted batch scoring: {e}")
        return local_fallback_scoring(poems_batch, targets_batch)

def local_fallback_scoring(poems_batch, targets_batch):
    """Local fallback scoring when Claude API is unavailable"""
    logging.info("🔄 Using local fallback scoring")
    
    scores = []
    for generated, target in zip(poems_batch, targets_batch):
        score = 0.0
        
        # Basic structure scoring
        gen_lines = [l.strip() for l in generated.split('\n') if l.strip()]
        target_lines = [l.strip() for l in target.split('\n') if l.strip()]
        
        if len(gen_lines) >= 2:
            score += 0.2  # Structure bonus
            
            # Simple rhyme detection
            if len(gen_lines) >= 2:
                endings = []
                for line in gen_lines[:4]:  # Check first 4 lines
                    words = line.split()
                    if words:
                        ending = words[-1][-2:] if len(words[-1]) >= 2 else words[-1]
                        endings.append(ending)
                
                # Check for rhyme patterns
                unique_endings = set(endings)
                if len(unique_endings) < len(endings):
                    score += 0.3  # Rhyme bonus
            
            # Length similarity
            if target_lines:
                target_avg_len = sum(len(line.split()) for line in target_lines) / len(target_lines)
                gen_avg_len = sum(len(line.split()) for line in gen_lines) / len(gen_lines)
                length_similarity = 1.0 - abs(target_avg_len - gen_avg_len) / max(target_avg_len, 1)
                score += length_similarity * 0.2
        
        scores.append(min(1.0, score))  # Cap at 1.0
    
    avg_score = sum(scores) / len(scores) if scores else 0
    logging.info(f"📊 Local fallback scoring average: {avg_score:.3f}")
    return scores

def calculate_structure_score(generated_poem, target_poem):
    """Calculate structural quality score"""
    gen_lines = [line.strip() for line in generated_poem.split('\n') if line.strip()]
    target_lines = [line.strip() for line in target_poem.split('\n') if line.strip()]
    
    if not gen_lines:
        return 0.0
    
    # Reward correct number of lines
    line_count_score = 1.0 if len(gen_lines) == len(target_lines) else 0.5
    
    # Reward proper Arabic poetry format (each line has ... separator)
    format_score = 0.0
    if len(gen_lines) >= 2:
        has_separators = sum(1 for line in gen_lines if '...' in line or '…' in line)
        format_score = has_separators / len(gen_lines)
    
    # Reward reasonable line lengths
    length_score = 0.0
    if gen_lines:
        avg_length = sum(len(line.split()) for line in gen_lines) / len(gen_lines)
        length_score = 1.0 if 5 <= avg_length <= 15 else 0.5
    
    return (line_count_score * 0.4) + (format_score * 0.3) + (length_score * 0.3)

def structure_reward_func(prompts, completions, poem, **kwargs) -> list[float]:
    """Reward proper poem structure"""
    rewards = []
    
    for completion, target_poem in zip(completions, poem):
        response = completion[0]['content']
        reward = calculate_structure_score(response, target_poem)
        rewards.append(reward)
    
    return rewards

def enhanced_explicit_poetry_reward_func(prompts, completions, poem, **kwargs) -> list[float]:
    """Enhanced version of explicit poetry reward with better scoring"""
    rewards = []
    
    for completion, target in zip(completions, poem):
        text = completion[0]['content'].strip()
        reward = 0.0
        
        # Check for basic poetry structure
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        if len(lines) >= 2:
            reward += 0.3  # Increased bonus for multiple lines
            
            # Enhanced rhyme detection
            endings = []
            for line in lines:
                words = line.split()
                if words:
                    # Get last 3 characters for better Arabic rhyme detection
                    clean_word = araby.strip_diacritics(words[-1]) if hasattr(araby, 'strip_diacritics') else words[-1]
                    ending = clean_word[-3:] if len(clean_word) >= 3 else clean_word
                    endings.append(ending)
            
            # Reward rhyme patterns
            if len(endings) >= 2:
                unique_endings = set(endings)
                rhyme_ratio = 1.0 - (len(unique_endings) / len(endings))
                reward += rhyme_ratio * 0.4  # Higher reward for rhyme
            
            # Check for Arabic poetry markers (enhanced list)
            text_lower = text.lower()
            poetry_markers = [
                '...', '…',           # Classical separators
                'قال', 'يقول',        # Poetry verbs
                'يا ', 'أيا ',        # Vocative particles
                'لا ', 'ما ',         # Negation
                'قد ', 'لقد',         # Emphasis
                'من ', 'في ', 'على ', 'إذا ', 'كان ', 'إن '
            ]
            marker_count = sum(1 for marker in poetry_markers if marker in text_lower)
            reward += min(0.3, marker_count * 0.05)
        
        # Penalize very short or very long responses
        word_count = len(text.split())
        if 10 <= word_count <= 100:  # Reasonable poetry length
            reward += 0.1
        
        rewards.append(min(1.0, reward))  # Cap at 1.0
    
    return rewards

def last_cluster(word: str) -> str:
    """
    Return final consonant + long vowel/alif/ya + tanwin/sukun if present.
    Very loose; good enough for reward shaping.
    """
    w = araby.strip_diacritics(word)
    # remove tatweel and punctuation
    w = re.sub(r"[ـ\W]+", "", w)
    if len(w) <= 2:
        return w
    # find last consonant – very naïve
    for i in range(len(w)-1, -1, -1):
        if w[i] not in "aeiouى":
            return w[i:]
    return w[-2:]

def rhyme_entropy(clusters):
    """
    0 → perfect single rhyme, ln(N) → all different.
    We invert and normalise to [0,1].
    """
    from collections import Counter
    cnt = Counter(clusters)
    probs = np.array(list(cnt.values()), dtype=float) / len(clusters)
    H = -(probs * np.log(probs)).sum()            # nats
    if len(clusters) == 1:
        return 1.0
    return 1 - H / np.log(len(clusters))          # 1 = perfect, 0 = all diff.

def smoother_rhyme_reward(prompts, completions, poem, **kw):
    rewards = []
    for compl, tgt in zip(completions, poem):
        lines = [l.strip() for l in compl[0]["content"].split("\n") if l.strip()]
        if len(lines) < 2:           # not a poem → reward 0
            rewards.append(0.0)
            continue

        clusters = [last_cluster(line.split()[-1]) for line in lines]
        ent_score = rhyme_entropy(clusters)           # 0–1

        # soft closeness to gold rhyme (0 = identical, 1+ far)
        tgt_cluster = last_cluster(tgt.split()[-1])
        lev_dist = min(lev(c, tgt_cluster) for c in clusters)
        gold_score = max(0.0, 1 - lev_dist / 3)       # taper after edit-3

        # combine: 0.7 weight on internal consistency, 0.3 on gold match
        reward = 0.7 * ent_score + 0.3 * gold_score
        rewards.append(reward)
    return rewards

def _load_rhyme_index() -> Dict[str, List[str]]:
    """Cache + return {rhyme_ending -> [valid words …]}"""
    global RHYME_INDEX
    if RHYME_INDEX is not None:
        return RHYME_INDEX

    if not os.path.isfile(RHYME_FILE):
        logging.error("⚠️  rhyme index not found: %s", RHYME_FILE)
        RHYME_INDEX = {}
        return RHYME_INDEX

    with open(RHYME_FILE, "r", encoding="utf-8") as f:
        RHYME_INDEX = json.load(f)
    logging.info("✅ loaded %d rhyme groups", len(RHYME_INDEX))
    return RHYME_INDEX

def _strip_diacritics(text: str) -> str:
    """Loose, fast Arabic tashkeel/haraka stripping (no external dep)."""
    return re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)

def _extract_target_rhyme(target: str) -> str:
    """
    Detect the rhyme of the reference poem:
      – remove tashkeel
      – take last up-to-3 chars of every non-empty line
      – return the longest suffix that is identical across *all* lines,
        else fall back to the longest suffix shared by 1st & 2nd line,
        else the final character of line-1.
    """
    lines = [l.strip() for l in _strip_diacritics(target).split("\n") if l.strip()]
    if len(lines) == 0:
        return ""                       # should never happen
    if len(lines) == 1:
        return lines[0][-1]             # trivial case

    def _common_suffix(a: str, b: str, k: int) -> str:
        return a[-k:] if a[-k:] == b[-k:] else ""

    # try length 3->2->1 across ALL lines
    for k in (3, 2, 1):
        suffix = lines[0][-k:]
        if all(line.endswith(suffix) for line in lines):
            return suffix

    # otherwise try first two lines only
    for k in (3, 2, 1):
        sfx = _common_suffix(lines[0], lines[1], k)
        if sfx:
            return sfx

    return lines[0][-1]

def rhyme_dictionary_reward(
    prompts, completions, poem, **kwargs
) -> list[float]:
    """
    + detects the correct rhyme from *poem*         (ground truth)
    + checks that ***every*** generated line ends with that rhyme
    + checks that the last word of each line occurs in our lexicon
    Returns a scalar in [0, 1] for each sample.
    """
    rhyme_index = _load_rhyme_index()
    rewards: list[float] = []

    for comp, reference in zip(completions, poem):
        gen_text = comp[0]["content"].strip()
        if not gen_text:
            rewards.append(0.0)
            continue

        # ------------------------------------------------------------------ #
        # Pre-processing
        lines = [
            l.strip() for l in _strip_diacritics(gen_text).split("\n") if l.strip()
        ]
        if len(lines) < 2:               # need at least a couplet
            rewards.append(0.0)
            continue

        target_rhyme = _extract_target_rhyme(reference)
        rhyme_len    = len(target_rhyme) if target_rhyme else 1

        # ------------------------------------------------------------------ #
        # Per-line checks
        total          = len(lines)
        rhyme_hits     = 0               # line ends with correct rhyme
        lexicon_hits   = 0               # last word ∈ allowed list
        endings        = []              # for self-consistency calc

        for line in lines:
            words = line.split()
            if not words:
                continue
            last_word = words[-1]

            endings.append(last_word[-rhyme_len:])

            # 1) rhyme correctness
            if last_word.endswith(target_rhyme):
                rhyme_hits += 1

            # 2) word in lexicon
            if target_rhyme in rhyme_index and last_word in rhyme_index[target_rhyme]:
                lexicon_hits += 1

        # 3) rhyme consistency across the whole poem
        most_common = max(set(endings), key=endings.count)
        consistency = endings.count(most_common) / total

        # ------------------------------------------------------------------ #
        # Merge into a single reward  (all weights sum to 1.0)
        w_rhyme       = 0.45         # uses correct ending
        w_lexicon     = 0.40         # uses valid vocab
        w_consistency = 0.15         # same ending everywhere

        score = (
            w_rhyme       * (rhyme_hits     / total) +
            w_lexicon     * (lexicon_hits   / total) +
            w_consistency * consistency
        )

        rewards.append(round(float(score), 4))

    return rewards

def setup_model_for_training(model_path_or_model, tokenizer, use_lora=True, from_lora_checkpoint=True):
    """
    Properly set up model for GRPO training with gradient support and numerical stability
    Supports both:
    - Creating new LoRA adapters on base model (use_lora=True, from_lora_checkpoint=False)
    - Loading existing LoRA checkpoint (use_lora=True, from_lora_checkpoint=True)
    - Full model training (use_lora=False)
    """
    logging.info("Setting up model for training with numerical stability fixes...")

    if from_lora_checkpoint:
        logging.info(f"🔄 Loading existing LoRA checkpoint from: {model_path_or_model}")
        
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path_or_model,
                torch_dtype=torch.bfloat16,  # Keep original dtype to match base model
                device_map="auto",
                trust_remote_code=True,
                is_trainable=True,
                low_cpu_mem_usage=True,
                # Add safety parameters
                attn_implementation="eager",  # Use eager attention for stability
            )
            logging.info("✅ Loaded LoRA model using AutoPeftModelForCausalLM")

            # CRITICAL: Properly configure caching and gradient checkpointing
            model.config.use_cache = False
            
            # Handle base model configuration
            if hasattr(model, 'base_model'):
                if hasattr(model.base_model, 'config'):
                    model.base_model.config.use_cache = False
                
                # Enable gradient checkpointing on base model
                if hasattr(model.base_model, 'gradient_checkpointing_enable'):
                    model.base_model.gradient_checkpointing_enable()
                    logging.info("✅ Enabled gradient checkpointing on base model")
            
            # Enable gradient checkpointing on PEFT model
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logging.info("✅ Enabled gradient checkpointing on PEFT model")

            # CRITICAL: Verify and fix gradient requirements
            peft_params_count = 0
            base_params_count = 0
            
            for name, param in model.named_parameters():
                if 'lora_' in name or 'adapter' in name:
                    param.requires_grad = True
                    peft_params_count += 1
                else:
                    param.requires_grad = False
                    base_params_count += 1
            
            logging.info(f"✅ Set gradients: {peft_params_count} LoRA params trainable, {base_params_count} base params frozen")

            # Log trainable parameters using PEFT's method
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
            else:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                logging.info(f"Trainable parameters: {trainable_params} || all params: {total_params} || trainable%: {trainable_params / total_params * 100:.4f}")

        except Exception as e:
            logging.error(f"❌ Error loading LoRA checkpoint: {e}")
            raise RuntimeError(f"Failed to load LoRA checkpoint from {model_path_or_model}: {e}")

    elif use_lora:  # Create new LoRA adapters on base model
        logging.info(f"🔄 Loading base model and creating new LoRA adapters: {model_path_or_model}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path_or_model,
                torch_dtype=torch.float32,  # Use fp32 for stability
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            logging.info("✅ Loaded base model")

            # Configure model for training
            model.config.use_cache = False
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

            # Create LoRA configuration
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj",
                                "gate_proj","up_proj","down_proj"],
                layers_to_transform=[0,1],     # NEW → limit to the first two blocks
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            
            # Apply LoRA to the model
            model = get_peft_model(model, lora_config)
            logging.info("✅ Applied LoRA configuration to base model")
            
            # Print trainable parameters
            model.print_trainable_parameters()

        except Exception as e:
            logging.error(f"❌ Error creating LoRA model: {e}")
            raise RuntimeError(f"Failed to create LoRA model from {model_path_or_model}: {e}")

    else:  # Full model training (no LoRA)
        logging.info(f"🔄 Loading full model for complete fine-tuning: {model_path_or_model}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path_or_model,
                torch_dtype=torch.float32,  # Use fp32 for stability
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            logging.info("✅ Loaded full model")
            
            # Configure for full training
            model.config.use_cache = False
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Enable gradients for all parameters
            for param in model.parameters():
                param.requires_grad = True
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"🔥 Full model training: {trainable_params} trainable params ({trainable_params / total_params * 100:.2f}%)")

        except Exception as e:
            logging.error(f"❌ Error loading full model: {e}")
            raise RuntimeError(f"Failed to load full model from {model_path_or_model}: {e}")

    # CRITICAL: Set model to training mode
    model.train()
    
    # CRITICAL: Numerical stability checks
    logging.info("🔍 Performing numerical stability checks...")
    
    nan_params = []
    inf_params = []
    zero_grad_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check for NaN values
            if torch.isnan(param).any():
                nan_params.append(name)
            
            # Check for infinite values
            if torch.isinf(param).any():
                inf_params.append(name)
            
            # Check if gradients are properly set up (will be None initially)
            if param.grad is not None and torch.isnan(param.grad).any():
                zero_grad_params.append(name)
    
    if nan_params:
        logging.error(f"❌ Found NaN values in parameters: {nan_params[:5]}")
        raise RuntimeError(f"Model has NaN parameters: {nan_params[:5]}")
    
    if inf_params:
        logging.error(f"❌ Found infinite values in parameters: {inf_params[:5]}")
        raise RuntimeError(f"Model has infinite parameters: {inf_params[:5]}")
    
    # Final verification of trainable parameters
    trainable_params_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params_count = len(trainable_params_names)
    
    logging.info(f"✅ Trainable parameters (final verification): {trainable_params_count}")
    
    if trainable_params_count == 0:
        logging.error("❌ No trainable parameters found after setup!")
        # Debug: Print all parameter names and their requires_grad status
        for name, param in model.named_parameters():
            logging.error(f"  {name}: requires_grad={param.requires_grad}")
        raise RuntimeError("No trainable parameters found after setup!")
    
    # Log sample trainable parameter names
    sample_params = trainable_params_names[:5]
    logging.info(f"Sample trainable params: {sample_params}")
    
    # CRITICAL: Move model to GPU if available and verify
    if torch.cuda.is_available():
        # Model should already be on GPU due to device_map="auto", but verify
        device = next(model.parameters()).device
        logging.info(f"✅ Model is on device: {device}")
        
        # Clear CUDA cache for clean start
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check CUDA memory
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(f"📊 CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    # CRITICAL: Test forward pass for numerical stability
    logging.info("🧪 Testing forward pass for numerical stability...")
    
    model.eval()  # Temporarily set to eval for testing
    
    with torch.no_grad():
        try:
            # Create a simple test input
            test_text = "Test input"
            test_input = tokenizer.encode(test_text, return_tensors="pt", max_length=50, truncation=True)
            
            # Move input to same device as model
            if torch.cuda.is_available():
                test_input = test_input.to(next(model.parameters()).device)
            
            # Forward pass
            test_output = model(test_input)
            
            # Check output for numerical issues
            if hasattr(test_output, 'logits'):
                logits = test_output.logits
                
                if torch.isnan(logits).any():
                    raise RuntimeError("Forward pass produces NaN logits")
                
                if torch.isinf(logits).any():
                    raise RuntimeError("Forward pass produces infinite logits")
                
                # Check if logits are in reasonable range
                logits_max = torch.max(logits).item()
                logits_min = torch.min(logits).item()
                
                if abs(logits_max) > 1e6 or abs(logits_min) > 1e6:
                    logging.warning(f"⚠️ Large logits detected: max={logits_max:.2e}, min={logits_min:.2e}")
                
                logging.info(f"✅ Forward pass successful - logits range: [{logits_min:.3f}, {logits_max:.3f}]")
            
        except Exception as e:
            logging.error(f"❌ Forward pass test failed: {e}")
            raise RuntimeError(f"Model forward pass is unstable: {e}")
    
    # Set back to training mode
    model.train()
    
    # CRITICAL: Verify model configuration
    if hasattr(model, 'config'):
        config_checks = {
            'use_cache': model.config.use_cache,
            'torch_dtype': getattr(model.config, 'torch_dtype', 'not_set'),
        }
        logging.info(f"📋 Model config verification: {config_checks}")
        
        if model.config.use_cache:
            logging.warning("⚠️ use_cache is True - this may cause issues with gradient checkpointing")
    
    # Final memory and parameter summary
    if torch.cuda.is_available():
        final_allocated = torch.cuda.memory_allocated() / 1e9
        final_reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(f"📊 Final CUDA Memory - Allocated: {final_allocated:.2f}GB, Reserved: {final_reserved:.2f}GB")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"📈 Final Parameter Summary:")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params:,}")
    logging.info(f"  Trainable percentage: {trainable_params / total_params * 100:.4f}%")
    
    logging.info("✅ Model setup complete - ready for GRPO training")
    return model

step_counter = 0          # keep outside so all batches share it
class FaseehPoetryGRPOTrainer:

    def __init__(self, tokenizer, base_model, output_dir, use_lora=True, **kwargs):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.use_lora = use_lora
        # CRITICAL FIX 1: Add padding token if missing
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logging.info(f"✅ Set pad_token to eos_token: {self.tokenizer.eos_token}")
            else:
                # Fallback to a safe token
                self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                logging.info("✅ Added new pad_token: <|pad|>")
        
        # Set up chat template for GRPO compatibility
        self.setup_chat_template()
        
        # Properly set up the model for training
        self.model = setup_model_for_training(base_model, tokenizer, use_lora=use_lora)
        
        # CRITICAL FIX 2: More conservative GRPO config to prevent numerical issues
        from trl import GRPOConfig
        self.training_args = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=8,      # AGGRESSIVE: Even higher batch size
            gradient_accumulation_steps=2,      # REDUCED: Still effective batch of 16
            optim="adamw_torch",
            save_steps=50,
            save_total_limit=3,
            logging_steps=1,
            learning_rate=1e-5,              # REDUCED: Much lower learning rate
            warmup_steps=5,                  # INCREASED: More warmup
            lr_scheduler_type="constant",    # Keep constant for stability
            bf16=False,                      # DISABLED: Use fp32 for stability
            fp16=False,                      # DISABLED: Use fp32 for stability
            dataloader_num_workers=0,
            gradient_checkpointing=True,
            max_completion_length=400,       # REDUCED: Shorter completions
            temperature=0.6,                 # INCREASED: Higher temperature for stability
            max_prompt_length=400,           # REDUCED: Shorter prompts
            num_generations=Z,               # REDUCED: Fewer generations per step
            remove_unused_columns=False,
            max_grad_norm=0.5,               # REDUCED: More aggressive gradient clipping
            dataloader_pin_memory=False,
            beta       = 0.005      #  ← sets the initial KL weight
        )
    
    def setup_chat_template(self):
        """Set up chat template for GRPO compatibility"""
        chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'user' %}"
            "<|start_header_id|>user<|end_header_id|>{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'assistant' %}"
            "<|start_header_id|>assistant<|end_header_id|>{{ message['content'] }}<|eot_id|>"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>"
            "{% endif %}"
        )
        
        self.tokenizer.chat_template = chat_template
        logging.info("✅ Chat template set for GRPO compatibility")

    def format_dataset(self, dataset):
        """Format dataset for poetry GRPO training"""
        def format_example(x):
            if 'conversation' in x:
                conversation = x['conversation']
                system_msg = conversation[0]['content'].strip()
                user_msg = conversation[1]['content'].strip()  
                assistant_msg = conversation[2]['content'].strip()
                
                return {
                    'prompt': [
                        {'role': 'system', 'content': system_msg},
                        {'role': 'user', 'content': user_msg}
                    ],
                    'poem': assistant_msg
                }
            else:
                return {
                    'prompt': [
                        {'role': 'system', 'content': 'أنت شاعر متخصص في القافية والوزن. يجب أن تلتزم بالقافية المطلوبة في كل بيت.'},
                        {'role': 'user', 'content': x['prompt']}
                    ],
                    'poem': x['poem']
                }
        
        return dataset.map(format_example)

    def train(self, dataset):

        ######################################################################
        # 0.  house-keeping
        ######################################################################
        logging.info("▶️  GRPO poetry training (no Anthropic)…")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        formatted_ds = self.format_dataset(dataset)

        trainable = sum(p.requires_grad for p in self.model.parameters())
        if trainable == 0:
            raise RuntimeError("⚠️  model has no trainable tensors")
        logging.info(f"🔍 trainable tensors: {trainable}")

        ######################################################################
        # 1.  curriculum constants
        ######################################################################
        PHASE_A_STEPS = 150         # warm-up on rhyme only
        WEIGHTS = {
            #          rhyme  struct  explicit
            "A":   (1.0,   0.0,    0.0),
            "B":   (0.6,   0.25,   0.15),
        }

        def zscore(x, eps=1e-8, clip=3.0):
            x = np.asarray(x, np.float32)
            std = x.std()
            if std < eps:
                return np.zeros_like(x)
            return np.clip((x - x.mean()) / (std + eps), -clip, clip).tolist()

        ######################################################################
        # 2.  wrapped reward
        ######################################################################
        self.global_step = 0

        def combined_reward(prompts, comps, targets, **kw):
            """Phase-aware combination of local rewards."""
            self.global_step += 1
            phase = "A" if self.global_step < PHASE_A_STEPS else "B"
            w_rhyme, w_struct, w_exp = WEIGHTS[phase]

            rhyme_raw   = rhyme_dictionary_reward(prompts, comps, targets)
            rhyme_z     = zscore(rhyme_raw)

            # light extras only after warm-up
            if phase == "A":
                struct_z = exp_z = [0.0]*len(rhyme_raw)
            else:
                struct_z = zscore(structure_reward_func(prompts, comps, targets))
                exp_z    = zscore(enhanced_explicit_poetry_reward_func(
                                    prompts, comps, targets))

            # linear mix
            rewards = [
                w_rhyme  * r +
                w_struct * s +
                w_exp    * e
                for r, s, e in zip(rhyme_z, struct_z, exp_z)
            ]

            if self.global_step % 20 == 0:
                logging.info(
                    f"step={self.global_step:4d}│{phase}"
                    f"│μ(rhyme_raw)={np.mean(rhyme_raw):.3f}"
                    f"│μ(total)={np.mean(rewards):.3f}"
                )
            return rewards
        ######################################################################
        # 3.  create trainer and launch
        ######################################################################
        trainer = GRPOTrainer(
            model            = self.model,
            reward_funcs     = [combined_reward],
            args             = self.training_args,
            train_dataset    = formatted_ds,
            processing_class = self.tokenizer,
        )

        logging.info("🚀  starting optimisation loop …")
        try:
            trainer.train()
            logging.info(f"✅ finished – model saved to {self.output_dir}")
        except Exception:
            logging.exception("❌ training crashed")
            raise