import sys
import os
import math
import numpy as np
import pandas as pd
import torch
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QListWidget, QTextEdit, QPushButton, 
                            QLabel, QComboBox, QFileDialog, QProgressBar, 
                            QSplitter, QGridLayout, QGroupBox, QMessageBox,
                            QSpinBox, QRadioButton, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QTextCharFormat, QFont, QTextCursor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import colorsys

def load_datasets_from_hf(questions_dataset):
    """
    Load datasets from HuggingFace.
    
    Args:
        questions_dataset: HuggingFace dataset name for questions
        
    Returns:
        DataFrame with all required data
    """
    # Load questions dataset
    from maknaz import pull
    try:
        questions_data = pull(questions_dataset)
        if 'train' in questions_data:
            questions_df = pd.DataFrame(questions_data['train'])
        else:
            # Use first split if train isn't available
            first_split = list(questions_data.keys())[0]
            questions_df = pd.DataFrame(questions_data[first_split])
    except Exception as e:
        print(f"Error loading questions dataset: {e}")
        raise
        
    
    return questions_df

def clear_gpu_memory():
    """Clear CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class SentenceRetriever:
    """
    A class that handles sentence-level retrieval across multiple contexts.
    """
    def __init__(self, retriever_model, retriever_tokenizer, answerer_model, answerer_tokenizer, 
                 retriever_device, answerer_device, max_seq_len=4096):
        self.retriever_model = retriever_model
        self.retriever_tokenizer = retriever_tokenizer
        self.answerer_model = answerer_model
        self.answerer_tokenizer = answerer_tokenizer
        self.retriever_device = retriever_device
        self.answerer_device = answerer_device
        self.max_seq_len = max_seq_len
        self.effective_seq_len = max_seq_len - 500  # Reserve tokens for prompt

    def preprocess_contexts(self, contexts_data):
        """
        Extract and clean sentences from all contexts, maintaining their root associations.
        
        Args:
            contexts_data: List of dicts with 'root', 'context', 'is_correct' keys
            
        Returns:
            List of sentence dicts with 'sentence', 'root', 'is_correct' keys
        """
        all_sentences = []
        
        # Process each context
        for context_info in contexts_data:
            root = context_info['root']
            context = context_info['context']
            is_correct = context_info.get('is_correct', False)
            original_root = context_info.get('original_root', root)
            
            print(f"Processing context from root '{root}', {len(context)} chars")
            
            # Check for empty context
            if not context or len(context) < 10:
                print(f"Skipping empty or very short context for root '{root}'")
                continue
            
            # Split the context into sentences using multiple delimiters
            # This approach works better for Arabic text
            raw_sentences = []
            
            # Try splitting by various punctuation marks
            # delimiters = ['.']#, '!', '؟', '?', '؛', ';', '،', ',']
            # current_sentence = ""
            
            # for char in context:
            #     current_sentence += char
            #     if char in delimiters:
            #         # We've reached the end of a sentence
            #         if len(current_sentence.strip()) > 0:
            #             raw_sentences.append(current_sentence.strip())
            #             current_sentence = ""
            raw_sentences = context.split(".")
            
            # # Add the last sentence if there's anything left
            # if len(current_sentence.strip()) > 0:
            #     raw_sentences.append(current_sentence.strip())
            
            # If we couldn't split into sentences, treat the whole context as one sentence
            if not raw_sentences:
                raw_sentences = [context]
            
            # Add each sentence to our collection with its metadata
            for i,sentence in enumerate(raw_sentences):
                sentence = sentence.strip()
                all_sentences.append({
                    'sentence': sentence,
                    'detailed_sentence': f"الكتاب: لسان العرب\nالجذر: {root}\n{sentence}",
                    'root': root,
                    'original_root': original_root,
                    'is_correct': is_correct,
                    'original_index': i
                })
        
            print(f"Extracted {len(raw_sentences)} sentences from root '{root}'")
        
        print(f"Total: Extracted {len(all_sentences)} sentences from {len(contexts_data)} contexts")
        return all_sentences        

    def create_sentence_chunks(self, sentences, max_tokens_per_chunk):
        """
        Group sentences into chunks that fit within the token limit,
        ensuring no sentences are split across chunks.
        
        Args:
            sentences: List of sentence dicts
            max_tokens_per_chunk: Maximum tokens per chunk
            
        Returns:
            List of chunk dicts with 'sentences', 'tokens', 'text' keys
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence_info in sentences:
            sentence = sentence_info['detailed_sentence']
            tokens = self.retriever_tokenizer.encode(sentence, add_special_tokens=False)
            
            # If this sentence would push us over the limit, start a new chunk
            if current_tokens + len(tokens) > max_tokens_per_chunk and current_chunk:
                # Create a chunk from the current sentences
                chunk_text = ".\n".join([s['detailed_sentence'] for s in current_chunk])
                chunks.append({
                    'sentences': current_chunk,
                    'num_tokens': current_tokens,
                    'text': chunk_text
                })
                # Start a new chunk
                current_chunk = []
                current_tokens = 0
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence_info)
            current_tokens += len(tokens)
        
        # Add the last chunk if it has any sentences
        if current_chunk:
            chunk_text = ".\n".join([s['detailed_sentence'] for s in current_chunk])
            chunks.append({
                'sentences': current_chunk,
                'num_tokens': current_tokens,
                'text': chunk_text
            })
        
        print(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk['num_tokens']} tokens, {len(chunk['sentences'])} sentences")
            
        return chunks

    def find_sentence_positions_fuzzy(self, all_tokens, sentence_infos):
        """
        Find positions of sentences in tokenized text using fuzzy matching.
        
        Args:
            all_tokens: List of all tokens from the model tokenizer
            sentence_infos: List of sentence info dictionaries
            
        Returns:
            List of tuples (start_idx, end_idx, sentence_info)
        """
        # Convert token list to strings for easier processing
        all_token_strings = [str(t) for t in all_tokens]
        full_text = self.retriever_tokenizer.decode(
            self.retriever_tokenizer.convert_tokens_to_ids(all_tokens)
        )
        
        sentence_token_positions = []
        
        for i, sentence_info in enumerate(sentence_infos):
            sentence = sentence_info['detailed_sentence']
            
            # Skip very long sentences or empty ones
            if len(sentence) < 2 or len(sentence) > 1000:
                print(f"Skipping very short or long sentence {i+1}: {sentence[:20]}...")
                continue
            
            try:
                # Method 1: Try direct substring search in the full text
                sentence_start_char = full_text.find(sentence)
                
                if sentence_start_char >= 0:
                    # Found exact match, now map character positions to token positions
                    sentence_end_char = sentence_start_char + len(sentence)
                    
                    # Get tokens that correspond to these character positions
                    # This is an approximation since character-to-token mapping isn't perfect
                    prefix_text = full_text[:sentence_start_char]
                    prefix_tokens = len(self.retriever_tokenizer.encode(prefix_text, add_special_tokens=False))
                    sentence_tokens = len(self.retriever_tokenizer.encode(sentence, add_special_tokens=False))
                    
                    # Adjust for possible tokenization differences
                    start_idx = max(0, prefix_tokens - 2)  # Allow for small offset
                    end_idx = min(len(all_tokens), start_idx + sentence_tokens + 4)  # Allow for small offset
                    
                    sentence_token_positions.append((start_idx, end_idx, sentence_info))
                    continue
                
                # Method 2: Try finding key n-grams from the sentence
                # This works better for sentences where exact match fails
                sentence_tokens = self.retriever_tokenizer.encode(sentence, add_special_tokens=False)
                sentence_token_strings = self.retriever_tokenizer.convert_ids_to_tokens(sentence_tokens)
                
                # For very long sentences, use the first and last few tokens as anchors
                if len(sentence_tokens) > 10:
                    # Use first 5 tokens as beginning n-gram
                    begin_ngram = sentence_token_strings[:5]
                    # Use last 5 tokens as ending n-gram
                    end_ngram = sentence_token_strings[-5:]
                    
                    # Find beginning n-gram
                    begin_pos = -1
                    for pos in range(len(all_token_strings) - 5):
                        match_count = sum(1 for j in range(5) if pos+j < len(all_token_strings) and 
                                        all_token_strings[pos+j] == begin_ngram[j])
                        # Allow for partial matches (at least 3 out of 5 tokens match)
                        if match_count >= 3:
                            begin_pos = pos
                            break
                    
                    # Find ending n-gram
                    end_pos = -1
                    for pos in range(len(all_token_strings) - 5):
                        match_count = sum(1 for j in range(5) if pos+j < len(all_token_strings) and 
                                        all_token_strings[pos+j] == end_ngram[j])
                        # Allow for partial matches (at least 3 out of 5 tokens match)
                        if match_count >= 3:
                            end_pos = pos + 5
                            break
                    
                    if begin_pos >= 0 and end_pos > begin_pos:
                        # Found match with beginning and ending n-grams
                        sentence_token_positions.append((begin_pos, end_pos, sentence_info))
                        continue
                    elif begin_pos >= 0:
                        # Only found beginning, estimate ending
                        approx_end = min(len(all_tokens), begin_pos + len(sentence_tokens) + 2)
                        sentence_token_positions.append((begin_pos, approx_end, sentence_info))
                        continue
                
                # Method 3: Use sliding window to find best partial match
                best_match_pos = -1
                best_match_score = 0
                
                for pos in range(len(all_token_strings) - min(len(sentence_token_strings), 20) + 1):
                    # Count matching tokens in a window
                    match_count = 0
                    window_size = min(len(sentence_token_strings), 20)
                    
                    for j in range(window_size):
                        if pos+j < len(all_token_strings) and j < len(sentence_token_strings):
                            # Exact token match
                            if all_token_strings[pos+j] == sentence_token_strings[j]:
                                match_count += 1
                            # Partial token match (for subwords)
                            elif (all_token_strings[pos+j] in sentence_token_strings[j] or 
                                sentence_token_strings[j] in all_token_strings[pos+j]):
                                match_count += 0.5
                    
                    match_score = match_count / window_size
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_pos = pos
                
                # If we found a reasonable match
                if best_match_score > 0.3 and best_match_pos >= 0:
                    start_idx = best_match_pos
                    end_idx = min(len(all_tokens), start_idx + len(sentence_tokens))
                    sentence_token_positions.append((start_idx, end_idx, sentence_info))
                    continue
                
                # Method 4: Last resort - use sentence order as approximation
                # Place this sentence after the last one, or at the beginning if it's first
                if i > 0 and len(sentence_token_positions) > 0:
                    prev_end = sentence_token_positions[-1][1]
                    approx_start = prev_end + 1
                    approx_end = min(len(all_tokens), approx_start + len(sentence_tokens))
                    sentence_token_positions.append((approx_start, approx_end, sentence_info))
                else:
                    # First sentence with no match, put at beginning of content area
                    approx_start = 20  # Skip some initial tokens for prompt
                    approx_end = min(len(all_tokens), approx_start + len(sentence_tokens))
                    sentence_token_positions.append((approx_start, approx_end, sentence_info))
                
                print(f"Using approximate position for sentence {i+1}: {sentence[:30]}...")
                
            except Exception as e:
                print(f"Error finding position for sentence {i+1}: {e}")
                # Fallback - use a reasonable guess based on order
                if i > 0 and len(sentence_token_positions) > 0:
                    prev_end = sentence_token_positions[-1][1]
                    approx_start = prev_end + 1
                    approx_end = min(len(all_tokens), approx_start + len(sentence_tokens))
                    sentence_token_positions.append((approx_start, approx_end, sentence_info))
                else:
                    approx_start = 20  # Skip some initial tokens
                    approx_end = min(len(all_tokens), approx_start + len(sentence_tokens))
                    sentence_token_positions.append((approx_start, approx_end, sentence_info))
        
        # Add memory cleanup every few sentences if processing a large batch
        if i > 0 and i % 25 == 0 and torch.cuda.is_available():
            # These objects are large and can be cleaned up periodically:
            del sentence_tokens, sentence_token_strings
            torch.cuda.empty_cache()
            # Only run gc.collect() occasionally as it's expensive
            if i % 100 == 0:
                gc.collect()

        # Sort by start position to ensure consistent ordering
        sentence_token_positions.sort(key=lambda x: x[0])
        
        # Check for overlaps and fix them
        fixed_positions = []
        for i, (start, end, info) in enumerate(sentence_token_positions):
            if i > 0:
                prev_start, prev_end, prev_info = fixed_positions[-1]
                if start < prev_end:
                    # Overlap detected, adjust current start position
                    new_start = prev_end + 1
                    new_end = min(len(all_tokens), new_start + (end - start))
                    fixed_positions.append((new_start, new_end, info))
                    continue
            fixed_positions.append((start, end, info))
        
        return fixed_positions

    def score_sentences_in_chunk(self, question, chunk):
        """
        Score all sentences in a chunk based on their relevance to the question,
        using fuzzy matching to find sentence positions.
        
        Args:
            question: The question text
            chunk: Dict with chunk information
            
        Returns:
            List of sentences with added 'score' field
        """
        try:
            # Build the prompt for this chunk
            chunk_text = chunk['text']

            
            # Make sure we have text to analyze
            if not chunk_text or len(chunk_text) < 10:
                print("Warning: Empty or very short chunk text")
                return [{**s, 'score': 0.01} for s in chunk['sentences']]
            
            print(f"Scoring chunk with {len(chunk['sentences'])} sentences, {len(chunk_text)} chars")
            
            # Format with the right template
            formatted_text = self._format_with_chat_template(question, chunk_text)
            print(formatted_text)
            # Tokenize the formatted text
            inputs = self.retriever_tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            ).to(self.retriever_device)
            
            # For debugging: show the formatted prompt
            print(f"Formatted prompt length: {len(formatted_text)} chars")
            print(f"Tokenized length: {inputs.input_ids.size(1)} tokens")
            
            # Get all tokens
            all_tokens = self.retriever_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            
            # Use fuzzy matching to find sentence positions
            sentence_token_positions = self.find_sentence_positions_fuzzy(all_tokens, chunk['sentences'])
            
            print(f"Found positions for {len(sentence_token_positions)}/{len(chunk['sentences'])} sentences")
            
            # Find question position with fuzzy matching as well
            question_tokens = self.retriever_tokenizer.encode(question, add_special_tokens=False)
            question_token_strings = self.retriever_tokenizer.convert_ids_to_tokens(question_tokens)
            
            # Look for question marker in the text
            full_text = self.retriever_tokenizer.decode(inputs.input_ids[0])
            question_marker = "السؤال:"
            marker_pos = full_text.find(question_marker)
            
            if marker_pos != -1:
                # Found the marker, estimate the token position
                prefix_text = full_text[:marker_pos + len(question_marker)]
                prefix_tokens = self.retriever_tokenizer.encode(prefix_text, add_special_tokens=False)
                question_start_idx = len(prefix_tokens)
                question_end_idx = question_start_idx + len(question_tokens)
                
                # Ensure within bounds
                question_start_idx = min(question_start_idx, len(all_tokens) - 1)
                question_end_idx = min(question_end_idx, len(all_tokens))
                question_indices = list(range(question_start_idx, question_end_idx))
            else:
                # Fallback: look at the end of the text (common for most templates)
                question_start_idx = max(0, len(all_tokens) - len(question_tokens) - 20)
                question_end_idx = min(len(all_tokens), question_start_idx + len(question_tokens) + 10)
                question_indices = list(range(question_start_idx, question_end_idx))
            
            # Make sure we have some question indices
            if not question_indices:
                question_indices = list(range(max(0, len(all_tokens) - 20), min(len(all_tokens), len(all_tokens) - 10)))
            
            # Run the model and get attention scores
            with torch.no_grad():
                outputs = self.retriever_model(**inputs, output_attentions=True)
                
                if outputs.attentions and len(outputs.attentions) > 0:
                    # Use all layers instead of just the last one
                    all_layers_attn = outputs.attentions
                    
                    # Initialize tensor to accumulate attention
                    accumulated_attention = None
                    
                    # Process each layer
                    for layer_idx, layer_attn in enumerate(all_layers_attn[:-2]):
                        # Check dimensions and process appropriately
                        if layer_attn.dim() == 4:  # [batch, heads, seq_len, seq_len]
                            # Average across heads to get [batch, seq_len, seq_len]
                            layer_avg_attention = torch.sum(layer_attn, dim=1)[0]
                        elif layer_attn.dim() == 3:  # [batch, seq_len, seq_len]
                            # Already the right shape, just get first batch
                            layer_avg_attention = layer_attn[0]
                        else:
                            print(f"Skipping layer {layer_idx} due to unexpected shape: {layer_attn.shape}")
                            continue
                        
                        # Convert to float32 to avoid numerical issues
                        layer_avg_attention = layer_avg_attention.to(torch.float32)
                        
                        # Initialize or accumulate
                        if accumulated_attention is None:
                            accumulated_attention = layer_avg_attention
                        else:
                            # Ensure dimensions match before adding
                            if accumulated_attention.shape == layer_avg_attention.shape:
                                accumulated_attention += layer_avg_attention
                            else:
                                print(f"Skipping layer {layer_idx} due to shape mismatch: {accumulated_attention.shape} vs {layer_avg_attention.shape}")
                    
                    # Average across all layers
                    if accumulated_attention is not None:
                        avg_attention = accumulated_attention / len(all_layers_attn)
                    else:
                        # Fallback to using just the last layer if accumulation failed
                        last_layer_attn = outputs.attentions[-1]
                        if last_layer_attn.dim() == 4:
                            avg_attention = torch.mean(last_layer_attn, dim=1)[0]
                        else:
                            avg_attention = last_layer_attn[0]
                    
                    # Convert to float32 in case it's not already
                    avg_attention = avg_attention.to(torch.float32)
                                    
                    # Score each sentence by its attention from question tokens
                    scored_sentences = []
                    
                    for start_idx, end_idx, sentence_info in sentence_token_positions:
                        # Ensure indices are within bounds
                        valid_start = max(0, min(start_idx, avg_attention.shape[1] - 1))
                        valid_end = max(valid_start + 1, min(end_idx, avg_attention.shape[1]))
                        
                        sentence_indices = list(range(valid_start, valid_end))
                        
                        if not sentence_indices:
                            print(f"Warning: Empty sentence indices for {sentence_info['sentence'][:30]}...")
                            scored_sentences.append({**sentence_info, 'score': 0.01})
                            continue
                        
                        try:
                            # Calculate attention from question to this sentence
                            q2s_attention = avg_attention[question_indices, :][:, sentence_indices]
                            
                            # Different scoring methods
                            total_attention = torch.sum(q2s_attention).item()
                            max_attention = torch.max(q2s_attention).item()
                            mean_attention = torch.mean(q2s_attention).item()
                            
                            # For longer sentences, adjust the score to avoid bias
                            length_factor = 1.0 / max(1.0, math.log(1 + len(sentence_indices) / 5))
                            
                            # Combine different metrics
                            score = ( 0.4 * total_attention + 
                                     0.4 * max_attention + 
                                     0.2 * mean_attention) * length_factor
                            #score = total_attention * length_factor

                            # Also compute a length-normalized score
                            norm_score = score / max(len(sentence_indices), 1)
                            
                            
                            # Store scores with the sentence
                            sentence_info_copy = sentence_info.copy()
                            sentence_info_copy.update({
                                'score': score,
                                'normalized_score': norm_score,
                                'token_positions': (valid_start, valid_end),
                                'num_tokens': len(sentence_indices),
                                'total_attention': total_attention,
                                'max_attention': max_attention,
                                'mean_attention': mean_attention
                            })
                            scored_sentences.append(sentence_info_copy)
                            
                        except Exception as e:
                            print(f"Error scoring sentence: {e}")
                            # Add with minimal score
                            scored_sentences.append({**sentence_info, 'score': 0.01})
                    
                    # Return sentences sorted by score
                    return sorted(scored_sentences, key=lambda x: x.get('score', 0), reverse=True)
                else:
                    print("Warning: No attention tensors from model")
                    # Fallback - assign scores based on TF-IDF similarity to question
                    return self.score_sentences_tfidf(question, chunk['sentences'])
                
        except Exception as e:
            print(f"Error scoring sentences: {e}")
            import traceback
            traceback.print_exc()
            
            # Return sentences with minimal scores as fallback
            return [
                {**s, 'score': 0.01, 'normalized_score': 0.01}
                for s in chunk['sentences']
            ]

    def score_sentences_tfidf(self, question, sentences):
        """
        Score sentences based on TF-IDF similarity to the question.
        This is a fallback method when attention-based scoring fails.
        
        Args:
            question: Question text
            sentences: List of sentence dictionaries
            
        Returns:
            List of sentences with scores added
        """
        import math
        from collections import Counter
        
        # Extract sentence texts
        sentence_texts = [s['detailed_sentence'] for s in sentences]
        
        # Simple tokenization - split by whitespace and punctuation
        def simple_tokenize(text):
            # Remove punctuation and split
            for char in ".,!?;:()[]{}\"'":
                text = text.replace(char, " ")
            return [token for token in text.split() if token]
        
        # Tokenize all texts
        question_tokens = simple_tokenize(question)
        sentence_tokens_list = [simple_tokenize(s) for s in sentence_texts]
        
        # Calculate document frequency
        term_df = Counter()
        for tokens in sentence_tokens_list:
            term_df.update(set(tokens))
        
        # Calculate IDF for each term
        num_docs = len(sentence_tokens_list)
        term_idf = {term: math.log(num_docs / (df + 1)) for term, df in term_df.items()}
        
        # Calculate TF-IDF scores
        scored_sentences = []
        for i, sentence_info in enumerate(sentences):
            sentence_tokens = sentence_tokens_list[i]
            
            # Skip empty sentences
            if not sentence_tokens:
                scored_sentences.append({**sentence_info, 'score': 0.01})
                continue
            
            # Calculate term frequency in this sentence
            term_tf = Counter(sentence_tokens)
            
            # Calculate TF-IDF vector for sentence
            tfidf_vector = {term: tf * term_idf.get(term, 0) for term, tf in term_tf.items()}
            
            # Calculate dot product with question
            score = sum(tfidf_vector.get(term, 0) for term in question_tokens)
            
            # Normalize by sentence length
            norm_score = score / len(sentence_tokens)
            
            # Add a small bonus for sentences from the correct root
            if sentence_info.get('is_correct', False):
                score *= 1.1
                norm_score *= 1.1
            
            # Store score
            scored_sentences.append({
                **sentence_info,
                'score': score,
                'normalized_score': norm_score
            })
        
        # Sort by score
        return sorted(scored_sentences, key=lambda x: x.get('score', 0), reverse=True)
    
    def _format_with_chat_template(self, question, content):
        """
        Format question and content using the appropriate chat template.
        
        Args:
            question: The question to ask
            content: The context content
            
        Returns:
            Formatted text for the model
        """
        # System prompt for retriever
        system_content = "أنت مساعد جيد"
        
        # User content with clear markers
        user_content = "--بداية السياق--\n" + content + "\n--نهاية السياق--\n"
        user_content += "السؤال: " + question
        
        # Create messages for the chat template
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Try to detect model type
        if hasattr(self.retriever_model, "config") and hasattr(self.retriever_model.config, "name_or_path"):
            model_name = self.retriever_model.config.name_or_path.lower()
            if "llama-2" in model_name or "llama2" in model_name:
                model_family = "llama2"
            elif "llama-3" in model_name or "llama3" in model_name:
                model_family = "llama3"
            elif hasattr(self.retriever_tokenizer, "chat_template") and self.retriever_tokenizer.chat_template is not None:
                model_family = "custom_template"
            else:
                model_family = "other"
        else:
            model_family = "other"
        
        # Apply the appropriate template
        if model_family == "llama2":
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            formatted = f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]"
            
        elif model_family == "llama3":
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            formatted = f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            
        elif model_family == "custom_template" and hasattr(self.retriever_tokenizer, "apply_chat_template"):
            formatted = self.retriever_tokenizer.apply_chat_template(messages, tokenize=False)
            
        else:
            # Fallback simple template
            formatted = f"System: {system_content}\nUser: {user_content}"
            
        return formatted
    
    def process_all_contexts(self, question, contexts_data, top_k=10, progress_callback=None):
        """
        Process all contexts, score sentences, and return the top-k most relevant sentences.
        
        Args:
            question: The question to answer
            contexts_data: List of context dictionaries
            top_k: Number of top sentences to return
            progress_callback: Function to call with progress updates
            
        Returns:
            Dict with retrieval results
        """
        # Extract sentences from all contexts
        if progress_callback:
            progress_callback(10, "Extracting sentences from contexts...")
        all_sentences = self.preprocess_contexts(contexts_data)
        
        # Create chunks of sentences
        if progress_callback:
            progress_callback(20, "Creating sentence chunks...")
        chunks = self.create_sentence_chunks(all_sentences, self.effective_seq_len)
        
        # Score sentences in each chunk
        all_scored_sentences = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_percent = 20 + int(60 * ((i + 1) / len(chunks)))
                progress_callback(progress_percent, f"Scoring sentences in chunk {i+1}/{len(chunks)}...")
            
            scored_sentences = self.score_sentences_in_chunk(question, chunk)
            all_scored_sentences.extend(scored_sentences)
            
            # Clean up memory after each chunk - already in your code but ensure it works
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Sort by score
        sorted_sentences = sorted(all_scored_sentences, key=lambda x: x.get('score', 0), reverse=True)
        
    
        # Get top-k sentences
        top_sentences = sorted_sentences[:top_k]
      
        
        # Count sentences by root
        root_counts = {}
        for sentence in top_sentences:
            root = sentence['original_root']
            if root not in root_counts:
                root_counts[root] = 0
            root_counts[root] += 1
        
        # Check if correct root is represented
        correct_roots = [sentence['original_root'] for sentence in top_sentences if sentence.get('is_correct', False)]
        correct_root_present = len(correct_roots) > 0
        
        # Sort roots by sentence count
        sorted_roots = sorted(root_counts.items(), key=lambda x: x[1], reverse=True)
        
        if progress_callback:
            progress_callback(90, "Generating answer with top sentences...")
            
        # Combine top sentences into a single context for answer generation
        #retrieved_text = ".\n".join([s['sentence'] for s in top_sentences])
        retrieved_text = "فيما يلي جمل من المعجم قد يكون لها علاقة بإجابة السؤال:\n\n"
        for i, sentence in enumerate(top_sentences):
            retrieved_text += f"{i+1}. الجذر: {sentence['root']}\n{sentence['sentence']}\n\n"

        return {
            'all_sentences': all_scored_sentences,
            'top_sentences': top_sentences,
            'root_counts': root_counts,
            'sorted_roots': sorted_roots,
            'correct_root_present': correct_root_present,
            'retrieved_text': retrieved_text
        }
    
    def generate_answer(self, question, retrieved_text):
        """
        Generate an answer using the retriever-selected text.
        
        Args:
            question: The question to answer
            retrieved_text: The retrieved context text
            
        Returns:
            Generated answer text
        """
        try:
            # Format for the answerer model
            system_content = "أنت مساعد جيد يجيب على الأسئلة من السياق المقدم"
            user_content = f"بناء على السياق التالي أجب على السؤال: \n\nالسياق:\n{retrieved_text}\nالسؤال:{question}"
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            # Use the model's chat template if available
            if hasattr(self.answerer_tokenizer, "apply_chat_template"):
                prompted_text = self.answerer_tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                # Fallback simple template
                prompted_text = f"System: {system_content}\nUser: {user_content}\nAssistant:"
            print(prompted_text)
            # Tokenize the prompt
            inputs = self.answerer_tokenizer(
                prompted_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            ).to(self.answerer_device)
            
            # Generate the answer
            with torch.no_grad():
                outputs = self.answerer_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    return_dict_in_generate=False,
                    output_attentions=False,
                    output_hidden_states=False
                )
            
            # Get just the new tokens (the answer)
            input_length = len(inputs.input_ids[0])
            new_tokens = outputs[0][input_length:]
            answer_text = self.answerer_tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return answer_text
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return "Error generating answer from retrieved sentences."

# To integrate this with your MultiRootAnalysisThread
class MultiRootAnalysisThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, dict, dict, dict, str, str)
    
    def __init__(self, retriever_model, retriever_tokenizer, answerer_model, answerer_tokenizer, 
                 question, contexts_data, retriever_device, answerer_device, max_seq_len, actual_answer):
        super().__init__()
        self.retriever_model = retriever_model
        self.retriever_tokenizer = retriever_tokenizer
        self.answerer_model = answerer_model
        self.answerer_tokenizer = answerer_tokenizer
        self.question = question
        self.contexts_data = contexts_data
        self.retriever_device = retriever_device
        self.answerer_device = answerer_device
        self.max_seq_len = max_seq_len
        self.actual_answer = actual_answer
        
    def run(self):
        try:
            # Clean memory at the start
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            total_contexts = len(self.contexts_data)
            if total_contexts == 0:
                self.progress_signal.emit(100, "No contexts to analyze!")
                self.finished_signal.emit(True, {}, {}, {}, "No contexts to analyze", "")
                return
            
            # Create a sentence retriever
            try:
                sentence_retriever = SentenceRetriever(
                    self.retriever_model,
                    self.retriever_tokenizer,
                    self.answerer_model,
                    self.answerer_tokenizer,
                    self.retriever_device,
                    self.answerer_device,
                    self.max_seq_len
                )
                
                # Process all contexts to get top sentences
                self.progress_signal.emit(5, "Processing contexts and extracting sentences...")
                
                retrieval_results = sentence_retriever.process_all_contexts(
                    self.question,
                    self.contexts_data,
                    top_k=20,  # Get top 20 sentences
                    progress_callback=self.progress_signal.emit
                )
            except Exception as e:
                print(f"Error in sentence retriever: {e}")
                import traceback
                traceback.print_exc()
                self.finished_signal.emit(False, {}, {}, {}, "Error in sentence analysis.", "Error in sentence analysis.")
                return
            
            # Generate an answer with the retrieved sentences
            self.progress_signal.emit(95, "Generating answer with retrieved sentences...")
            
            # Get the retrieved text
            retrieved_text = retrieval_results['retrieved_text']
            
            # Generate the answer
            try:
                grounded_answer = sentence_retriever.generate_answer(self.question, retrieved_text)
            except Exception as e:
                print(f"Error generating grounded answer: {e}")
                grounded_answer = "Error generating answer with context."
            
            # Also generate an ungrounded answer without context for comparison
            self.progress_signal.emit(98, "Generating ungrounded answer for comparison...")
            
            # Simple ungrounded answer generation
            try:
                ungrounded_answer = self.generate_ungrounded_answer()
            except Exception as e:
                print(f"Error generating ungrounded answer: {e}")
                ungrounded_answer = "Error generating answer without context."
            
            # Prepare root scores dict (for API compatibility)
            # Use sentence counts as scores
            root_scores = {root: count for root, count in retrieval_results['root_counts'].items()}
            
            # Prepare tokens dict (for API compatibility)
            # Use top sentences for each root
            tokens_dict = {}
            scores_dict = {}
            
            # Group sentences by root
            sentences_by_root = {}
            for sentence in retrieval_results['all_sentences']:
                root = sentence['root']
                if root not in sentences_by_root:
                    sentences_by_root[root] = []
                sentences_by_root[root].append(sentence)
            
            # For each root, collect tokens and scores from its top sentences
            for root, sentences in sentences_by_root.items():
                # Sort by score
                sorted_sentences = sorted(sentences, key=lambda x: x.get('original_index', 0), reverse=False)
                
                # Get tokens (we'll just use sentence text as "tokens" for visualization)
                tokens = [s['sentence'] for s in sorted_sentences]  
                tokens_dict[root] = tokens
                
                # Get scores
                scores = [s.get('score', 0.01) for s in sorted_sentences]
                scores_dict[root] = scores
            
            # Finish
            self.progress_signal.emit(100, "Analysis complete!")
            self.finished_signal.emit(
                True, 
                root_scores, 
                tokens_dict,
                scores_dict,
                grounded_answer,
                ungrounded_answer
            )
            
        except Exception as e:
            print(f"Error in multi-root analysis thread: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean memory even when there's an error
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                
            self.finished_signal.emit(False, {}, {}, {}, "Error during analysis.", "Error during analysis.")
            
    def generate_ungrounded_answer(self):
        """Generate an answer without using any context."""
        try:
            # Create direct messages without using the standard template
            direct_messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions accurately and concisely."},
                {"role": "user", "content": self.question}
            ]
            
            # Use the model's chat template if available
            if hasattr(self.answerer_tokenizer, "apply_chat_template"):
                ungrounded_prompt = self.answerer_tokenizer.apply_chat_template(
                    direct_messages, tokenize=False
                )
            else:
                # Fallback to simple format
                ungrounded_prompt = f"Assistant: I'm a helpful assistant.\n\nUser: {self.question}\n\nAssistant:"
            
            # Tokenize
            ungrounded_inputs = self.answerer_tokenizer(
                ungrounded_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            ).to(self.answerer_device)
            
            # Generate ungrounded answer
            with torch.no_grad():
                ungrounded_outputs = self.answerer_model.generate(
                    **ungrounded_inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    return_dict_in_generate=False,
                    output_attentions=False,
                    output_hidden_states=False
                )
            
            # Get just the newly generated tokens
            ungrounded_input_length = len(ungrounded_inputs.input_ids[0])
            ungrounded_new_tokens = ungrounded_outputs[0][ungrounded_input_length:]
            ungrounded_answer = self.answerer_tokenizer.decode(
                ungrounded_new_tokens, skip_special_tokens=True
            )
            
            return ungrounded_answer
            
        except Exception as e:
            print(f"Error generating ungrounded answer: {e}")
            return "Error generating ungrounded answer."

class AttentionVisualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Model Attention Visualizer for Root Comparison")
        self.setGeometry(100, 100, 1500, 1000)
        
        # Model and data attributes
        self.retriever_model = None
        self.retriever_tokenizer = None
        self.answerer_model = None
        self.answerer_tokenizer = None
        self.data_df = None  # Main dataframe with all data (questions, answers)
        self.roots_df = None  # Dictionary of roots and their content
        self.all_roots = []  # All available roots
        self.current_question = None
        self.current_root = None  # The correct root for current question
        self.sampled_roots = []  # Sampled roots for comparison (includes the correct one)
        
        # Use GPU 0 for retriever and GPU 1 for answerer if available
        if torch.cuda.device_count() >= 2:
            self.retriever_device = torch.device("cuda:0")
            self.answerer_device = torch.device("cuda:1")
        elif torch.cuda.is_available():
            # Fall back to using the same GPU for both
            self.retriever_device = torch.device("cuda:0")
            self.answerer_device = torch.device("cuda:0")
        else:
            # Fall back to CPU
            self.retriever_device = torch.device("cpu")
            self.answerer_device = torch.device("cpu")
            
        self.retriever_model_type = "other"  # Default model type
        self.answerer_model_type = "other"  # Default model type
        self.generated_answer = ""  # Store the generated answer
        
        # Initialize the UI
        self.init_ui()
        
    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create top panel for model and data loading
        top_panel = QGroupBox("Load Models and Data")
        top_layout = QGridLayout(top_panel)
        
        # Retriever model selection
        retriever_label = QLabel("Retriever Model Path:")
        self.retriever_path_combo = QComboBox()
        self.retriever_path_combo.setEditable(True)
        self.retriever_path_combo.setMinimumWidth(300)
        # Add some commonly used models
        self.retriever_path_combo.addItem("/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/mysam/oryx-2.0-1B-Instruct-Llama")
        retriever_browse_btn = QPushButton("Browse...")
        retriever_browse_btn.clicked.connect(lambda: self.browse_model("retriever"))
        
        # Answerer model selection
        answerer_label = QLabel("Answerer Model Path:")
        self.answerer_path_combo = QComboBox()
        self.answerer_path_combo.setEditable(True)
        self.answerer_path_combo.setMinimumWidth(300)
        # You can set a default path or leave empty
        self.answerer_path_combo.addItem("/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/ALLaM-7B-Instruct-preview")
        # Add the same model as a second option for testing with smaller models
        self.answerer_path_combo.addItem("/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/mysam/oryx-2.0-1B-Instruct-Llama")
        answerer_browse_btn = QPushButton("Browse...")
        answerer_browse_btn.clicked.connect(lambda: self.browse_model("answerer"))
        
        # Questions dataset selection
        data_label = QLabel("Questions Dataset:")
        self.data_path_combo = QComboBox()
        self.data_path_combo.setEditable(True)
        self.data_path_combo.setMinimumWidth(300)
        self.data_path_combo.addItem("mysam/kalima_w_maana")
        data_browse_btn = QPushButton("Browse...")
        data_browse_btn.clicked.connect(self.browse_data)
        
        # Roots dataset selection
        roots_label = QLabel("Roots Dataset:")
        self.roots_path_combo = QComboBox()
        self.roots_path_combo.setEditable(True)
        self.roots_path_combo.setMinimumWidth(300)
        self.roots_path_combo.addItem("mysam/lisan_alarab")  # Default roots dataset
        roots_browse_btn = QPushButton("Browse...")
        roots_browse_btn.clicked.connect(self.browse_roots)
        
        # Quantization options
        retriever_quant_label = QLabel("Retriever Quantization:")
        self.retriever_quant_combo = QComboBox()
        self.retriever_quant_combo.addItems(["fp16", "4bit", "8bit", "none"])
        self.retriever_quant_combo.setCurrentIndex(0)  # Set fp16 as default
        
        answerer_quant_label = QLabel("Answerer Quantization:")
        self.answerer_quant_combo = QComboBox()
        self.answerer_quant_combo.addItems(["fp16", "4bit", "8bit", "none"])
        self.answerer_quant_combo.setCurrentIndex(0)  # Set fp16 as default
        
        # Max sequence length
        seq_len_label = QLabel("Max Sequence Length:")
        self.seq_len_spin = QSpinBox()
        self.seq_len_spin.setRange(512, 8192)
        self.seq_len_spin.setValue(4096)
        self.seq_len_spin.setSingleStep(512)
        
        # Device selectors
        device_layout = QHBoxLayout()
        device_label = QLabel("GPU Devices:")
        self.device_info_label = QLabel("")
        
        if torch.cuda.device_count() >= 2:
            self.device_info_label.setText(f"Using GPU 0 for retriever and GPU 1 for answerer ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_name(1)})")
        elif torch.cuda.is_available():
            self.device_info_label.setText(f"Using GPU 0 for both models ({torch.cuda.get_device_name(0)})")
        else:
            self.device_info_label.setText("Using CPU for both models (no GPU available)")
            
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_info_label, 1)  # Give it stretch
        
        # Memory cleanup option
        self.memory_cleanup_checkbox = QCheckBox("Enable GPU memory cleanup between contexts")
        self.memory_cleanup_checkbox.setChecked(True)
        self.memory_cleanup_checkbox.setToolTip("Free GPU memory after each context analysis to prevent out-of-memory errors")
        
        # Load button
        self.load_btn = QPushButton("Load Models & Data")
        self.load_btn.clicked.connect(self.load_model_and_data)
        
        # Add widgets to top layout
        # First row
        top_layout.addWidget(retriever_label, 0, 0)
        top_layout.addWidget(self.retriever_path_combo, 0, 1)
        top_layout.addWidget(retriever_browse_btn, 0, 2)
        top_layout.addWidget(retriever_quant_label, 0, 3)
        top_layout.addWidget(self.retriever_quant_combo, 0, 4)
        
        # Second row
        top_layout.addWidget(answerer_label, 1, 0)
        top_layout.addWidget(self.answerer_path_combo, 1, 1)
        top_layout.addWidget(answerer_browse_btn, 1, 2)
        top_layout.addWidget(answerer_quant_label, 1, 3)
        top_layout.addWidget(self.answerer_quant_combo, 1, 4)
        
        # Third row
        top_layout.addWidget(data_label, 2, 0)
        top_layout.addWidget(self.data_path_combo, 2, 1)
        top_layout.addWidget(data_browse_btn, 2, 2)
        top_layout.addWidget(seq_len_label, 2, 3)
        top_layout.addWidget(self.seq_len_spin, 2, 4)
        
        # Fourth row
        top_layout.addWidget(roots_label, 3, 0)
        top_layout.addWidget(self.roots_path_combo, 3, 1)
        top_layout.addWidget(roots_browse_btn, 3, 2)
        
        # Fifth row
        top_layout.addLayout(device_layout, 4, 0, 1, 5)
        top_layout.addWidget(self.memory_cleanup_checkbox, 5, 0, 1, 3)
        
        # Load button spans all rows
        top_layout.addWidget(self.load_btn, 0, 5, 6, 1)
        
        # Add top panel to main layout
        main_layout.addWidget(top_panel)
        
        # Create splitter for main content area
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)  # Give it stretch factor
        
        # Left panel: Questions list and analysis controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Questions list
        questions_label = QLabel("Questions:")
        self.questions_list = QListWidget()
        self.questions_list.currentRowChanged.connect(self.question_selected)
        
        # Analysis controls
        analysis_group = QGroupBox("Analysis Controls")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Number of contexts to include
        contexts_layout = QHBoxLayout()
        contexts_label = QLabel("Number of contexts to sample:")
        self.contexts_spin = QSpinBox()
        self.contexts_spin.setRange(3, 9120)
        self.contexts_spin.setValue(5)
        contexts_layout.addWidget(contexts_label)
        contexts_layout.addWidget(self.contexts_spin)
        
        # Analysis button
        self.analyze_btn = QPushButton("Analyze with Random Contexts")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)  # Will be enabled when model and data are loaded
        
        # Add to analysis layout
        analysis_layout.addLayout(contexts_layout)
        analysis_layout.addWidget(self.analyze_btn)
        
        # Add to left layout
        left_layout.addWidget(questions_label)
        left_layout.addWidget(self.questions_list, 3)  # Give it more stretch factor
        left_layout.addWidget(analysis_group, 1)
        
        # Right panel: Results visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Root selection (now for visualization only)
        root_layout = QHBoxLayout()
        root_label = QLabel("Visualize Context:")
        self.root_combo = QComboBox()
        self.root_combo.currentIndexChanged.connect(self.root_changed_for_visualization)
        root_layout.addWidget(root_label)
        root_layout.addWidget(self.root_combo, 1)  # Give it stretch factor
        
        # Add indicator if it's the correct root
        self.root_indicator = QLabel("")
        root_layout.addWidget(self.root_indicator)
        
        # Content visualization
        content_label = QLabel("Content with Token Attention Highlighting:")
        self.content_text = QTextEdit()
        self.content_text.setReadOnly(True)
        
        # Add color legend
        legend_layout = QHBoxLayout()
        legend_label = QLabel("Attention Scale:")
        self.color_legend = QWidget()
        self.color_legend.setFixedHeight(20)
        self.color_legend.setMinimumWidth(200)
        self.color_legend.paintEvent = self.paint_legend
        legend_layout.addWidget(legend_label)
        legend_layout.addWidget(self.color_legend, 1)  # Give it stretch factor
        
        # Question and answer section in a 2x2 grid
        qa_group = QGroupBox("Question and Answers")
        qa_grid_layout = QGridLayout(qa_group)
        
        # Row 1, Col 1: Question
        question_label = QLabel("Question:")
        self.question_text = QTextEdit()
        self.question_text.setReadOnly(True)
        self.question_text.setMaximumHeight(80)  # Reduce height to save space
        # Style for Arabic text display
        self.question_text.setStyleSheet("""
            padding: 8px;
            margin: 0px;
            direction: rtl;
            text-align: right;
            font-size: 12pt;
            border: 1px solid #d9d9d9;
            border-radius: 4px;
        """)
        
        # Row 1, Col 2: Correct Answer
        answer_label = QLabel("Correct Answer:")
        self.answer_text = QTextEdit()
        self.answer_text.setReadOnly(True)
        self.answer_text.setMaximumHeight(80)
        # Style for Arabic text display
        self.answer_text.setStyleSheet("""
            padding: 8px;
            margin: 0px;
            direction: rtl;
            text-align: right;
            font-size: 12pt;
            border: 1px solid #d9d9d9;
            border-radius: 4px;
        """)
        
        # Row 2, Col 1: Grounded answer (with context)
        grounded_answer_label = QLabel("Grounded Answer (with context):")
        self.generated_answer_text = QTextEdit()
        self.generated_answer_text.setReadOnly(True)
        self.generated_answer_text.setMaximumHeight(80)
        # Set text direction to RTL for Arabic and add proper padding and margins
        self.generated_answer_text.setStyleSheet("""
            background-color: #e6f7ff; /* Light blue background */
            padding: 8px;
            margin: 0px;
            direction: rtl;
            text-align: right;
            font-size: 12pt;
            border: 1px solid #1890ff;
            border-radius: 4px;
        """)
        
        # Row 2, Col 2: Ungrounded answer (without context)
        ungrounded_answer_label = QLabel("Ungrounded Answer (no context):")
        self.ungrounded_answer_text = QTextEdit()
        self.ungrounded_answer_text.setReadOnly(True)
        self.ungrounded_answer_text.setMaximumHeight(80)
        # Set text direction to RTL for Arabic and add proper padding and margins
        self.ungrounded_answer_text.setStyleSheet("""
            background-color: #fff1f0; /* Light red background */
            padding: 8px;
            margin: 0px;
            direction: rtl;
            text-align: right;
            font-size: 12pt;
            border: 1px solid #ff4d4f;
            border-radius: 4px;
        """)
        
        # Add widgets to the grid layout
        # Row 1
        qa_grid_layout.addWidget(question_label, 0, 0)
        qa_grid_layout.addWidget(answer_label, 0, 1)
        qa_grid_layout.addWidget(self.question_text, 1, 0)
        qa_grid_layout.addWidget(self.answer_text, 1, 1)
        
        # Row 2
        qa_grid_layout.addWidget(grounded_answer_label, 2, 0)
        qa_grid_layout.addWidget(ungrounded_answer_label, 2, 1)
        qa_grid_layout.addWidget(self.generated_answer_text, 3, 0)
        qa_grid_layout.addWidget(self.ungrounded_answer_text, 3, 1)
        
        # Set equal column widths
        qa_grid_layout.setColumnStretch(0, 1)
        qa_grid_layout.setColumnStretch(1, 1)
        
        # Add some spacing
        qa_grid_layout.setSpacing(10)
        
        # Attention distribution visualization
        attn_label = QLabel("Comparative Attention Distribution:")
        self.attn_plot = FigureCanvas(Figure(figsize=(5, 5)))  # Increase figure height
        self.attn_plot.setMinimumHeight(300)  # Set a minimum height for the chart
        self.attn_plot.figure.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.15)
        self.attn_ax = self.attn_plot.figure.add_subplot(111)
        
        # Add everything to right layout
        right_layout.addLayout(root_layout)
        right_layout.addWidget(content_label)
        right_layout.addWidget(self.content_text, 3)  # Give content stretch factor
        right_layout.addLayout(legend_layout)
        right_layout.addWidget(qa_group, 1)  # Add the question and answers group with minimal stretch
        right_layout.addWidget(attn_label)
        right_layout.addWidget(self.attn_plot, 4)  # Increase stretch factor for the chart
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])  # Set initial sizes
        
        # Status bar for showing progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.statusBar().addPermanentWidget(self.progress_bar, 1)
        self.statusBar().showMessage("Ready")
        
        # Initialize color gradient for attention visualization
        self.setup_color_gradient()
        
        # Data structures for analysis results
        self.analysis_results = {}  # Will store results of the latest analysis
        
    def setup_color_gradient(self):
        """
        Create a single-color gradient from white to red.
        """
        # Create a custom white-to-red colormap
        from matplotlib.colors import LinearSegmentedColormap
        
        # Define the colors for the gradient (white to red)
        colors = [(1, 1, 1), (0.8, 0, 0)]  # From white to red
        
        # Create the colormap
        self.cmap = LinearSegmentedColormap.from_list('WhiteToRed', colors, N=256)
    
    def paint_legend(self, event):
        from PyQt5.QtGui import QPainter, QLinearGradient, QBrush, QColor
        painter = QPainter(self.color_legend)
        width = self.color_legend.width()
        height = self.color_legend.height()
        
        # Create gradient
        gradient = QLinearGradient(0, 0, width, 0)
        
        # Add color stops for seismic gradient (blue-white-red)
        for i in range(11):
            pos = i / 10.0
            color_rgba = self.cmap(pos)
            color = QColor(
                int(color_rgba[0] * 255), 
                int(color_rgba[1] * 255), 
                int(color_rgba[2] * 255)
            )
            gradient.setColorAt(pos, color)
        
        # Fill rectangle with gradient
        painter.fillRect(0, 0, width, height, QBrush(gradient))
        
        # Add text markers
        painter.setPen(Qt.black)
        # Convert float values to integers for drawText
        painter.drawText(0, int(height-2), "Low")
        # Convert end position to int
        end_x = int(width-30)
        painter.drawText(end_x, int(height-2), "High")

    def browse_model(self, model_type="retriever"):
        """Browse for model directory for either retriever or answerer model."""
        folder = QFileDialog.getExistingDirectory(self, f"Select {model_type.capitalize()} Model Directory")
        if folder:
            if model_type == "retriever":
                self.retriever_path_combo.setCurrentText(folder)
            else:
                self.answerer_path_combo.setCurrentText(folder)
            
    def browse_data(self):
        """Browse for questions dataset file or HF dataset name."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Questions Dataset File", "", "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)")
        if file_path:
            self.data_path_combo.setCurrentText(file_path)
            
    def browse_roots(self):
        """Browse for roots dataset file or HF dataset name."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Roots Dataset File", "", "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)")
        if file_path:
            self.roots_path_combo.setCurrentText(file_path)
            
    def load_model_and_data(self):
        """Load both retriever and answerer models and dataset."""
        retriever_path = self.retriever_path_combo.currentText()
        answerer_path = self.answerer_path_combo.currentText()
        data_path = self.data_path_combo.currentText()
        roots_path = self.roots_path_combo.currentText()
        
        # Validate paths
        if not os.path.exists(retriever_path):
            QMessageBox.warning(self, "Error", f"Retriever model path does not exist: {retriever_path}")
            return
            
        if not os.path.exists(answerer_path):
            QMessageBox.warning(self, "Error", f"Answerer model path does not exist: {answerer_path}")
            return
        
        # Start loading in a separate thread to keep UI responsive
        self.load_thread = DualModelLoaderThread(
            retriever_path,
            answerer_path,
            data_path,
            roots_path,
            self.retriever_quant_combo.currentText(),
            self.answerer_quant_combo.currentText(),
            self.seq_len_spin.value(),
            self.retriever_device,
            self.answerer_device
        )
        self.load_thread.progress_signal.connect(self.update_progress)
        self.load_thread.finished_signal.connect(self.loading_finished)
        
        # Disable UI during loading
        self.load_btn.setEnabled(False)
        self.statusBar().showMessage("Loading models and data...")
        self.progress_bar.setValue(0)
        
        # Start the thread
        self.load_thread.start()
        
    def update_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(message)
        
    def loading_finished(self, success, retriever_model, retriever_tokenizer, 
                         answerer_model, answerer_tokenizer, data_df, roots_df):
        if success:
            self.retriever_model = retriever_model
            self.retriever_tokenizer = retriever_tokenizer
            self.answerer_model = answerer_model
            self.answerer_tokenizer = answerer_tokenizer
            self.data_df = data_df
            self.roots_df = roots_df
            
            # Extract all unique roots - if we have a roots dataset, use that instead
            if self.roots_df is not None and 'root' in self.roots_df.columns:
                self.all_roots = self.roots_df['root'].unique().tolist()
                print(f"Loaded {len(self.all_roots)} unique roots from roots dataset")
            else:
                # Fallback to using roots from the questions dataset
                self.all_roots = data_df['root'].unique().tolist()
                print(f"Loaded {len(self.all_roots)} unique roots from questions dataset")
            
            # Detect retriever model type
            if hasattr(retriever_model, "config") and hasattr(retriever_model.config, "name_or_path"):
                model_name = retriever_model.config.name_or_path.lower()
                if "llama-2" in model_name or "llama2" in model_name:
                    self.retriever_model_type = "llama2"
                elif "llama-3" in model_name or "llama3" in model_name:
                    self.retriever_model_type = "llama3"
                elif hasattr(retriever_tokenizer, "chat_template") and retriever_tokenizer.chat_template is not None:
                    self.retriever_model_type = "custom_template"
                else:
                    self.retriever_model_type = "other"
                    
                print(f"Detected retriever model type: {self.retriever_model_type}")
                
            # Detect answerer model type
            if hasattr(answerer_model, "config") and hasattr(answerer_model.config, "name_or_path"):
                model_name = answerer_model.config.name_or_path.lower()
                if "llama-2" in model_name or "llama2" in model_name:
                    self.answerer_model_type = "llama2"
                elif "llama-3" in model_name or "llama3" in model_name:
                    self.answerer_model_type = "llama3"
                elif hasattr(answerer_tokenizer, "chat_template") and answerer_tokenizer.chat_template is not None:
                    self.answerer_model_type = "custom_template"
                else:
                    self.answerer_model_type = "other"
                    
                print(f"Detected answerer model type: {self.answerer_model_type}")
            
            # Process the dataframe
            # Check if dataset has the required columns
            if {'root', 'context', 'question', 'answer'}.issubset(data_df.columns):
                # Extract unique questions
                unique_questions = data_df[['question']].drop_duplicates().reset_index(drop=True)
                
                # Populate the questions list
                self.questions_list.clear()
                self.questions_list.addItems(unique_questions['question'].tolist())
                
                # Enable the analyze button now that data is loaded
                self.analyze_btn.setEnabled(True)
                
                self.statusBar().showMessage(
                    f"Retriever model ({self.retriever_model_type}) and Answerer model ({self.answerer_model_type}) " +
                    f"loaded successfully with {len(unique_questions)} questions and {len(self.all_roots)} roots."
                )
            else:
                missing = []
                for col in ['root', 'context', 'question', 'answer']:
                    if col not in data_df.columns:
                        missing.append(col)
                QMessageBox.warning(self, "Error", f"Data file missing required columns: {', '.join(missing)}")
                self.statusBar().showMessage("Error: Data file missing required columns.")
        else:
            QMessageBox.warning(self, "Error", "Failed to load models or data. Check console for details.")
            self.statusBar().showMessage("Error loading models or data.")
            
        # Re-enable UI
        self.load_btn.setEnabled(True)
        
    def question_selected(self, row):
        """
        Update the UI when a question is selected.
        This only updates the question display, not the analysis.
        """
        if row >= 0 and self.data_df is not None:
            # Get the selected question text
            question_text = self.questions_list.item(row).text()
            self.current_question = question_text
            
            # Find the entry for this question
            question_entry = self.data_df[self.data_df['question'] == question_text].iloc[0]
            
            # Store the correct root for this question
            self.current_root = question_entry['root']
            
            # Update the question and answer text fields
            self.question_text.setText(question_text)
            self.answer_text.setText(question_entry['answer'])
            self.generated_answer_text.clear()  # Clear the grounded answer
            self.ungrounded_answer_text.clear()  # Clear the ungrounded answer
            
            # Clear other displays until analysis is performed
            self.content_text.clear()
            self.root_combo.clear()
            self.root_indicator.setText("")
            
            # Clear the plot
            self.attn_ax.clear()
            self.attn_plot.draw()
            
            # Update status
            self.statusBar().showMessage(f"Selected question: '{question_text[:50]}...' with root '{self.current_root}'. Click 'Analyze' to perform analysis.")

    def plot_comparative_attention(self, root_scores):
        """
        Create a horizontal bar chart comparing sentence counts across roots.
        Highlight the correct root.
        """
        # Clear previous plot
        self.attn_ax.clear()
        
        # Skip if no valid data
        if not root_scores or len(root_scores) == 0:
            self.attn_ax.text(0.5, 0.5, "No sentences retrieved to display", 
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.attn_ax.transAxes, fontsize=12)
            self.attn_plot.draw()
            return
        
        # Sort roots by sentence count for better visualization
        sorted_items = sorted(root_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_items:  # Double-check to avoid empty sequences
            self.attn_ax.text(0.5, 0.5, "No sentence counts to display", 
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.attn_ax.transAxes, fontsize=12)
            self.attn_plot.draw()
            return
            
        roots = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        
        # Create label mapping for chunked roots
        display_labels = []
        colors = []
        
        for root in roots:
            # Check if this is a chunked root
            is_chunked = "_" in root and any(
                context_info.get('is_chunk', False) 
                for context_info in self.contexts_data 
                if context_info['root'] == root
            )
            
            if is_chunked:
                # Get chunk info
                for context_info in self.contexts_data:
                    if context_info['root'] == root:
                        original_root = context_info.get('original_root', root)
                        chunk_idx = context_info.get('chunk_idx', 0)
                        total_chunks = context_info.get('total_chunks', 0)
                        # Create a shorter label for the plot
                        display_label = f"{original_root} [{chunk_idx}/{total_chunks}]"
                        display_labels.append(display_label)
                        
                        # Determine color based on original root
                        if original_root == self.current_root:
                            colors.append('#28a745')  # Green for correct root
                        else:
                            colors.append('#007bff')  # Blue for others
                        break
                else:
                    # Fallback if chunk info not found
                    display_labels.append(root)
                    colors.append('#007bff')  # Default blue
            else:
                # Not chunked, use as is
                display_labels.append(root)
                if root == self.current_root:
                    colors.append('#28a745')  # Green for correct root
                else:
                    colors.append('#007bff')  # Blue for others
        
        # Create the horizontal bar chart
        bars = self.attn_ax.barh(range(len(roots)), counts, color=colors)
        self.attn_ax.set_yticks(range(len(roots)))
        self.attn_ax.set_yticklabels(display_labels)
        
        # Add rank indicators
        for i, (root, count, label) in enumerate(zip(roots, counts, display_labels)):
            rank_marker = f"#{i+1}: "
            
            # Check if this root or its original is the correct one
            is_correct = False
            if root == self.current_root:
                is_correct = True
            elif "_" in root:  # Check if it's a chunk
                for context_info in self.contexts_data:
                    if context_info['root'] == root and context_info.get('original_root') == self.current_root:
                        is_correct = True
                        break
            
            if is_correct:
                rank_marker += "✓ "
            
            # Add text annotation to the bar
            self.attn_ax.text(count + max(counts) * 0.01, i, str(count), va='center')
            
            # Replace the y-tick label with one that includes the rank
            self.attn_ax.get_yticklabels()[i].set_text(f"{rank_marker}{label}")
        
        # Set title and labels
        self.attn_ax.set_title("Sentences Retrieved by Root (Green = Correct Root)")
        self.attn_ax.set_xlabel("Number of Sentences")
        
        # Auto-adjust x-axis limits
        self.attn_ax.set_xlim(0, max(counts) * 1.15)  # Add 15% space for labels
        
        # Update the plot
        self.attn_plot.figure.tight_layout()
        self.attn_plot.draw()
    
    def highlight_content(self, sentences, scores=None, max_score=1.0, min_score=0.0):
        """
        Highlight the content using retrieved sentences rather than tokens.
        
        Args:
            sentences: List of sentence texts
            scores: Optional list of sentence scores for coloring
        """
        self.content_text.clear()
        
        if not sentences:
            self.content_text.setText("No sentences available to display.")
            return
            
        cursor = self.content_text.textCursor()
        
        # Format settings
        normal_format = QTextCharFormat()
        normal_format.setFontPointSize(10)
        
        # If no scores provided, use uniform coloring
        if scores is None:
            scores = [0.5] * len(sentences)
        
        # Normalize scores for coloring
        if len(scores) > 0:
            # max_score = max(scores)
            # min_score = min(scores)
            range_score = max_score - min_score
            if range_score > 0:
                normalized_scores = [(s - min_score) / range_score for s in scores]
            else:
                normalized_scores = [0.5] * len(scores)
        else:
            normalized_scores = []
        
        # Apply formats based on scores
        for i, (sentence, score) in enumerate(zip(sentences, normalized_scores)):
            # Get color from colormap
            color_rgba = self.cmap(score)
            color = QColor(
                int(color_rgba[0] * 255), 
                int(color_rgba[1] * 255), 
                int(color_rgba[2] * 255)
            )
            
            # Create format with background color
            sentence_format = QTextCharFormat(normal_format)
            sentence_format.setBackground(color)
            
            # For higher intensity backgrounds, make text white for readability
            if score > 0.7:
                sentence_format.setForeground(QColor("white"))
            else:
                sentence_format.setForeground(QColor("black"))
            
            # Insert sentence with format
            cursor.insertText(sentence, sentence_format)
            
            # Add a space between sentences
            if i < len(sentences) - 1:
                cursor.insertText(" ", normal_format)
        
        # Scroll to beginning
        self.content_text.moveCursor(QTextCursor.Start)
    
    def root_changed_for_visualization(self, index):
        """
        Update the content display when a different root is selected for viewing.
        This doesn't trigger new analysis, just shows different results.
        """
        if index >= 0:
            selected_root = self.root_combo.itemText(index)
            
            # Check if this is a chunked root
            is_chunked = "_" in selected_root and any(
                context_info.get('is_chunk', False) 
                for context_info in self.contexts_data 
                if context_info['root'] == selected_root
            )
            
            # For chunked roots, find the original root
            if is_chunked:
                # Get the chunk info
                for context_info in self.contexts_data:
                    if context_info['root'] == selected_root:
                        original_root = context_info.get('original_root', selected_root)
                        chunk_idx = context_info.get('chunk_idx', 0)
                        total_chunks = context_info.get('total_chunks', 0)
                        
                        # Update the indicator to show chunk information
                        if original_root == self.current_root:
                            self.root_indicator.setText(f"✓ CORRECT ROOT (Chunk {chunk_idx}/{total_chunks})")
                            self.root_indicator.setStyleSheet("color: green; font-weight: bold;")
                        else:
                            self.root_indicator.setText(f"❌ INCORRECT ROOT (Chunk {chunk_idx}/{total_chunks})")
                            self.root_indicator.setStyleSheet("color: red; font-weight: bold;")
                        break
            else:
                # Update the indicator label to show if this is the correct root
                if selected_root == self.current_root:
                    self.root_indicator.setText("✓ CORRECT ROOT")
                    self.root_indicator.setStyleSheet("color: green; font-weight: bold;")
                else:
                    self.root_indicator.setText("❌ INCORRECT ROOT")
                    self.root_indicator.setStyleSheet("color: red; font-weight: bold;")
            
            # If we have analysis results, display this root's sentences with highlighting
            if selected_root in self.analysis_results:
                result = self.analysis_results[selected_root]
                # get max score from all roots
                max_score = max([max(r['scores']) for r in self.analysis_results.values()])
                # get min score from all roots
                min_score = min([min(r['scores']) for r in self.analysis_results.values()])
                
                # Use top sentences for display
                sentences = result.get('sentences', [])
                scores = result.get('scores', [])
                
                # Display with highlighting
                self.highlight_content(sentences, scores, max_score, min_score)
                
                # Add sentence count info
                sentence_count = result.get('sentence_count', 0)
                if sentence_count > 0:
                    count_text = f"Retrieved {sentence_count} sentences from this root"
                    self.statusBar().showMessage(count_text)
            else:
                self.content_text.setText("No analysis data available for this root.")
    
    def analysis_finished(self, success, root_scores, tokens_dict, scores_dict, 
                         generated_answer, ungrounded_answer):
        # Re-enable UI
        self.analyze_btn.setEnabled(True)
        self.questions_list.setEnabled(True)
        
        if not success:
            QMessageBox.warning(self, "Error", "Failed to analyze sentences.")
            self.statusBar().showMessage("Error analyzing sentences.")
            return
        
        # Store and display both answers
        self.generated_answer = generated_answer
        self.generated_answer_text.setText(generated_answer)
        
        self.ungrounded_answer = ungrounded_answer
        self.ungrounded_answer_text.setText(ungrounded_answer)
        
        # Prepare analysis results for visualization
        self.analysis_results = {}
        for root in tokens_dict.keys():
            sentences = tokens_dict[root]  # Now sentences instead of tokens
            scores = scores_dict[root]  # Scores for sentences
            
            # Count sentences retrieved from this root
            sentence_count = root_scores.get(root, 0)
            
            # Store for visualization
            self.analysis_results[root] = {
                'sentences': sentences,
                'scores': scores,
                'sentence_count': sentence_count,
                'is_correct': (root == self.current_root)
            }
        
        # Show the content for the currently selected root
        current_index = self.root_combo.currentIndex()
        if current_index >= 0:
            self.root_changed_for_visualization(current_index)
        
        # Plot comparative sentence counts distribution
        self.plot_comparative_attention(root_scores)
        
        # Find the top-scoring root (most sentences retrieved)
        if root_scores:
            top_root = max(root_scores.items(), key=lambda x: x[1])[0]
            is_correct = (top_root == self.current_root)
            result_status = "correct" if is_correct else "incorrect"
            
            self.statusBar().showMessage(
                f"Analysis complete. Most sentences retrieved from root ({top_root}) is {result_status}. " + 
                f"Select different roots to compare retrieved sentences."
            )
        else:
            self.statusBar().showMessage(
                "Analysis complete, but no roots contributed sentences. " +
                "Try with different parameters or check for errors."
            )
    
    def start_analysis(self):
        """
        Start the analysis with all contexts using the sentence retrieval approach.
        """
        if self.current_question is None or self.current_root is None:
            QMessageBox.warning(self, "Error", "Please select a question first.")
            return
        
        # Check if both models are loaded
        if self.retriever_model is None or self.answerer_model is None:
            QMessageBox.warning(self, "Error", "Please load both retriever and answerer models first.")
            return
        
        # Get the number of contexts to sample
        num_contexts = self.contexts_spin.value()
        
        # Sample random roots from the roots dataset (excluding the correct one)
        other_roots = [root for root in self.all_roots if root != self.current_root]
        
        # Make sure we don't try to sample more than available
        num_to_sample = min(num_contexts - 1, len(other_roots))
        
        if num_to_sample <= 0:
            QMessageBox.warning(self, "Error", "Not enough unique roots available for sampling.")
            return
        
        sampled_others = random.sample(other_roots, num_to_sample)
        
        # Combine with the correct root
        self.sampled_roots = [self.current_root] + sampled_others
        
        # Shuffle to avoid bias
        random.shuffle(self.sampled_roots)
        
        # We'll populate the combo box after creating chunks
        
        # Get the actual answer for the current question
        question_entry = self.data_df[self.data_df['question'] == self.current_question].iloc[0]
        actual_answer = question_entry['answer']
        
        # Prepare context data for each root, with chunking for long contexts
        self.contexts_data = []  # Store at class level so other methods can access
        # Maximum effective sequence length, allowing space for the prompt
        effective_seq_len = self.seq_len_spin.value() - 500
        
        # Print debugging information
        print(f"Preparing contexts for {len(self.sampled_roots)} roots")
        print(f"Current root: {self.current_root}")
        if self.roots_df is not None:
            print(f"Roots dataset available with {len(self.roots_df)} entries")
            if 'root' in self.roots_df.columns and 'context' in self.roots_df.columns:
                roots_in_roots_df = set(self.roots_df['root'].unique())
                print(f"Roots dataset has {len(roots_in_roots_df)} unique roots")
                # Check how many of our sampled roots are in the roots dataset
                sampled_in_roots = [root for root in self.sampled_roots if root in roots_in_roots_df]
                print(f"{len(sampled_in_roots)}/{len(self.sampled_roots)} sampled roots found in roots dataset")
            else:
                print("Warning: Roots dataset missing required columns")
        
        successful_roots = []
        # Process each sampled root
        for root in self.sampled_roots:
            context = None
            is_correct = (root == self.current_root)
            context_source = "unknown"
            
            if is_correct:
                # For the correct root, always use context from the question dataset
                root_entries = self.data_df[self.data_df['root'] == root]
                if not root_entries.empty:
                    entry = root_entries.iloc[0]
                    context = entry['context']
                    context_source = "questions dataset (correct root)"
            else:
                # For other roots, use context from the roots dataset if available
                if self.roots_df is not None and 'root' in self.roots_df.columns and 'context' in self.roots_df.columns:
                    root_entries = self.roots_df[self.roots_df['root'] == root]
                    if not root_entries.empty:
                        entry = root_entries.iloc[0]
                        context = entry['context']
                        context_source = "roots dataset"
                
                # If not found in roots dataset or no roots dataset, fall back to questions dataset
                if context is None:
                    root_entries = self.data_df[self.data_df['root'] == root]
                    if not root_entries.empty:
                        entry = root_entries.iloc[0]
                        context = entry['context']
                        context_source = "questions dataset (fallback)"
            
            # Skip if no context found for this root
            if context is None:
                print(f"Warning: No context found for root '{root}', skipping")
                continue
                
            print(f"Found context for root '{root}' from {context_source} ({len(context)} chars)")
            successful_roots.append(root)
            
        # Update sampled_roots to only include those with successful contexts
        self.sampled_roots = successful_roots
        if not self.sampled_roots:
            print("ERROR: No roots have valid contexts available!")
            return
            
        # Process the successfully found contexts and split long ones into chunks
        self.contexts_data = []  # Clear any existing contexts
        for root in self.sampled_roots:
            # Find context for this root again - we know it exists
            context = None
            is_correct = (root == self.current_root)
            
            if is_correct:
                # For the correct root, use context from the question dataset
                root_entries = self.data_df[self.data_df['root'] == root]
                if not root_entries.empty:
                    entry = root_entries.iloc[0]
                    context = entry['context']
            else:
                # For other roots, try the roots dataset first
                if self.roots_df is not None and 'root' in self.roots_df.columns and 'context' in self.roots_df.columns:
                    root_entries = self.roots_df[self.roots_df['root'] == root]
                    if not root_entries.empty:
                        entry = root_entries.iloc[0]
                        context = entry['context']
                
                # Fallback to questions dataset if needed
                if context is None:
                    root_entries = self.data_df[self.data_df['root'] == root]
                    if not root_entries.empty:
                        entry = root_entries.iloc[0]
                        context = entry['context']
                        
            # We should always have a context here since we've already filtered
            if context is None:
                print(f"Strange error: Lost context for root {root}")
                continue
                
            # Handle long contexts with chunking
            # Approximate character length that would fit in the sequence length
            approx_token_char_ratio = 3.5  # Assumes average of 3.5 chars per token for Arabic
            max_chars = int(effective_seq_len * approx_token_char_ratio)  # Ensure it's an integer
            
            if len(context) > max_chars:
                print(f"Long context detected for {root} ({len(context)} chars). Splitting into chunks.")
                # Split into chunks with some overlap
                overlap = int(max_chars * 0.1)  # 10% overlap between chunks
                chunk_size = int(max_chars - overlap)  # Ensure chunk_size is an integer
                
                chunks = []
                for i in range(0, len(context), chunk_size):
                    # Get chunk with overlap
                    if i > 0:
                        chunk_start = i - overlap
                    else:
                        chunk_start = 0
                    
                    chunk_end = min(i + chunk_size, len(context))
                    chunk = context[chunk_start:chunk_end]
                    chunks.append(chunk)
                
                # Add each chunk as a separate context entry
                for idx, chunk in enumerate(chunks):
                    chunked_root = f"{root}_{idx+1}"
                    self.contexts_data.append({
                        'root': chunked_root,
                        'context': chunk,
                        'is_correct': is_correct,
                        'original_root': root,
                        'is_chunk': True,
                        'chunk_idx': idx+1,
                        'total_chunks': len(chunks)
                    })
                    print(f"Created chunk {idx+1}/{len(chunks)} for {root} with {len(chunk)} chars")
            else:
                # No chunking needed
                self.contexts_data.append({
                    'root': root,
                    'context': context,
                    'is_correct': is_correct,
                    'original_root': root,
                    'is_chunk': False
                })
        
        # Check if we have any contexts
        if not self.contexts_data:
            QMessageBox.warning(self, "Error", "No contexts could be found for the selected roots. Analysis cannot proceed.")
            return
        
        # Update combo box with all context roots (including chunked ones)
        self.root_combo.clear()
        all_roots = [context_info['root'] for context_info in self.contexts_data]
        self.root_combo.addItems(all_roots)
        
        # Get the correct answer from the data_df
        question_entry = self.data_df[self.data_df['question'] == self.current_question].iloc[0]
        correct_answer = question_entry['answer']
        
        # Start analysis thread with both models
        self.analyze_thread = MultiRootAnalysisThread(
            self.retriever_model,
            self.retriever_tokenizer,
            self.answerer_model,
            self.answerer_tokenizer,
            self.current_question,
            self.contexts_data,
            self.retriever_device,
            self.answerer_device,
            self.seq_len_spin.value(),
            correct_answer  # Use the answer from the question data
        )
        self.analyze_thread.progress_signal.connect(self.update_progress)
        self.analyze_thread.finished_signal.connect(self.analysis_finished)
        
        # Disable UI during analysis
        self.analyze_btn.setEnabled(False)
        self.questions_list.setEnabled(False)
        self.statusBar().showMessage("Analyzing contexts using sentence retrieval...")
        self.progress_bar.setValue(0)
        
        # Start the thread
        self.analyze_thread.start()

class DualModelLoaderThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, object, object, object, object, object, object)
    
    def __init__(self, retriever_path, answerer_path, data_path, roots_path,
                 retriever_quantization, answerer_quantization, 
                 max_seq_len, retriever_device, answerer_device):
        super().__init__()
        self.retriever_path = retriever_path
        self.answerer_path = answerer_path
        self.data_path = data_path
        self.roots_path = roots_path
        self.retriever_quantization = retriever_quantization
        self.answerer_quantization = answerer_quantization
        self.max_seq_len = max_seq_len
        self.retriever_device = retriever_device
        self.answerer_device = answerer_device
        
    def load_model(self, model_path, quantization, device, model_type="retriever"):
        """Helper function to load a model with specified configuration."""
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Configure quantization
        quantization_config = None
        torch_dtype = None
        
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        elif quantization == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
        # Set attn_implementation based on model type
        # For answerer, we don't need attention matrices, so we can use default impl
        attn_impl = "eager" if model_type == "retriever" else "sdpa"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            output_attentions=(model_type == "retriever"),  # Only need output_attentions for retriever
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            attn_implementation=attn_impl
        ).to(device)
        
        model.eval()  # Set to evaluation mode
        
        return model, tokenizer
        
    def run(self):
        try:
            # Load retriever model
            self.progress_signal.emit(10, "Loading retriever tokenizer...")
            retriever_model, retriever_tokenizer = self.load_model(
                self.retriever_path, 
                self.retriever_quantization, 
                self.retriever_device,
                "retriever"
            )
            
            # Load answerer model
            self.progress_signal.emit(40, "Loading answerer tokenizer...")
            answerer_model, answerer_tokenizer = self.load_model(
                self.answerer_path, 
                self.answerer_quantization, 
                self.answerer_device,
                "answerer"
            )
            
            # Load questions dataset
            self.progress_signal.emit(80, "Loading questions dataset...")
            data_df = load_datasets_from_hf(self.data_path)
            
            # Load roots dataset 
            self.progress_signal.emit(90, "Loading roots dataset...")
            try:
                roots_df = load_datasets_from_hf(self.roots_path)
                # Ensure the roots dataset has the required columns
                if 'train' in roots_df:
                    roots_df = pd.DataFrame(roots_df['train'])
                else:
                    # Use first split if train isn't available
                    first_split = list(roots_df.keys())[0]
                    roots_df = pd.DataFrame(roots_df[first_split])
                
                # Check if the roots dataset has the required columns
                required_columns = ['root', 'content']
                missing = [col for col in required_columns if col not in roots_df.columns]
                if missing:
                    self.progress_signal.emit(95, f"Warning: Roots dataset missing columns: {', '.join(missing)}")
                    roots_df = None
                else:
                    # rename content to context
                    roots_df = roots_df.rename(columns={'content': 'context'})
            except Exception as e:
                print(f"Error loading roots dataset: {e}")
                self.progress_signal.emit(95, f"Warning: Failed to load roots dataset: {e}")
                roots_df = None
            
            self.progress_signal.emit(100, "Loading complete!")
            self.finished_signal.emit(
                True, 
                retriever_model, 
                retriever_tokenizer, 
                answerer_model, 
                answerer_tokenizer, 
                data_df,
                roots_df
            )
            
        except Exception as e:
            print(f"Error in dual model loading thread: {e}")
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, None, None, None, None, None, None)

def main():
    app = QApplication(sys.argv)
    window = AttentionVisualizerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()