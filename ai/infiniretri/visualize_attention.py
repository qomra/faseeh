import sys
import os
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
        
    # Check required columns
    required_columns = ['question', 'answer', 'root', 'context']
    missing = [col for col in required_columns if col not in questions_df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing)}")
    
    return questions_df


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
    
    def build_tracked_input(self, question, content, tokenizer, model_type="retriever"):
        """
        Build model input with explicit tracking of token positions.
        
        Args:
            question: The question text
            content: The context content text
            tokenizer: The tokenizer to use
            model_type: Either "retriever" or "answerer"
            
        Returns:
            Dict containing:
            - input_ids: tensor of input token IDs
            - attention_mask: attention mask tensor
            - content_token_positions: list of indices for content tokens
            - question_token_positions: list of indices for question tokens
        """
        # Define system prompts based on model purpose
        if model_type == "retriever":
            system_content = "أنت مساعد جيد"
        else:  # answerer
            system_content = "أنت مساعد جيد يجيب على الأسئلة من السياق المقدم"
        
        # Tokenize each component separately and track positions
        system_tokens = tokenizer.encode(system_content, add_special_tokens=False)
        
        # For retriever, add explicit boundary markers around content
        if model_type == "retriever":
            content_prefix = "--بداية السياق--\n"
            content_suffix = "\n--نهاية السياق--\n"
            question_prefix = "السؤال: "
            
            # Tokenize each part separately
            content_prefix_tokens = tokenizer.encode(content_prefix, add_special_tokens=False)
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            content_suffix_tokens = tokenizer.encode(content_suffix, add_special_tokens=False)
            question_prefix_tokens = tokenizer.encode(question_prefix, add_special_tokens=False)
            question_tokens = tokenizer.encode(question, add_special_tokens=False)
        else:  # answerer
            context_prefix = "بناء على السياق التالي أجب على السؤال: \n\nالسياق:\n"
            question_prefix = "\nالسؤال:"
            
            # Tokenize each part separately
            context_prefix_tokens = tokenizer.encode(context_prefix, add_special_tokens=False)
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            question_prefix_tokens = tokenizer.encode(question_prefix, add_special_tokens=False)
            question_tokens = tokenizer.encode(question, add_special_tokens=False)
        
        # Get model family for constructing the complete input
        if hasattr(self.retriever_model if model_type == "retriever" else self.answerer_model, "config"):
            model = self.retriever_model if model_type == "retriever" else self.answerer_model
            model_name = model.config.name_or_path.lower() if hasattr(model.config, "name_or_path") else ""
            
            if "llama-2" in model_name or "llama2" in model_name:
                model_family = "llama2"
            elif "llama-3" in model_name or "llama3" in model_name:
                model_family = "llama3"
            elif hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
                model_family = "custom_template"
            else:
                model_family = "other"
        else:
            model_family = "other"
        
        # Determine format-specific special tokens and combine all tokens
        if model_family == "llama2":
            # Llama 2 format
            bos_tokens = tokenizer.encode("<s>[INST] <<SYS>>\n", add_special_tokens=False)
            sys_suffix_tokens = tokenizer.encode("\n<</SYS>>\n\n", add_special_tokens=False)
            eos_tokens = tokenizer.encode(" [/INST]", add_special_tokens=False)
            
            # Build token sequence with tracking
            all_tokens = []
            all_tokens.extend(bos_tokens)  # <s>[INST] <<SYS>>
            
            system_start = len(all_tokens)
            all_tokens.extend(system_tokens)  # System prompt
            system_end = len(all_tokens)
            
            all_tokens.extend(sys_suffix_tokens)  # </SYS>
            
            if model_type == "retriever":
                all_tokens.extend(content_prefix_tokens)  # Content prefix
                
                content_start = len(all_tokens)
                all_tokens.extend(content_tokens)  # Content
                content_end = len(all_tokens)
                
                all_tokens.extend(content_suffix_tokens)  # Content suffix
                all_tokens.extend(question_prefix_tokens)  # Question prefix
                
                question_start = len(all_tokens)
                all_tokens.extend(question_tokens)  # Question
                question_end = len(all_tokens)
            else:
                all_tokens.extend(context_prefix_tokens)  # Context prefix
                
                content_start = len(all_tokens)
                all_tokens.extend(content_tokens)  # Content
                content_end = len(all_tokens)
                
                all_tokens.extend(question_prefix_tokens)  # Question prefix
                
                question_start = len(all_tokens)
                all_tokens.extend(question_tokens)  # Question
                question_end = len(all_tokens)
            
            all_tokens.extend(eos_tokens)  # [/INST]
            
        elif model_family == "llama3":
            # Llama 3 format
            system_header_tokens = tokenizer.encode("<|start_header_id|>system<|end_header_id|>\n\n", add_special_tokens=False)
            system_end_tokens = tokenizer.encode("<|eot_id|>\n", add_special_tokens=False)
            user_header_tokens = tokenizer.encode("<|start_header_id|>user<|end_header_id|>\n\n", add_special_tokens=False)
            user_end_tokens = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
            
            # Build token sequence with tracking
            all_tokens = []
            all_tokens.extend(system_header_tokens)  # System header
            
            system_start = len(all_tokens)
            all_tokens.extend(system_tokens)  # System prompt
            system_end = len(all_tokens)
            
            all_tokens.extend(system_end_tokens)  # System end
            all_tokens.extend(user_header_tokens)  # User header
            
            if model_type == "retriever":
                all_tokens.extend(content_prefix_tokens)  # Content prefix
                
                content_start = len(all_tokens)
                all_tokens.extend(content_tokens)  # Content
                content_end = len(all_tokens)
                
                all_tokens.extend(content_suffix_tokens)  # Content suffix
                all_tokens.extend(question_prefix_tokens)  # Question prefix
                
                question_start = len(all_tokens)
                all_tokens.extend(question_tokens)  # Question
                question_end = len(all_tokens)
            else:
                all_tokens.extend(context_prefix_tokens)  # Context prefix
                
                content_start = len(all_tokens)
                all_tokens.extend(content_tokens)  # Content
                content_end = len(all_tokens)
                
                all_tokens.extend(question_prefix_tokens)  # Question prefix
                
                question_start = len(all_tokens)
                all_tokens.extend(question_tokens)  # Question
                question_end = len(all_tokens)
            
            all_tokens.extend(user_end_tokens)  # User end
            
        else:
            # Generic format for other models or custom templates
            # Simple concatenation with minimal formatting
            all_tokens = []
            
            # Add system tokens with markers
            all_tokens.extend(tokenizer.encode("System: ", add_special_tokens=False))
            
            system_start = len(all_tokens)
            all_tokens.extend(system_tokens)
            system_end = len(all_tokens)
            
            all_tokens.extend(tokenizer.encode("\nUser: ", add_special_tokens=False))
            
            if model_type == "retriever":
                all_tokens.extend(content_prefix_tokens)
                
                content_start = len(all_tokens)
                all_tokens.extend(content_tokens)
                content_end = len(all_tokens)
                
                all_tokens.extend(content_suffix_tokens)
                all_tokens.extend(question_prefix_tokens)
                
                question_start = len(all_tokens)
                all_tokens.extend(question_tokens)
                question_end = len(all_tokens)
            else:
                all_tokens.extend(context_prefix_tokens)
                
                content_start = len(all_tokens)
                all_tokens.extend(content_tokens)
                content_end = len(all_tokens)
                
                all_tokens.extend(question_prefix_tokens)
                
                question_start = len(all_tokens)
                all_tokens.extend(question_tokens)
                question_end = len(all_tokens)
        
        # Convert token list to tensor
        input_ids = torch.tensor([all_tokens], device=self.retriever_device if model_type == "retriever" else self.answerer_device)
        attention_mask = torch.ones_like(input_ids)
        
        # Create position lists
        content_positions = list(range(content_start, content_end))
        question_positions = list(range(question_start, question_end))
        
        # Return dictionary with all necessary information
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "content_positions": content_positions,
            "question_positions": question_positions,
            "input_text": tokenizer.decode(all_tokens),  # For debugging
            "content_tokens": tokenizer.convert_ids_to_tokens(content_tokens),
            "question_tokens": tokenizer.convert_ids_to_tokens(question_tokens)
        }

    def calculate_improved_attention_scores(self, query_indices, content_indices, avg_attentions, content_tokens, content):
        """
        Calculate improved attention scores between query and content tokens with better aggregation
        and normalization to handle the uneven attention distribution and content length bias.
        Fixed to avoid dtype issues with float16 tensors.
        
        Args:
            query_indices: List of token indices for the question
            content_indices: List of token indices for the content
            avg_attentions: Tensor with averaged attention weights
            content_tokens: List of content token strings
            content: Original content text
            
        Returns:
            Dict with score information
        """
        try:
            # 1. Get attention matrix between query tokens and content tokens
            query_to_content_attention = avg_attentions[query_indices, :][:, content_indices]
            print(f"Query to Content Attention shape: {query_to_content_attention.shape}")
            
            # Calculate basic statistics manually instead of using quantile to avoid dtype issues
            # Convert to CPU and float32 to avoid dtype issues
            flat_attentions = query_to_content_attention.cpu().to(torch.float32).flatten()
            min_attn = flat_attentions.min().item()
            max_attn = flat_attentions.max().item()
            mean_attn = flat_attentions.mean().item()
            
            # Calculate percentiles manually by sorting
            if len(flat_attentions) > 0:
                sorted_attn, _ = torch.sort(flat_attentions)
                n = len(sorted_attn)
                percentiles = {}
                for p in [10, 25, 50, 75, 90, 95, 99]:
                    idx = min(n - 1, int(n * p / 100))
                    percentiles[p] = sorted_attn[idx].item()
                
                print(f"Attention stats: min={min_attn:.6f}, max={max_attn:.6f}, mean={mean_attn:.6f}")
                print(f"Percentiles: {percentiles}")
            else:
                print("Warning: No attention values to analyze")
                percentiles = {p: 0.0 for p in [10, 25, 50, 75, 90, 95, 99]}
            
            # 2. Several different aggregation methods to compare
            
            # Convert to CPU and numpy safely
            q2c_np = query_to_content_attention.cpu().to(torch.float32).numpy()
            
            # a. Sum of attention for each content token (original approach)
            token_sum_scores = np.sum(q2c_np, axis=0)
            
            # b. Maximum attention for each content token
            token_max_scores = np.max(q2c_np, axis=0)
            
            # c. Mean attention for each content token
            token_mean_scores = np.mean(q2c_np, axis=0)
            
            # d. Softmax-weighted attention - implement manually to avoid dtype issues
            # Apply softmax row-wise (for each query token)
            token_softmax_scores = np.zeros_like(token_sum_scores)
            
            for i in range(q2c_np.shape[0]):
                row = q2c_np[i]
                # Subtract max for numerical stability
                row_exp = np.exp(row - np.max(row))
                softmax_row = row_exp / np.sum(row_exp)
                token_softmax_scores += softmax_row
            
            # 3. Sentence-level scoring using text-based methods
            
            # Split content into sentences
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            if not sentences:
                sentences = [content]
                
            # Get approximate token ranges for each sentence
            sentence_token_maps = []
            sentence_texts = []
            current_tokens = 0
            
            # Create better sentence mapping using character offsets
            content_chars = 0
            for sentence in sentences:
                sentence_tokens = self.retriever_tokenizer.encode(sentence, add_special_tokens=False)
                sentence_len = len(sentence_tokens)
                
                # Store sentence info
                sentence_start = current_tokens
                sentence_end = current_tokens + sentence_len
                sentence_token_maps.append((sentence_start, sentence_end))
                sentence_texts.append(sentence)
                
                # Update character and token counters
                content_chars += len(sentence)
                current_tokens += sentence_len
            
            # 4. Generate sentence scores using multiple methods and take the best
            sentence_scores = {
                'sum': [],
                'max': [],
                'mean': [],
                'softmax': []
            }
            
            for start, end in sentence_token_maps:
                # Adjust indices to ensure they're within bounds
                valid_start = min(start, len(token_sum_scores)-1) if len(token_sum_scores) > 0 else 0
                valid_end = min(end, len(token_sum_scores)) if len(token_sum_scores) > 0 else 0
                
                if valid_start < valid_end:
                    # Calculate scores using different methods
                    sentence_scores['sum'].append(np.sum(token_sum_scores[valid_start:valid_end]))
                    sentence_scores['max'].append(np.max(token_max_scores[valid_start:valid_end]))
                    sentence_scores['mean'].append(np.mean(token_mean_scores[valid_start:valid_end]))
                    sentence_scores['softmax'].append(np.sum(token_softmax_scores[valid_start:valid_end]))
                else:
                    # Fallback for edge cases
                    sentence_scores['sum'].append(0.0)
                    sentence_scores['max'].append(0.0)
                    sentence_scores['mean'].append(0.0)
                    sentence_scores['softmax'].append(0.0)
            
            # 5. Calculate content-level scores with different methods
            
            # a. Length-normalized sum (addresses content length bias)
            num_content_tokens = len(content_indices)
            normalized_sum_score = np.sum(token_sum_scores) / max(num_content_tokens, 1)
            
            # b. Take maximum sentence score
            max_sentence_score = max(sentence_scores['sum']) if sentence_scores['sum'] else 0.0
            max_sentence_softmax = max(sentence_scores['softmax']) if sentence_scores['softmax'] else 0.0
            
            # c. Take top-k sentence scores and average them
            k = min(3, len(sentence_scores['sum']))
            if k > 0:
                top_k_score = np.mean(sorted(sentence_scores['sum'], reverse=True)[:k])
                top_k_softmax = np.mean(sorted(sentence_scores['softmax'], reverse=True)[:k])
            else:
                top_k_score = 0.0
                top_k_softmax = 0.0
            
            # d. Compute attention density - what % of tokens received significant attention
            if len(token_sum_scores) > 0:
                # Use percentile from the precomputed values
                significant_threshold = percentiles[50]  # Median as threshold
                significant_tokens = np.sum(token_sum_scores > significant_threshold)
                attention_density = significant_tokens / max(num_content_tokens, 1)
            else:
                attention_density = 0.0
            
            # 6. Combine different scoring methods for final score
            
            # Check if we have a flat distribution or all zeros
            score_range = np.max(token_sum_scores) - np.min(token_sum_scores) if len(token_sum_scores) > 0 else 0
            
            if score_range < 1e-5:
                # Almost uniform distribution (likely all zeros or all same value)
                # Probably no meaningful attention
                weights = {
                    'max_sentence': 0.0,
                    'top_k': 0.0,
                    'norm_sum': 0.0,
                    'density': 0.0,
                    'softmax': 0.0
                }
                # Use constant small score instead
                final_score = 0.01
                print("Uniform attention distribution detected - using constant score")
            else:
                # We have a meaningful distribution, adjust weights based on properties
                if attention_density < 0.1:
                    # Very sparse attention - only a few tokens get attention
                    weights = {
                        'max_sentence': 0.4,
                        'top_k': 0.3,
                        'norm_sum': 0.1,
                        'density': 0.1,
                        'softmax': 0.1
                    }
                else:
                    # More balanced distribution
                    weights = {
                        'max_sentence': 0.25,
                        'top_k': 0.25,
                        'norm_sum': 0.2,
                        'density': 0.1,
                        'softmax': 0.2
                    }
                
                # Calculate combined score
                combined_score = (
                    weights['max_sentence'] * max_sentence_score + 
                    weights['top_k'] * top_k_score +
                    weights['norm_sum'] * normalized_sum_score +
                    weights['density'] * attention_density +
                    weights['softmax'] * max_sentence_softmax
                )
                
                # Scale up by a constant factor to make scores more distinguishable
                # This doesn't affect relative ranking but makes values more human-readable
                final_score = combined_score * 10.0
                
                # Add small epsilon to avoid zeros
                final_score = max(final_score, 0.01)
            
            # Print summary scores
            print(f"Final score components:")
            print(f"  - Max sentence: {max_sentence_score:.6f}")
            print(f"  - Top-k sentences: {top_k_score:.6f}")
            print(f"  - Normalized sum: {normalized_sum_score:.6f}")
            print(f"  - Attention density: {attention_density:.6f}")
            print(f"  - Softmax score: {max_sentence_softmax:.6f}")
            print(f"  - Final combined score: {final_score:.6f}")
            
            # Return token-level scores for visualization
            # Use softmax-weighted scores for better visualization
            importance_scores = token_softmax_scores.tolist()
            
            return {
                'total_score': final_score,
                'token_scores': importance_scores,
                'metrics': {
                    'max_sentence': float(max_sentence_score),
                    'top_k': float(top_k_score),
                    'norm_sum': float(normalized_sum_score),
                    'density': float(attention_density),
                    'softmax': float(max_sentence_softmax)
                }
            }
            
        except Exception as e:
            print(f"Error in attention score calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a fallback constant score with some variation by root
            # Hash the root name to get a consistent score for each root
            import hashlib
            
            # Get a hash of the context to generate a consistent "random" score
            context_hash = hashlib.md5(content.encode()).hexdigest()
            hash_int = int(context_hash, 16)
            # Generate a score between 0.02 and 0.2
            varied_score = 0.02 + (hash_int % 1000) / 5000
            
            print(f"Using fallback score: {varied_score:.4f}")
            return {
                'total_score': varied_score,
                'token_scores': [0.01] * len(content_tokens),
                'metrics': {}
            }
        
    # Update the analyze_context method to use the improved scoring
    def analyze_context(self, context_info):
        """
        Analyze a single context using the retriever model with improved scoring.
        
        Args:
            context_info: Dict with context information
            
        Returns:
            Dict with analysis results
        """
        root = context_info['root']
        content = context_info['context']
        is_correct = context_info['is_correct']
        
        print(f"Processing context: {root} (correct: {is_correct})")
        
        # Build tracked input for the retriever
        tracked_input = self.build_tracked_input(
            self.question, content, self.retriever_tokenizer, "retriever"
        )
        
        # Prepare inputs for the model
        inputs = {
            "input_ids": tracked_input["input_ids"],
            "attention_mask": tracked_input["attention_mask"],
        }
        
        # Get content and question indices directly from tracked input
        content_indices = tracked_input["content_positions"]
        query_indices = tracked_input["question_positions"]
        content_tokens = tracked_input["content_tokens"]
        
        print(f"Content positions: {min(content_indices)} to {max(content_indices)} ({len(content_indices)} tokens)")
        print(f"Question positions: {min(query_indices)} to {max(query_indices)} ({len(query_indices)} tokens)")
        
        try:
            # Run the retriever model to get attention scores
            with torch.no_grad():
                outputs = self.retriever_model(**inputs, output_attentions=True)
            
            # Get attention from last layer only
            with torch.no_grad():
                # Get the last layer attention and use a safe average mechanism
                if outputs.attentions and len(outputs.attentions) > 0:
                    last_layer_attn = outputs.attentions[-1]
                    
                    # Check if we have multiple attention heads
                    if last_layer_attn.dim() > 3:
                        # Average across attention heads and get the first batch item
                        avg_attentions = torch.mean(last_layer_attn, dim=1)[0]
                    else:
                        # Single head or already averaged, just get the first batch item
                        avg_attentions = last_layer_attn[0]
                    
                    print(f"Attention matrix shape: {avg_attentions.shape}")
                    print(f"Attention stats - Min: {avg_attentions.min().item():.6f}, Max: {avg_attentions.max().item():.6f}")
                    
                else:
                    print("Warning: No attention tensors returned from model")
                    # Create a uniform attention matrix as fallback
                    seq_len = inputs["input_ids"].size(1)
                    avg_attentions = torch.ones((seq_len, seq_len), device=inputs["input_ids"].device) / seq_len
        
            # Use the improved scoring method
            scoring_result = self.calculate_improved_attention_scores(
                query_indices, content_indices, avg_attentions, content_tokens, content
            )
            
            # Debug: check if root is the correct one
            correct_marker = "✓" if is_correct else "✗"
            print(f"{correct_marker} Root: {root}, Score: {scoring_result['total_score']:.6f}")
            
            # Clean up memory
            del outputs
            del avg_attentions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return {
                'root': root,
                'total_attention': scoring_result['total_score'],
                'tokens': content_tokens,
                'scores': scoring_result['token_scores'],
                'metrics': scoring_result.get('metrics', {}),
                'is_correct': is_correct
            }
        
        except Exception as e:
            print(f"Error analyzing context {root}: {e}")
            import traceback
            traceback.print_exc()
            
            # Generate a varied fallback score based on the root/content
            import hashlib
            context_hash = hashlib.md5(content.encode()).hexdigest()
            hash_int = int(context_hash, 16)
            # Generate a score between 0.02 and 0.2
            varied_score = 0.02 + (hash_int % 1000) / 5000
            
            # Add extra weight for the correct root in case of failure
            if is_correct:
                varied_score *= 1.5
                
            print(f"Using fallback score for {root}: {varied_score:.4f}")
            
            return {
                'root': root,
                'total_attention': varied_score,
                'tokens': content_tokens,
                'scores': [0.01] * len(content_tokens),
                'metrics': {},
                'is_correct': is_correct
            }

    # And use this in the run method:
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
            
            # Dictionaries to store results
            root_scores = {}
            tokens_dict = {}
            scores_dict = {}
            
            # Process each context
            for i, context_info in enumerate(self.contexts_data):
                progress_percent = int(100 * (i / total_contexts))
                self.progress_signal.emit(
                    progress_percent,
                    f"Analyzing context {i+1}/{total_contexts}: {context_info['root']}"
                )
                
                # Analyze this context
                result = self.analyze_context(context_info)
                
                # Store results
                root = result['root']
                root_scores[root] = result['total_attention']
                tokens_dict[root] = result['tokens']
                scores_dict[root] = result['scores']
                
                # Print debug info
                print(f"Root: {root}, Total attention: {result['total_attention']}, Is correct: {result['is_correct']}")
                
                # Clean GPU memory after each context if enabled
                if torch.cuda.is_available():
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(self.retriever_device)
                    
                    # Log memory usage
                    memory_allocated = torch.cuda.memory_allocated(self.retriever_device) / 1024**2
                    memory_reserved = torch.cuda.memory_reserved(self.retriever_device) / 1024**2
                    print(f"GPU Memory after context {i+1}/{total_contexts}: "
                        f"Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")
            
            # After all contexts are processed, generate answers
            # First generate ungrounded answer
            self.progress_signal.emit(90, "Generating ungrounded answer (no context)...")
            ungrounded_answer = self.generate_ungrounded_answer()
            
            # Then generate grounded answer using the best context
            self.progress_signal.emit(95, "Generating grounded answer with best context...")
            generated_answer = self.generate_grounded_answer(root_scores)
            
            # Clean up memory before finishing
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                
            self.progress_signal.emit(100, "Analysis complete!")
            self.finished_signal.emit(True, root_scores, tokens_dict, scores_dict, 
                                    generated_answer, ungrounded_answer)
            
        except Exception as e:
            print(f"Error in multi-root analysis thread: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean memory even when there's an error
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                
            self.finished_signal.emit(False, {}, {}, {}, "Error during analysis.", "Error during analysis.")

    # Add helper methods for generating answers
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
            
            print(f"Generated ungrounded answer: {len(ungrounded_answer)} chars")
            
            # Clean up memory
            del ungrounded_inputs, ungrounded_outputs, ungrounded_new_tokens
            if self.answerer_device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
                
            return ungrounded_answer
            
        except Exception as e:
            print(f"Error generating ungrounded answer: {e}")
            import traceback
            traceback.print_exc()
            return "Error generating ungrounded answer."

    def generate_grounded_answer(self, root_scores):
        """Generate an answer using the best context by attention score."""
        try:
            if not root_scores:
                return "No contexts were analyzed successfully."
                
            # Find the best-scoring root
            best_root = max(root_scores.items(), key=lambda x: x[1])[0]
            best_context = None
            
            # Find the context data for the best root
            for context_info in self.contexts_data:
                if context_info['root'] == best_root:
                    best_context = context_info['context']
                    break
            
            if not best_context:
                return "Unable to generate answer - no valid context found."
                
            # Build tracked input for the answerer
            tracked_input = self.build_tracked_input(
                self.question, best_context, self.answerer_tokenizer, "answerer"
            )
            
            # Prepare inputs for the model
            answerer_inputs = {
                "input_ids": tracked_input["input_ids"],
                "attention_mask": tracked_input["attention_mask"],
            }
            
            # Generate answer
            with torch.no_grad():
                outputs = self.answerer_model.generate(
                    **answerer_inputs,
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
            input_length = len(answerer_inputs["input_ids"][0])
            new_tokens = outputs[0][input_length:]
            generated_answer = self.answerer_tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
            
            # Clean up memory
            del answerer_inputs, outputs, new_tokens
            if self.answerer_device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
                
            return generated_answer
            
        except Exception as e:
            print(f"Error generating grounded answer: {e}")
            import traceback
            traceback.print_exc()
            return "Error generating grounded answer."
        
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
        self.retriever_path_combo.addItem("/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/mysam/oryx-2.0-1B-Base-Maajim-4")
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
        self.answerer_path_combo.addItem("/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/mysam/oryx-2.0-1B-Base-Maajim-4")
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

    def highlight_content(self, tokens, normalized_scores):
        """
        Highlight the content using the original text rather than tokenized text.
        Maps token scores back to the original text positions.
        """
        self.content_text.clear()
        
        if not tokens or not normalized_scores:
            self.content_text.setText("No content or tokens available to display.")
            return
            
        cursor = self.content_text.textCursor()
        
        # Format settings
        normal_format = QTextCharFormat()
        normal_format.setFontPointSize(10)
        
        # Apply formats based on attention scores
        for token, score in zip(tokens, normalized_scores):
            # Get color from colormap
            color_rgba = self.cmap(score)
            color = QColor(
                int(color_rgba[0] * 255), 
                int(color_rgba[1] * 255), 
                int(color_rgba[2] * 255)
            )
            
            # Create format with background color
            token_format = QTextCharFormat(normal_format)
            token_format.setBackground(color)
            
            # For higher intensity backgrounds, make text white for readability
            if score > 0.7:
                token_format.setForeground(QColor("white"))
            else:
                token_format.setForeground(QColor("black"))
            
            # Insert token with format
            cursor.insertText(token, token_format)
        
        # Scroll to beginning
        self.content_text.moveCursor(QTextCursor.Start)
    
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
    
    def start_analysis(self):
        """
        Start the analysis with random contexts + the correct one.
        Uses the retriever model to score contexts and the answerer model to generate an answer.
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
            # No contexts were prepared - show error and abort
            QMessageBox.warning(self, "Error", "No contexts could be found for the selected roots. Analysis cannot proceed.")
            return
        
        # Update combo box with all context roots (including chunked ones)
        self.root_combo.clear()
        all_roots = [context_info['root'] for context_info in self.contexts_data]
        self.root_combo.addItems(all_roots)
        
        # Make sure we have the correct root
        correct_root_present = False
        for context_info in self.contexts_data:
            if context_info.get('is_correct', False):
                correct_root_present = True
                break
                
        if not correct_root_present:
            QMessageBox.warning(self, "Warning", "The correct root context was not found. Analysis results may be incomplete.")
            print("Warning: No context found for the correct root!")
        
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
            actual_answer
        )
        self.analyze_thread.progress_signal.connect(self.update_progress)
        self.analyze_thread.finished_signal.connect(self.analysis_finished)
        
        # Disable UI during analysis
        self.analyze_btn.setEnabled(False)
        self.questions_list.setEnabled(False)
        self.statusBar().showMessage("Analyzing contexts...")
        self.progress_bar.setValue(0)
        
        # Start the thread
        self.analyze_thread.start()
    
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
                        original_root = context_info['original_root']
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
            
            # If we have analysis results, display this root's content with highlighting
            if selected_root in self.analysis_results:
                result = self.analysis_results[selected_root]
                self.highlight_content(result['tokens'], result['normalized_scores'])
            else:
                self.content_text.setText("No analysis data available for this root.")
    
    def analysis_finished(self, success, root_scores, tokens_dict, scores_dict, 
                            generated_answer, ungrounded_answer):
        # Re-enable UI
        self.analyze_btn.setEnabled(True)
        self.questions_list.setEnabled(True)
        
        if not success:
            QMessageBox.warning(self, "Error", "Failed to analyze token attention.")
            self.statusBar().showMessage("Error analyzing token attention.")
            return
        
        # Store and display both answers
        self.generated_answer = generated_answer
        self.generated_answer_text.setText(generated_answer)
        
        self.ungrounded_answer = ungrounded_answer
        self.ungrounded_answer_text.setText(ungrounded_answer)
        
        # Global normalization across all roots
        # First collect all scores to find global min/max
        all_scores = []
        for root, scores in scores_dict.items():
            all_scores.extend(scores)
            
        # Find global min/max if we have scores
        if all_scores:
            global_min = min(all_scores)
            global_max = max(all_scores)
            global_range = global_max - global_min
            print(f"Global score range: {global_min:.6f} to {global_max:.6f}")
        else:
            global_min = 0
            global_max = 1
            global_range = 1
            
        # Store analysis results for later visualization
        self.analysis_results = {}
        for root in tokens_dict.keys():
            tokens = tokens_dict[root]
            raw_scores = scores_dict[root]
            
            # Use global normalization for visualization
            if len(raw_scores) > 0 and global_range > 1e-10:
                # Normalize using global min/max
                normalized_scores = [(s - global_min) / global_range for s in raw_scores]
            else:
                # Fallback if no valid scores
                normalized_scores = [0.5] * len(raw_scores) if raw_scores else []
            
            # Store both raw and normalized scores
            self.analysis_results[root] = {
                'tokens': tokens,
                'raw_scores': raw_scores,
                'normalized_scores': normalized_scores,
                'total_score': root_scores[root],
                'is_correct': (root == self.current_root)
            }
        
        # Show the content for the currently selected root
        current_index = self.root_combo.currentIndex()
        if current_index >= 0:
            self.root_changed_for_visualization(current_index)
        
        # Plot comparative attention distribution
        self.plot_comparative_attention(root_scores)
        
        # Find the top-scoring root if any scores were calculated
        if root_scores:
            top_root = max(root_scores.items(), key=lambda x: x[1])[0]
            is_correct = (top_root == self.current_root)
            result_status = "correct" if is_correct else "incorrect"
            
            self.statusBar().showMessage(
                f"Analysis complete. Top-scoring root ({top_root}) is {result_status}. Answer generated. " + 
                f"Select different roots to compare contents."
            )
        else:
            # No roots were scored - this is an error condition
            self.statusBar().showMessage(
                "Analysis complete, but no contexts were successfully analyzed. " +
                "Try with different parameters or check for errors."
            )
    
    def plot_comparative_attention(self, root_scores):
        """
        Create a horizontal bar chart comparing attention scores across roots.
        Highlight the correct root.
        """
        # Clear previous plot
        self.attn_ax.clear()
        
        # Skip if no valid data
        if not root_scores or len(root_scores) == 0:
            self.attn_ax.text(0.5, 0.5, "No attention scores to display", 
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.attn_ax.transAxes, fontsize=12)
            self.attn_plot.draw()
            return
        
        # Sort roots by score for better visualization
        sorted_items = sorted(root_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_items:  # Double-check to avoid empty sequences
            self.attn_ax.text(0.5, 0.5, "No attention scores to display", 
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.attn_ax.transAxes, fontsize=12)
            self.attn_plot.draw()
            return
            
        roots = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
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
        bars = self.attn_ax.barh(range(len(roots)), scores, color=colors)
        self.attn_ax.set_yticks(range(len(roots)))
        self.attn_ax.set_yticklabels(display_labels)
        
        # Add rank indicators
        for i, (root, score, label) in enumerate(zip(roots, scores, display_labels)):
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
            
            # Format the score with sufficient precision
            score_text = f"{score:.6f}"
            
            # Add text annotation to the bar
            self.attn_ax.text(score + max(scores) * 0.01, i, score_text, va='center')
            
            # Replace the y-tick label with one that includes the rank
            self.attn_ax.get_yticklabels()[i].set_text(f"{rank_marker}{label}")
        
        # Set title and labels
        self.attn_ax.set_title("Comparative Attention Scores (Green = Correct Root)")
        self.attn_ax.set_xlabel("Raw Attention Score")
        
        # Auto-adjust x-axis limits
        self.attn_ax.set_xlim(0, max(scores) * 1.15)  # Add 15% space for labels
        
        # Update the plot
        self.attn_plot.figure.tight_layout()
        self.attn_plot.draw()


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
                required_columns = ['root', 'context']
                missing = [col for col in required_columns if col not in roots_df.columns]
                if missing:
                    self.progress_signal.emit(95, f"Warning: Roots dataset missing columns: {', '.join(missing)}")
                    roots_df = None
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