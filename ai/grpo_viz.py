import sys
import os
import torch
import pandas as pd
import numpy as np
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QListWidget, QTextEdit, QPushButton, 
                            QLabel, QComboBox, QFileDialog, QProgressBar, 
                            QSplitter, QGridLayout, QGroupBox, QMessageBox,
                            QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QTextCharFormat, QFont, QTextCursor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc
import re
import time
import json
from threading import Thread

def clear_gpu_memory():
    """Clear CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_dataset_from_hf(dataset_name):
    """
    Load dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset name
        
    Returns:
        DataFrame with all required data
    """
    # Load dataset
    from maknaz import pull
    try:
        dataset = pull(dataset_name)
        if 'train' in dataset:
            df = pd.DataFrame(dataset['train'])
        else:
            # Use first split if train isn't available
            first_split = list(dataset.keys())[0]
            df = pd.DataFrame(dataset[first_split])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
        
    return df

class DirectAnswerThread(QThread):
    progress_signal = pyqtSignal(int, str)
    streaming_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    
    def __init__(self, model, tokenizer, question, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.question = question
        self.device = device
        
    def run(self):
        try:
            self.progress_signal.emit(10, "Preparing direct answer query...")
            
            # Format for direct answer - but don't use chat template
            system_content = "You are a helpful Arabic assistant that answers questions accurately based on your knowledge."
            user_content = self.question
            
            # Use a simpler prompt format without the chat template markup
            formatted_text = f"{system_content}\n\nQuestion: {user_content}\n\nAnswer:"
            
            # Tokenize the prompt
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            self.progress_signal.emit(30, "Generating direct answer...")
            
            # Use non-streaming generation first, it's more reliable
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    repetition_penalty=1.1
                )
            
            # Get just the new tokens (the answer)
            input_length = len(inputs.input_ids[0])
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up any remaining instruction tags
            response = self.clean_response(response)
            
            # Send the full response (no streaming)
            self.streaming_signal.emit(response)
            
            self.progress_signal.emit(100, "Direct answer generation complete")
            self.finished_signal.emit(response)
            
        except Exception as e:
            print(f"Error in direct answer generation: {e}")
            import traceback
            traceback.print_exc()
            self.progress_signal.emit(100, f"Error: {str(e)}")
            self.finished_signal.emit(f"Error generating direct answer: {str(e)}")
            self.streaming_signal.emit(f"Error generating direct answer: {str(e)}")
    
    def clean_response(self, text):
        """Clean up any instruction tags or formatting in the response."""
        # Remove any [INST] or [/INST] tags
        text = re.sub(r'\[INST\]|\[\/INST\]', '', text)
        
        # Remove <<SYS>> and <</SYS>> blocks
        text = re.sub(r'<<SYS>>.*?<</SYS>>', '', text, flags=re.DOTALL)
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        return text

class RootRetrievalThread(QThread):
    progress_signal = pyqtSignal(int, str)
    thinking_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str, str)  # root, raw_response
    
    def __init__(self, model, tokenizer, question, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.question = question
        self.device = device
        
    def run(self):
        try:
            self.progress_signal.emit(10, "Preparing root extraction query...")
            
            # Format the question for root extraction - without chat template
            system_content = "You are a helpful Arabic linguistic analyzer. First analyze the word being asked about, identify its morphological patterns, and then determine its root. Output your thinking and then the root in a structured format."
            user_content = self.question
            
            # Use a simpler prompt format
            formatted_text = f"{system_content}\n\nQuestion: {user_content}\n\nAnalysis:"
            
            # Tokenize the prompt
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            self.progress_signal.emit(30, "Extracting root with fine-tuned model...")
            
            # Use non-streaming for more reliability
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    repetition_penalty=1.1
                )
            
            # Get just the new tokens (the answer)
            input_length = len(inputs.input_ids[0])
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up any remaining instruction tags
            response = self.clean_response(response)
            
            # Send the full response at once
            self.thinking_signal.emit(response)
            
            self.progress_signal.emit(60, "Extracting root from response...")
            
            # Extract the root from the response
            root = self.extract_root_from_response(response)
            
            self.progress_signal.emit(100, f"Root extraction complete: {root}")
            self.result_signal.emit(root, response)
            
        except Exception as e:
            print(f"Error in root extraction: {e}")
            import traceback
            traceback.print_exc()
            self.progress_signal.emit(100, f"Error: {str(e)}")
            self.result_signal.emit("error", f"Error extracting root: {str(e)}")
            self.thinking_signal.emit(f"Error extracting root: {str(e)}")
    
    def clean_response(self, text):
        """Clean up any instruction tags or formatting in the response."""
        # Remove any [INST] or [/INST] tags
        text = re.sub(r'\[INST\]|\[\/INST\]', '', text)
        
        # Remove <<SYS>> and <</SYS>> blocks
        text = re.sub(r'<<SYS>>.*?<</SYS>>', '', text, flags=re.DOTALL)
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_root_from_response(self, response):
        """Extract the root from the response using regex patterns."""
        # Normalize Arabic text
        response_norm = response
        
        # Try different patterns to extract the root, in order of preference
        root_patterns = [
            # Match "Root: XXX" pattern (both Arabic and English)
            r'Root[: ]+([""«»\'\'ـ\w]+)',
            r'جذر[: ]+([""«»\'\'ـ\w]+)',
            r'الجذر[: ]+([""«»\'\'ـ\w]+)',
            r'root[: ]+([""«»\'\'ـ\w]+)',
            r'جذر الكلمة[: ]+([""«»\'\'ـ\w]+)',
            
            # Try to find Arabic characters (assuming the root will be in Arabic)
            r'([ء-ي]{2,4})[.:\s]*$',  # 2-4 Arabic characters near the end of a line
            
            # More specific patterns for different formats
            r'(?:Root|جذر)[^:]*?:\s*[""«»\'\']*([ء-ي]{2,4})[""«»\'\']*',
            r'(?:الجذر|root)[^:]*?:\s*[""«»\'\']*([ء-ي]{2,4})[""«»\'\']*'
        ]
        
        for pattern in root_patterns:
            match = re.search(pattern, response_norm, re.IGNORECASE)
            if match and match.group(1):
                # Only return if it contains Arabic characters
                if re.search(r'[ء-ي]', match.group(1)):
                    return match.group(1).strip()
        
        # Try to extract from JSON-like structures
        try:
            json_match = re.search(r'{.*}', response_norm)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                if 'root' in data:
                    return data['root']
        except:
            pass
        
        # Try to extract from list-like structures
        list_match = re.search(r'\[(["\'"]?[\w\u0600-\u06FF، ]+["\'"]?(?:,\s*["\'"]?[\w\u0600-\u06FF، ]+["\'"]?)*)\]', response_norm)
        if list_match:
            items = re.findall(r'["\'"]?([\w\u0600-\u06FF، ]+)["\'"]?', list_match.group(1))
            if items:
                # Only return if it contains Arabic characters
                if re.search(r'[ء-ي]', items[0]):
                    return items[0].strip()
        
        # Find all Arabic words that might be roots (2-4 characters)
        all_arabic_words = re.findall(r'[ء-ي]{2,4}', response_norm)
        if all_arabic_words:
            # Try to find a word near "root" or "جذر"
            root_contexts = [
                r'(?:root|جذر|الجذر)[^ء-ي]{1,20}([ء-ي]{2,4})',
                r'([ء-ي]{2,4})[^ء-ي]{1,20}(?:root|جذر|الجذر)'
            ]
            
            for pattern in root_contexts:
                match = re.search(pattern, response_norm, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            # If we can't find by context, return the last Arabic word
            return all_arabic_words[-1]
        
        # Last resort: return "unknown"
        return "unknown"
class RetrievalAnswerThread(QThread):
    progress_signal = pyqtSignal(int, str)
    streaming_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    
    def __init__(self, model, tokenizer, question, root, context, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.question = question
        self.root = root
        self.context = context
        self.device = device
        
    def run(self):
        try:
            self.progress_signal.emit(10, "Preparing retrieval-based answer query...")
            
            # Format for retrieval-based answer - without using chat template
            system_content = "You are a helpful Arabic assistant that answers questions based on the provided context."
            user_content = f"Context about root '{self.root}':\n{self.context}\n\nQuestion: {self.question}"
            
            # Use a simpler prompt format
            formatted_text = f"{system_content}\n\n{user_content}\n\nAnswer:"
            
            # Tokenize the prompt
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            self.progress_signal.emit(30, "Generating retrieval-based answer...")
            
            # Use non-streaming generation for more reliability
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    repetition_penalty=1.1
                )
            
            # Get just the new tokens (the answer)
            input_length = len(inputs.input_ids[0])
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up any remaining instruction tags
            response = self.clean_response(response)
            
            # Send the full response at once
            self.streaming_signal.emit(response)
            
            self.progress_signal.emit(100, "Retrieval-based answer generation complete")
            self.finished_signal.emit(response)
            
        except Exception as e:
            print(f"Error in retrieval-based answer generation: {e}")
            import traceback
            traceback.print_exc()
            self.progress_signal.emit(100, f"Error: {str(e)}")
            self.finished_signal.emit(f"Error generating retrieval-based answer: {str(e)}")
            self.streaming_signal.emit(f"Error generating retrieval-based answer: {str(e)}")
    
    def clean_response(self, text):
        """Clean up any instruction tags or formatting in the response."""
        # Remove any [INST] or [/INST] tags
        text = re.sub(r'\[INST\]|\[\/INST\]', '', text)
        
        # Remove <<SYS>> and <</SYS>> blocks
        text = re.sub(r'<<SYS>>.*?<</SYS>>', '', text, flags=re.DOTALL)
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        return text

class TextStreamer:
    """Stream generation results as they're produced."""
    def __init__(self, tokenizer, callback):
        self.tokenizer = tokenizer
        self.callback = callback
        self.buffer = ""
        
    def put(self, token_ids):
        # More robust handling of different token_ids structures
        try:
            # Convert to list if it's a tensor
            if hasattr(token_ids, 'cpu'):
                token_ids = token_ids.cpu().tolist()
            
            # If it's a list of lists, take the first list
            if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            
            # Convert single integer to list if needed
            if isinstance(token_ids, int):
                token_ids = [token_ids]
                
            # Decode and update the buffer
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            if text and text != self.buffer:
                new_text = text[len(self.buffer):]
                self.buffer = text
                self.callback.emit(new_text)
                
        except Exception as e:
            # Log error but don't crash
            print(f"Error in TextStreamer.put: {e}")
            
            # Try to salvage the situation by emitting something
            try:
                if isinstance(token_ids, (list, tuple)) and token_ids:
                    # Try decoding each token individually and joining
                    pieces = []
                    for token_id in token_ids:
                        if isinstance(token_id, int):
                            pieces.append(self.tokenizer.decode([token_id], skip_special_tokens=True))
                    
                    if pieces:
                        joined_text = "".join(pieces)
                        self.callback.emit(f"{joined_text}")
            except:
                # Last resort: just notify that an error occurred
                self.callback.emit(" [Error decoding tokens] ")
    
    def end(self):
        # Final callback with any remaining text
        pass

class ModelLoaderThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, object, object, object, object, object)
    
    def __init__(self, base_model_path, adapter_path, dataset_name, base_device, adapter_device):
        super().__init__()
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.dataset_name = dataset_name
        self.base_device = base_device
        self.adapter_device = adapter_device
        
    def run(self):
        try:
            # Clean memory
            clear_gpu_memory()
            
            # Load tokenizer
            self.progress_signal.emit(10, "Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model with proper BitsAndBytesConfig
            self.progress_signal.emit(20, "Loading base model...")
            # quantization_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     llm_int8_skip_modules=None,
            #     llm_int8_threshold=6.0
            # )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                #quantization_config=quantization_config,
                device_map=self.base_device,
                torch_dtype=torch.float16 
            )
            
            # Load adapter model
            self.progress_signal.emit(40, "Loading GRPO adapter model...")
            adapter_model = None
            
            if self.adapter_path and os.path.exists(self.adapter_path):
                # Load the adapter config
                adapter_config = PeftConfig.from_pretrained(self.adapter_path)
                
                # Load the base model for adapter with proper BitsAndBytesConfig
                adapter_base = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    #quantization_config=quantization_config,
                    device_map=self.adapter_device,
                    torch_dtype=torch.float16
                )
                
                # Load the adapter onto the model
                adapter_model = PeftModel.from_pretrained(adapter_base, self.adapter_path)
            else:
                self.progress_signal.emit(50, "No adapter path provided or path doesn't exist. Using base model only.")
                adapter_model = base_model  # Fallback to base model if no adapter
            
            # Load dataset
            self.progress_signal.emit(70, "Loading dataset...")
            dataset = load_dataset_from_hf(self.dataset_name)
            
            self.progress_signal.emit(100, "Loading complete!")
            self.finished_signal.emit(True, base_model, adapter_model, tokenizer, tokenizer, dataset)
            
        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            self.progress_signal.emit(100, f"Error: {str(e)}")
            self.finished_signal.emit(False, None, None, None, None, None)

class ArabicRootComparatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arabic Root Analysis Comparator")
        self.setGeometry(100, 100, 1500, 900)
        
        # Model attributes
        self.base_model = None
        self.adapter_model = None
        self.tokenizer = None
        self.dataset = None
        
        # Use GPU if available
        if torch.cuda.device_count() >= 2:
            self.base_device = "cuda:0"
            self.adapter_device = "cuda:1"
        elif torch.cuda.is_available():
            self.base_device = "cuda:0"
            self.adapter_device = "cuda:0"
        else:
            self.base_device = "cpu"
            self.adapter_device = "cpu"
        
        # Initialize the UI
        self.init_ui()

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create a vertical layout for the top section (first 3 rows + GPU info)
        top_section_layout = QVBoxLayout()
        top_section_layout.setSpacing(10)
        
        # 1. Base model selection row
        base_row = QWidget()
        base_layout = QHBoxLayout(base_row)
        base_layout.setContentsMargins(0, 0, 0, 0)
        
        base_label = QLabel("Base Model:")
        base_label.setFixedWidth(100)
        self.base_model_combo = QComboBox()
        self.base_model_combo.setEditable(True)
        self.base_model_combo.addItem("/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/ALLaM-7B-Instruct-preview")
        self.base_model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        base_browse_btn = QPushButton("Browse...")
        base_browse_btn.setFixedWidth(100)
        base_browse_btn.clicked.connect(lambda: self.browse_model("base"))
        
        base_layout.addWidget(base_label)
        base_layout.addWidget(self.base_model_combo, 1)  # Give stretch factor for width
        base_layout.addWidget(base_browse_btn)
        base_layout.addStretch(1)  # Add stretch after inputs to limit width to half
        
        # 2. Adapter model selection row
        adapter_row = QWidget()
        adapter_layout = QHBoxLayout(adapter_row)
        adapter_layout.setContentsMargins(0, 0, 0, 0)
        
        adapter_label = QLabel("Adapter:")
        adapter_label.setFixedWidth(100)
        self.adapter_combo = QComboBox()
        self.adapter_combo.setEditable(True)
        self.adapter_combo.addItem("/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/mysam/Allam-7B-GRPO/checkpoint-762/")
        self.adapter_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        adapter_browse_btn = QPushButton("Browse..")
        #adapter_browse_btn.setFixedWidth(50)
        adapter_browse_btn.clicked.connect(lambda: self.browse_model("adapter"))
        
        adapter_layout.addWidget(adapter_label)
        adapter_layout.addWidget(self.adapter_combo, 1)  # Give stretch factor for width
        adapter_layout.addWidget(adapter_browse_btn)
        adapter_layout.addStretch(1)  # Add stretch after inputs to limit width to half
        
        # 3. Dataset selection row
        dataset_row = QWidget()
        dataset_layout = QHBoxLayout(dataset_row)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        
        dataset_label = QLabel("Dataset:")
        dataset_label.setFixedWidth(100)
        self.dataset_combo = QComboBox()
        self.dataset_combo.setEditable(True)
        self.dataset_combo.addItem("mysam/soal_w_jathr")
        self.dataset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.setFixedWidth(100)
        self.load_btn.clicked.connect(self.load_models_and_dataset)
        # self.load_btn.setStyleSheet("""
        #     QPushButton {
        #         background-color: #4CAF50;
        #         color: white;
        #         font-weight: bold;
        #         border-radius: 4px;
        #     }
        # """)
        
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.dataset_combo, 1)  # Give stretch factor for width
        dataset_layout.addWidget(self.load_btn)
        dataset_layout.addStretch(1)  # Add stretch after inputs to limit width to half
        
        # 4. GPU info row - full width
        gpu_row = QWidget()
        gpu_layout = QHBoxLayout(gpu_row)
        gpu_layout.setContentsMargins(0, 0, 0, 0)
        
        device_label = QLabel("GPU:")
        device_label.setFixedWidth(40)
        self.device_info_label = QLabel("")
        
        if torch.cuda.device_count() >= 2:
            self.device_info_label.setText("Using GPU 0 for base model, GPU 1 for adapter")
        elif torch.cuda.is_available():
            self.device_info_label.setText("Using GPU 0 for both models")
        else:
            self.device_info_label.setText("Using CPU for both models")
        
        gpu_layout.addWidget(device_label)
        gpu_layout.addWidget(self.device_info_label)
        gpu_layout.addStretch(1)  # Push content to the left
        
        # Add all rows to the top section
        top_section_layout.addWidget(base_row)
        top_section_layout.addWidget(adapter_row)
        top_section_layout.addWidget(dataset_row)
        top_section_layout.addWidget(gpu_row)
        
        # Add top section to main layout
        main_layout.addLayout(top_section_layout)
        
        # Add a thin separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # Create splitter for main content area
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)  # Give stretch
        
        # Left panel: Questions list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Questions list
        questions_label = QLabel("Questions:")
        self.questions_list = QListWidget()
        self.questions_list.currentItemChanged.connect(self.question_selected)
        
        # Analysis button
        self.analyze_btn = QPushButton("Compare Models")
        self.analyze_btn.clicked.connect(self.start_comparison)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        # Add to left layout
        left_layout.addWidget(questions_label)
        left_layout.addWidget(self.questions_list, 1)
        left_layout.addWidget(self.analyze_btn)
        
        # Right panel: Results display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Question display in its own row
        question_group = QGroupBox("Question")
        question_layout = QVBoxLayout(question_group)
        question_layout.setContentsMargins(5, 5, 5, 5)
        self.question_text = QTextEdit()
        self.question_text.setReadOnly(True)
        self.question_text.setMaximumHeight(60)
        self.question_text.setStyleSheet("""
            direction: rtl;
            text-align: right;
            font-size: 12pt;
            padding: 5px;
            background-color: white;
            border: 1px solid #ccc;
        """)
        question_layout.addWidget(self.question_text)
        
        # Root extraction display
        root_group = QGroupBox("Root Extraction (GRPO Model)")
        root_layout = QVBoxLayout(root_group)
        root_layout.setContentsMargins(5, 5, 5, 5)
        self.root_text = QTextEdit()
        self.root_text.setReadOnly(True)
        self.root_text.setStyleSheet("""
            direction: rtl;
            text-align: right;
            font-size: 12pt;
            padding: 10px;
            background-color: #f0f8ff;
        """)
        root_layout.addWidget(self.root_text)
        
        # Results in a grid layout
        results_group = QGroupBox("Results Comparison")
        results_layout = QGridLayout(results_group)
        results_layout.setContentsMargins(5, 5, 5, 5)
        
        # Direct answer (left column)
        direct_label = QLabel("Direct Answer (Base Model Only)")
        self.direct_answer_text = QTextEdit()
        self.direct_answer_text.setReadOnly(True)
        self.direct_answer_text.setMinimumHeight(200)
        self.direct_answer_text.setStyleSheet("""
            direction: rtl;
            text-align: right;
            font-size: 12pt;
            padding: 10px;
            background-color: #fff8f0;
        """)
        
        # Retrieval answer (right column)
        retrieval_label = QLabel("Retrieval Answer (Root → Content → Base Model)")
        self.retrieval_answer_text = QTextEdit()
        self.retrieval_answer_text.setReadOnly(True)
        self.retrieval_answer_text.setMinimumHeight(200)
        self.retrieval_answer_text.setStyleSheet("""
            direction: rtl;
            text-align: right;
            font-size: 12pt;
            padding: 10px;
            background-color: #f0fff8;
        """)
        
        # Add to results layout
        results_layout.addWidget(direct_label, 0, 0)
        results_layout.addWidget(retrieval_label, 0, 1)
        results_layout.addWidget(self.direct_answer_text, 1, 0)
        results_layout.addWidget(self.retrieval_answer_text, 1, 1)
        
        # Set equal column widths
        results_layout.setColumnStretch(0, 1)
        results_layout.setColumnStretch(1, 1)
        
        # Context display
        context_group = QGroupBox("Retrieved Context")
        context_layout = QVBoxLayout(context_group)
        context_layout.setContentsMargins(5, 5, 5, 5)
        self.context_text = QTextEdit()
        self.context_text.setReadOnly(True)
        self.context_text.setStyleSheet("""
            direction: rtl;
            text-align: right;
            font-size: 12pt;
            padding: 10px;
            background-color: #f8f8f8;
        """)
        context_layout.addWidget(self.context_text)
        
        # Add all widgets to right layout with stretch factors
        right_layout.addWidget(question_group)     # Question on top row
        right_layout.addWidget(root_group, 1)      # Root extraction
        right_layout.addWidget(results_group, 3)   # Results comparison (most space)
        right_layout.addWidget(context_group, 2)   # Context
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1100])
        
        # Status bar and progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.statusBar().addPermanentWidget(self.progress_bar, 1)
        self.statusBar().showMessage("Ready")

    def browse_model(self, model_type="base"):
        """Browse for model directory."""
        folder = QFileDialog.getExistingDirectory(self, f"Select {model_type.capitalize()} Model Directory")
        if folder:
            if model_type == "base":
                self.base_model_combo.setCurrentText(folder)
            else:
                self.adapter_combo.setCurrentText(folder)
    
    def load_models_and_dataset(self):
        """Load both models and the dataset."""
        base_model_path = self.base_model_combo.currentText()
        adapter_path = self.adapter_combo.currentText()
        dataset_name = self.dataset_combo.currentText()
        
        # Validate paths
        if not os.path.exists(base_model_path):
            QMessageBox.warning(self, "Error", f"Base model path does not exist: {base_model_path}")
            return
        
        # Start loading thread
        self.load_thread = ModelLoaderThread(
            base_model_path,
            adapter_path,
            dataset_name,
            self.base_device,
            self.adapter_device
        )
        self.load_thread.progress_signal.connect(self.update_progress)
        self.load_thread.finished_signal.connect(self.loading_finished)
        
        # Disable UI during loading
        self.load_btn.setEnabled(False)
        self.statusBar().showMessage("Loading models and dataset...")
        self.progress_bar.setValue(0)
        
        # Start thread
        self.load_thread.start()
    
    def update_progress(self, progress, message):
        """Update progress bar and status message."""
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(message)
    
    def loading_finished(self, success, base_model, adapter_model, base_tokenizer, adapter_tokenizer, dataset):
        """Handle the completion of model and dataset loading."""
        if success:
            self.base_model = base_model
            self.adapter_model = adapter_model
            self.tokenizer = base_tokenizer  # Using same tokenizer for both
            self.dataset = dataset
            
            # Populate questions list
            self.questions_list.clear()
            if 'question' in dataset.columns:
                # Take first 100 questions to avoid overwhelming the UI
                questions = dataset['question'].unique().tolist()[:100]
                self.questions_list.addItems(questions)
                
                self.analyze_btn.setEnabled(True)
                
                self.statusBar().showMessage(
                    f"Models and dataset loaded successfully. {len(questions)} questions available."
                )
            else:
                QMessageBox.warning(self, "Error", "Dataset missing 'question' column")
                self.statusBar().showMessage("Error: Dataset missing required columns")
        else:
            QMessageBox.warning(self, "Error", "Failed to load models or dataset")
            self.statusBar().showMessage("Error loading models or dataset")
        
        # Re-enable UI
        self.load_btn.setEnabled(True)
    
    def question_selected(self, current, previous):
        """Update the UI when a question is selected."""
        if current:
            question_text = current.text()
            self.question_text.setText(question_text)
            
            # Clear previous results
            self.root_text.clear()
            self.direct_answer_text.clear()
            self.retrieval_answer_text.clear()
            self.context_text.clear()
            
            self.statusBar().showMessage(f"Selected question: '{question_text[:50]}...'")
    
    def start_comparison(self):
        """Start the comparison between direct and retrieval-based methods."""
        if self.questions_list.currentItem() is None:
            QMessageBox.warning(self, "Error", "Please select a question first")
            return
        
        if self.base_model is None or self.adapter_model is None:
            QMessageBox.warning(self, "Error", "Please load models first")
            return
        
        # Get the selected question
        question = self.questions_list.currentItem().text()
        
        # Clear previous results
        self.root_text.clear()
        self.direct_answer_text.clear()
        self.retrieval_answer_text.clear()
        self.context_text.clear()
        
        # Disable UI during processing
        self.analyze_btn.setEnabled(False)
        self.questions_list.setEnabled(False)
        
        # Start root extraction and direct answer threads in parallel
        # 1. Start root extraction thread with adapter model
        self.root_thread = RootRetrievalThread(
            self.adapter_model,
            self.tokenizer,
            question,
            self.adapter_device
        )
        self.root_thread.progress_signal.connect(self.update_progress)
        self.root_thread.thinking_signal.connect(self.update_root_thinking)
        self.root_thread.result_signal.connect(self.root_extraction_finished)
        
        # 2. Start direct answer thread with base model
        self.direct_thread = DirectAnswerThread(
            self.base_model,
            self.tokenizer,
            question,
            self.base_device
        )
        self.direct_thread.progress_signal.connect(lambda p, m: self.update_progress(p//2, f"Direct: {m}"))
        self.direct_thread.streaming_signal.connect(self.update_direct_answer)
        self.direct_thread.finished_signal.connect(self.direct_answer_finished)
        
        # Start both threads
        self.statusBar().showMessage("Starting parallel analysis...")
        self.progress_bar.setValue(0)
        self.root_thread.start()
        self.direct_thread.start()
    
    def update_root_thinking(self, text):
        """Update the root extraction thinking display with streaming text."""
        cursor = self.root_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.root_text.setTextCursor(cursor)
        self.root_text.ensureCursorVisible()
    
    def update_direct_answer(self, text):
        """Update the direct answer display with streaming text."""
        cursor = self.direct_answer_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.direct_answer_text.setTextCursor(cursor)
        self.direct_answer_text.ensureCursorVisible()
    
    def update_retrieval_answer(self, text):
        """Update the retrieval answer display with streaming text."""
        cursor = self.retrieval_answer_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.retrieval_answer_text.setTextCursor(cursor)
        self.retrieval_answer_text.ensureCursorVisible()
    
    def root_extraction_finished(self, root, raw_response):
        """Handle completion of root extraction and start retrieval-based answer generation."""
        # Show the extracted root and highlight it
        highlight_text = f"\n\nExtracted root: {root}"
        self.update_root_thinking(highlight_text)
        
        # Find context for this root in the dataset
        self.retrieve_context_for_root(root)
    
    def retrieve_context_for_root(self, root):
        """Find the context for the given root in the dataset."""
        try:
            # Look for the root in the dataset
            if 'root' in self.dataset.columns and 'context' in self.dataset.columns:
                root_entries = self.dataset[self.dataset['root'] == root]
                
                if not root_entries.empty:
                    # Found matching root
                    entry = root_entries.iloc[0]
                    context = entry['context']
                    
                    # Update the context display
                    self.context_text.setText(f"Context for root '{root}':\n\n{context}")
                    
                    # Start the retrieval-based answer generation
                    question = self.question_text.toPlainText()
                    
                    self.retrieval_thread = RetrievalAnswerThread(
                        self.base_model,
                        self.tokenizer,
                        question,
                        root,
                        context,
                        self.base_device
                    )
                    self.retrieval_thread.progress_signal.connect(lambda p, m: self.update_progress(50 + p//2, f"Retrieval: {m}"))
                    self.retrieval_thread.streaming_signal.connect(self.update_retrieval_answer)
                    self.retrieval_thread.finished_signal.connect(self.retrieval_answer_finished)
                    
                    # Start the thread
                    self.retrieval_thread.start()
                else:
                    # No matching root found
                    self.context_text.setText(f"No context found for root '{root}' in the dataset.")
                    self.retrieval_answer_text.setText("Cannot generate retrieval-based answer without context.")
                    
                    # Re-enable UI
                    self.analyze_btn.setEnabled(True)
                    self.questions_list.setEnabled(True)
                    self.statusBar().showMessage("Retrieval-based answer failed: No context found for root.")
            else:
                # Missing required columns
                self.context_text.setText("Dataset missing required columns (root, context).")
                self.retrieval_answer_text.setText("Cannot generate retrieval-based answer: Dataset missing required columns.")
                
                # Re-enable UI
                self.analyze_btn.setEnabled(True)
                self.questions_list.setEnabled(True)
                self.statusBar().showMessage("Retrieval-based answer failed: Dataset missing required columns.")
                
        except Exception as e:
            print(f"Error retrieving context: {e}")
            self.context_text.setText(f"Error retrieving context: {str(e)}")
            self.statusBar().showMessage(f"Error retrieving context: {str(e)}")
            
            # Re-enable UI
            self.analyze_btn.setEnabled(True)
            self.questions_list.setEnabled(True)
    
    def direct_answer_finished(self, full_response):
        """Handle completion of direct answer generation."""
        self.statusBar().showMessage("Direct answer generation complete")
    
    def retrieval_answer_finished(self, full_response):
        """Handle completion of retrieval-based answer generation."""
        self.statusBar().showMessage("Both answer methods complete")
        
        # Re-enable UI
        self.analyze_btn.setEnabled(True)
        self.questions_list.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = ArabicRootComparatorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()