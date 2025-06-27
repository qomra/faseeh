import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

class ArabicDictionaryRetriever:
    def __init__(self, 
                 model_name="/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/ALLaM-7B-Instruct-preview",
                 device="cuda",
                 max_cache_tokens=2048,
                 answer_tokens=500,
                 phrase_token_num=3,
                 top_k=5,
                 quantization="4bit",
                 max_seq_len=None):
        """
        Initialize the Arabic Dictionary Retriever system.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on ("cuda" or "cpu")
            max_cache_tokens: Maximum tokens to keep in cache
            answer_tokens: Tokens reserved for answer generation
            phrase_token_num: Size of phrase to consider when calculating attention
            top_k: Number of top contents to keep in cache
            quantization: Quantization method to use ("none", "4bit", "8bit", "fp16")
            max_seq_len: Maximum sequence length for the model (if None, will use model's max_position_embeddings)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} on {self.device} with {quantization} quantization...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure quantization
        quantization_config = None
        torch_dtype = None
        
        if quantization == "4bit":
            # 4-bit quantization using bitsandbytes
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            # 8-bit quantization using bitsandbytes
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        elif quantization == "fp16":
            # Half precision (16-bit)
            torch_dtype = torch.float16
        else:
            # No quantization, use model's default precision
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load the model with appropriate quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_attentions=True,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            attn_implementation="eager"
        ).to(self.device)
        
        # Detect model type based on name
        self.model_type = self._detect_model_type(model_name)
        print(f"Detected model type: {self.model_type}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Configuration parameters
        self.max_cache_tokens = max_cache_tokens
        self.answer_tokens = answer_tokens
        self.phrase_token_num = phrase_token_num
        self.top_k = top_k
        
        # Set max sequence length - use provided value or model's default
        self.max_seq_len = max_seq_len if max_seq_len is not None else self.model.config.max_position_embeddings
        print(f"Using max sequence length: {self.max_seq_len}")
        
        # Cache for relevant content
        self.cache_contents = []
        self.cache_token_count = 0
        
        # Check if the tokenizer has padding token, if not, set it
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _detect_model_type(self, model_name):
        """
        Detect the model type based on the model name.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            String indicating model type: "llama2", "llama3", or "other"
        """
        model_name_lower = model_name.lower()
        
        if "llama-2" in model_name_lower or "llama2" in model_name_lower:
            return "llama2"
        elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
            return "llama3"
        else:
            # Check if the tokenizer has a chat template
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
                return "custom_template"
            else:
                return "other"
                
    def _format_with_chat_template(self, question, content):
        """
        Format the inputs using the appropriate chat template.
        
        Args:
            question: The question to ask
            content: The content to analyze
            
        Returns:
            Formatted text using the model's chat template
        """
        # Create messages for the chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes texts to answer questions."},
            {"role": "user", "content": f"I need to find information relevant to this question: {question}\n\nHere is the content to analyze:\n{content}"}
        ]
        
        if self.model_type == "llama2":
            # Llama 2 template (manually constructed)
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            formatted = f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]"
            
        elif self.model_type == "llama3":
            # Llama 3 template (manually constructed)
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            formatted = f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            
        elif self.model_type == "custom_template" and hasattr(self.tokenizer, "apply_chat_template"):
            # Use model's built-in chat template
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
        else:
            # Fallback to a simple format
            formatted = f"Question: {question}\n\nContent: {content}"
            
        return formatted
            
    def load_dictionary_data(self, dictionary_df):
        """
        Load the Arabic dictionary data.
        
        Args:
            dictionary_df: DataFrame with columns 'root' and 'content'
        """
        # Ensure required columns exist
        if 'root' not in dictionary_df.columns or 'content' not in dictionary_df.columns:
            missing = []
            if 'root' not in dictionary_df.columns: missing.append('root')
            if 'content' not in dictionary_df.columns: missing.append('content')
            raise ValueError(f"Dictionary DataFrame missing required columns: {', '.join(missing)}")
        
        # Filter out rows with empty content
        valid_rows = dictionary_df[dictionary_df['content'].notna() & (dictionary_df['content'] != '')]
        
        if len(valid_rows) < len(dictionary_df):
            print(f"Filtered out {len(dictionary_df) - len(valid_rows)} entries with empty content")
            
        self.dictionary_df = valid_rows.reset_index(drop=True)
        print(f"Loaded dictionary with {len(self.dictionary_df)} root entries")
        
    def calculate_attention_scores(self, query, content):
        """
        Calculate attention scores between query and content to identify important parts.
        
        Args:
            query: The query/question text
            content: The dictionary content to analyze
            
        Returns:
            Tuple of (content_tokens, attention_scores)
        """
        # Format with appropriate chat template
        formatted_text = self._format_with_chat_template(query, content)
        
        # Get token positions for query and content
        query_marker = f"question: {query}" if self.model_type == "other" else query
        content_marker = f"content: {content}" if self.model_type == "other" else content
        
        # Tokenize the formatted text
        formatted_tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        query_tokens = self.tokenizer.encode(query_marker, add_special_tokens=False)
        content_tokens = self.tokenizer.encode(content_marker, add_special_tokens=False)
        
        # Find approximate positions of query and content in formatted text
        # This is a heuristic approach and might need adjustment for specific models
        formatted_text_str = ' '.join(map(str, formatted_tokens))
        query_tokens_str = ' '.join(map(str, query_tokens))
        content_tokens_str = ' '.join(map(str, content_tokens))
        
        # Use a sliding window to find the best match position for query and content
        query_pos = -1
        for i in range(len(formatted_tokens) - len(query_tokens) + 1):
            window = ' '.join(map(str, formatted_tokens[i:i+len(query_tokens)]))
            if self._sequence_similarity(window, query_tokens_str) > 0.7:  # 70% similarity threshold
                query_pos = i
                break
                
        content_pos = -1
        for i in range(len(formatted_tokens) - len(content_tokens) + 1):
            window = ' '.join(map(str, formatted_tokens[i:i+len(content_tokens)]))
            if self._sequence_similarity(window, content_tokens_str) > 0.7:  # 70% similarity threshold
                content_pos = i
                break
        
        # If we couldn't find the positions, fall back to a simpler approach
        if query_pos == -1 or content_pos == -1:
            return self._fallback_attention_calculation(query, content)
        
        # Tokenize the formatted text for the model
        inputs = self.tokenizer(formatted_text, return_tensors="pt", 
                               truncation=True, max_length=self.max_seq_len).to(self.device)
        
        # Get the attention scores from the model
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention from last layer - most semantic information
        attentions = outputs.attentions[-1]  # Shape: [batch, num_heads, seq_len, seq_len]
        
        # Average across attention heads to get overall attention pattern
        avg_attentions = torch.mean(attentions, dim=1)[0]  # Shape: [seq_len, seq_len]
        
        # Define query and content token positions
        query_indices = list(range(query_pos, query_pos + len(query_tokens)))
        content_indices = list(range(content_pos, content_pos + len(content_tokens)))
        
        # Adjust indices if they're out of bounds
        query_indices = [i for i in query_indices if i < avg_attentions.shape[0]]
        content_indices = [i for i in content_indices if i < avg_attentions.shape[1]]
        
        if not query_indices or not content_indices:
            return self._fallback_attention_calculation(query, content)
        
        # Extract attention from query to content
        query_to_content_attention = avg_attentions[query_indices, :][:, content_indices]
        
        # Get content tokens
        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        content_tokens = [all_tokens[i] for i in content_indices if i < len(all_tokens)]
        
        return content_tokens, query_to_content_attention.cpu().numpy()
        
    def _sequence_similarity(self, seq1, seq2):
        """
        Calculate similarity between two token sequences (as strings).
        
        Args:
            seq1: First sequence as string
            seq2: Second sequence as string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity for token overlap
        set1 = set(seq1.split())
        set2 = set(seq2.split())
        
        if not set1 or not set2:
            return 0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
        
    def _fallback_attention_calculation(self, query, content):
        """
        Fallback method to calculate attention when chat template processing fails.
        
        Args:
            query: The query/question text
            content: The dictionary content to analyze
            
        Returns:
            Tuple of (content_tokens, attention_scores)
        """
        # Combine query and content with separator
        combined_text = query + " [SEP] " + content
        
        # Tokenize the combined text
        inputs = self.tokenizer(combined_text, return_tensors="pt", 
                               truncation=True, max_length=self.max_seq_len).to(self.device)
        
        # Get the attention scores from the model
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention from last layer
        attentions = outputs.attentions[-1]
        avg_attentions = torch.mean(attentions, dim=1)[0]
        
        # Split into query and content sections
        query_tokens = self.tokenizer.encode(query, add_special_tokens=True)
        sep_tokens = self.tokenizer.encode(" [SEP] ", add_special_tokens=False)
        query_len = len(query_tokens) - 1  # Adjust for special tokens
        
        # Get indices for query and content
        query_indices = list(range(query_len))
        content_indices = list(range(query_len + len(sep_tokens), inputs.input_ids.shape[1]))
        
        # Extract attention from query to content
        query_to_content_attention = avg_attentions[query_indices, :][:, content_indices]
        
        # Get content tokens
        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        content_tokens = [all_tokens[i] for i in content_indices]
        
        return content_tokens, query_to_content_attention.cpu().numpy()

    def get_importance_scores(self, attention_matrix):
        """
        Calculate importance scores for each token using 1D convolution approach.
        
        Args:
            attention_matrix: Matrix of attention scores from query to content
            
        Returns:
            Array of importance scores for each token
        """
        # Create convolutional kernel for phrase-level attention
        kernel_size = min(self.phrase_token_num, attention_matrix.shape[1])
        kernel = np.ones(kernel_size)
        
        # Apply convolution to get phrase-level importance
        importance_scores = np.zeros(attention_matrix.shape[1])
        
        # For each position, sum the attention from all query tokens
        for i in range(attention_matrix.shape[0]):  # For each query token
            # Apply 1D convolution
            conv_result = np.convolve(attention_matrix[i], kernel, mode='same')
            importance_scores += conv_result
            
        return importance_scores
        
    def retrieve_relevant_content(self, question, root_content):
        """
        Retrieve relevant content for a question from a given root_content.
        
        Args:
            question: The question text
            root_content: The dictionary content to analyze
            
        Returns:
            Tuple of (importance_score, root_content)
        """
        # Get content tokens and attention scores
        content_tokens, attention_scores = self.calculate_attention_scores(question, root_content)
        
        # Calculate importance scores
        importance_scores = self.get_importance_scores(attention_scores)
        
        # Calculate overall importance score for this content
        overall_importance = np.mean(importance_scores)
        
        return overall_importance, root_content

    def chunk_content(self, content):
        """
        Divide content into chunks that fit within the model's sequence length.
        
        Args:
            content: The text content to chunk
            
        Returns:
            List of content chunks
        """
        # Calculate effective max length (accounting for answer tokens and some padding)
        effective_max_len = self.max_seq_len - self.answer_tokens - 500  # 500 tokens buffer for prompt and other elements
        
        # Tokenize the content
        tokens = self.tokenizer.encode(content)
        
        # If content already fits, return as is
        if len(tokens) <= effective_max_len:
            return [content]
        
        # Create chunks based on token lengths
        chunks = []
        current_chunk_tokens = []
        current_length = 0
        
        for token in tokens:
            if current_length + 1 > effective_max_len:
                # Current chunk is full, decode it and add to chunks
                chunk_text = self.tokenizer.decode(current_chunk_tokens)
                chunks.append(chunk_text)
                
                # Reset for next chunk
                current_chunk_tokens = [token]
                current_length = 1
            else:
                # Add token to current chunk
                current_chunk_tokens.append(token)
                current_length += 1
        
        # Add final chunk if there's anything left
        if current_chunk_tokens:
            chunk_text = self.tokenizer.decode(current_chunk_tokens)
            chunks.append(chunk_text)
        
        # Log chunking info
        print(f"Split content into {len(chunks)} chunks to fit max sequence length")
        return chunks
        
    def update_cache(self, new_content_tuple):
        """
        Update cache with new relevant content.
        
        Args:
            new_content_tuple: Tuple of (importance_score, content)
        """
        # Add new content to cache
        self.cache_contents.append(new_content_tuple)
        
        # Sort cache by importance score (descending)
        self.cache_contents.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only top_k items
        self.cache_contents = self.cache_contents[:self.top_k]
        
        # Recalculate token count
        self.cache_token_count = sum(len(self.tokenizer.encode(content)) 
                                     for _, content in self.cache_contents)
        
        # If cache exceeds max token limit, remove least important entries
        while self.cache_token_count > self.max_cache_tokens and len(self.cache_contents) > 1:
            removed = self.cache_contents.pop()
            self.cache_token_count -= len(self.tokenizer.encode(removed[1]))
    
    def process_dictionary(self, question, max_entries=None, progress_callback=None):
        """
        Process the dictionary to find relevant content for the question.
        
        Args:
            question: The question to answer
            max_entries: Maximum number of dictionary entries to process (None for all)
            progress_callback: Optional callback for progress updates
        """
        # Reset cache
        self.cache_contents = []
        self.cache_token_count = 0
        
        # Choose subset of dictionary if max_entries specified
        if max_entries is not None and max_entries < len(self.dictionary_df):
            process_df = self.dictionary_df.sample(max_entries, random_state=42)
            print(f"Processing random sample of {max_entries} dictionary entries")
        else:
            process_df = self.dictionary_df
            
        # Process each root and its content
        for idx, row in tqdm(process_df.iterrows(), total=len(process_df), desc="Processing dictionary"):
            root = row['root']
            content = row['content']
            
            # Skip if content is not a string
            if not isinstance(content, str) or not content.strip():
                continue
                
            # Update progress if callback provided
            if progress_callback:
                progress_callback((idx + 1) / len(process_df), 
                                  f"Processing root {idx+1}/{len(process_df)}: {root}")
            
            try:
                # Check if content needs to be chunked
                content_token_length = len(self.tokenizer.encode(content))
                effective_max_len = self.max_seq_len - self.answer_tokens - 500
                
                if content_token_length > effective_max_len:
                    # Content is too large, split it into chunks
                    chunks = self.chunk_content(content)
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            # Calculate relevance of this chunk
                            importance_score, chunk_text = self.retrieve_relevant_content(question, chunk)
                            
                            # Add chunk identifier for tracking
                            labeled_chunk = f"[ROOT: {root}, CHUNK: {i+1}/{len(chunks)}] {chunk_text}"
                            
                            # Update cache with this chunk
                            self.update_cache((importance_score, labeled_chunk))
                        except Exception as e:
                            print(f"Error processing chunk {i+1} of entry {root}: {e}")
                            continue
                else:
                    # Content fits within model's capacity, process it normally
                    importance_score, content_text = self.retrieve_relevant_content(question, content)
                    
                    # Update cache with this content
                    labeled_content = f"[ROOT: {root}] {content_text}"
                    self.update_cache((importance_score, labeled_content))
                
            except Exception as e:
                print(f"Error processing entry {root}: {e}")
                continue
    
    def answer_question(self, question, prompt_template=None):
        """
        Answer a question using the cached relevant contents.
        
        Args:
            question: The question to answer
            prompt_template: Custom prompt template (optional)
            
        Returns:
            The generated answer
        """
        if not self.cache_contents:
            return "No relevant content found to answer this question."
        
        # Extract contents from cache (ignoring scores)
        contents = [content for _, content in self.cache_contents]
        
        # Combine all cached content
        combined_content = "\n\n".join(contents)
        
        # Create messages for chat format
        messages = [
            {"role": "system", "content": "You are an expert in Arabic language and dictionary meanings. Answer questions accurately and concisely based on the dictionary content provided."},
            {"role": "user", "content": f"Based on this dictionary content:\n\n{combined_content}\n\nPlease answer this question: {question}"}
        ]
        
        # Use the appropriate formatting based on model type
        if self.model_type == "custom_template" and hasattr(self.tokenizer, "apply_chat_template"):
            # Use model's built-in chat template
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        elif self.model_type == "llama2":
            # Llama 2 template
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            prompt = f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]"
        elif self.model_type == "llama3":
            # Llama 3 template
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif prompt_template:
            # Use custom prompt template if provided
            prompt = prompt_template.format(content=combined_content, question=question)
        else:
            # Default prompt format
            prompt = f"""Given the following dictionary content, please answer the question about Arabic word meaning:

Dictionary Content:
{combined_content}

Question: {question}

Answer:"""

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len).to(self.device)
        
        # Generate answer
        max_input_length = len(inputs.input_ids[0])
        max_new_tokens = min(self.answer_tokens, self.max_seq_len - max_input_length)
        
        with torch.no_grad():
            generated = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.7,
                num_beams=1,
            )
        
        # Decode the answer
        full_output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if self.model_type == "llama3":
            # For Llama 3, the answer starts after the assistant header
            answer_start = full_output.find("<|start_header_id|>assistant<|end_header_id|>")
            if answer_start != -1:
                answer_start += len("<|start_header_id|>assistant<|end_header_id|>")
                answer_end = full_output.find("<|eot_id|>", answer_start)
                if answer_end != -1:
                    answer = full_output[answer_start:answer_end].strip()
                else:
                    answer = full_output[answer_start:].strip()
            else:
                answer = full_output[len(prompt):].strip()
        elif "Answer:" in full_output:
            # If the output contains "Answer:", extract the text after it
            answer_start = full_output.find("Answer:") + len("Answer:")
            answer = full_output[answer_start:].strip()
        elif len(prompt) < len(full_output):
            # If we can't find a clear marker, just return everything after the prompt
            answer = full_output[len(prompt):].strip()
        else:
            # Fallback - return the whole output
            answer = full_output.strip()
        
        return answer

def load_datasets_from_hf(dictionary_dataset, questions_dataset):
    """
    Load datasets from HuggingFace.
    
    Args:
        dictionary_dataset: HuggingFace dataset name for dictionary
        questions_dataset: HuggingFace dataset name for questions
        
    Returns:
        Tuple of (dictionary_df, questions_df)
    """
    # Load dictionary dataset
    from maknaz import pull
    try:
        #dictionary_data = load_dataset(dictionary_dataset)
        dictionary_data = pull(dictionary_dataset)
        if 'train' in dictionary_data:
            dictionary_df = pd.DataFrame(dictionary_data['train'])
        else:
            # Use first split if train isn't available
            first_split = list(dictionary_data.keys())[0]
            dictionary_df = pd.DataFrame(dictionary_data[first_split])
    except Exception as e:
        print(f"Error loading dictionary dataset: {e}")
        raise
        
    # Load questions dataset
    try:
        #questions_data = load_dataset(questions_dataset)
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
        
    # Verify required columns exist
    if 'root' not in dictionary_df.columns or 'content' not in dictionary_df.columns:
        missing = []
        if 'root' not in dictionary_df.columns: missing.append('root')
        if 'content' not in dictionary_df.columns: missing.append('content')
        raise ValueError(f"Dictionary dataset missing required columns: {', '.join(missing)}")
        
    if 'question' not in questions_df.columns or 'answer' not in questions_df.columns:
        missing = []
        if 'question' not in questions_df.columns: missing.append('question')
        if 'answer' not in questions_df.columns: missing.append('answer')
        raise ValueError(f"Questions dataset missing required columns: {', '.join(missing)}")
    
    return dictionary_df, questions_df

def evaluate_model(dictionary_dataset, questions_dataset, model_name="/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/ALLaM-7B-Instruct-preview", quantization="4bit", max_cache_tokens=3000, max_seq_len=None):
    """
    Evaluate the model on a set of questions.
    
    Args:
        dictionary_dataset: HuggingFace dataset name or path for dictionary with 'root' and 'content' columns
        questions_dataset: HuggingFace dataset name or path for questions with 'question' and 'answer' columns
        model_name: HuggingFace model name or path
        quantization: Quantization method to use ("none", "4bit", "8bit", "fp16")
        max_cache_tokens: Maximum tokens to keep in cache
        max_seq_len: Maximum sequence length for processing
    """
    # Load datasets from HuggingFace
    dictionary_df, questions_df = load_datasets_from_hf(dictionary_dataset, questions_dataset)
    
    # Initialize retriever with quantization
    retriever = ArabicDictionaryRetriever(
        model_name=model_name,
        max_cache_tokens=max_cache_tokens,
        top_k=5,
        quantization=quantization,
        max_seq_len=max_seq_len
    )
    
    # Load dictionary
    retriever.load_dictionary_data(dictionary_df)
    
    # Evaluate each question
    results = []
    
    # Create a sample if needed
    if len(questions_df) > 50:  # Limit to 50 questions for faster evaluation
        print(f"Sampling 50 questions from {len(questions_df)} for evaluation")
        eval_df = questions_df.sample(50, random_state=42)
    else:
        eval_df = questions_df
    
    prompt_template = """You are an expert in Arabic language and dictionary meanings.
Given the following dictionary content, please answer the question about Arabic word meaning:

Dictionary Content:
{content}

Question: {question}

Answer:"""
    
    # Process 100 dictionary entries at most to speed up evaluation
    max_dictionary_entries = len(dictionary_df)  # Changed from min(100, len(dictionary_df))
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating questions"):
        question = row['question']
        gold_answer = row['answer']
        
        try:
            # Process dictionary to find relevant content
            retriever.process_dictionary(question, max_entries=max_dictionary_entries)
            
            # Generate answer
            predicted_answer = retriever.answer_question(question, prompt_template=prompt_template)
            
            # Store results
            results.append({
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'cache_size': len(retriever.cache_contents)
            })
            # save the results
            results_df = pd.DataFrame(results)
            results_df.to_csv('evaluation_results.csv', index=False)
        except Exception as e:
            print(f"Error evaluating question '{question}': {e}")
            results.append({
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': f"ERROR: {str(e)}",
                'cache_size': 0
            })
        
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"Evaluation complete. {len(results_df)} questions evaluated.")
    return results_df

def main():
    """
    Main function to run the model.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Arabic Dictionary Meaning Retriever')
    parser.add_argument('--dictionary', default="mysam/lisan_alarab", help='HuggingFace dataset name for dictionary (must have root and content columns)')
    parser.add_argument('--questions', default="mysam/kalima_w_maana", help='HuggingFace dataset name for questions (must have question and answer columns)')
    parser.add_argument('--model', default="/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/ALLaM-7B-Instruct-preview", help='HuggingFace model name or path')
    parser.add_argument('--output', default="evaluation_results.csv", help='Path to save evaluation results')
    parser.add_argument('--sample_size', type=int, default=3, help='Number of questions to evaluate (None for all)')
    parser.add_argument('--quantization', type=str, default="4bit", choices=["none", "4bit", "8bit", "fp16"], 
                      help='Quantization method to use for the model')
    parser.add_argument('--max_cache_tokens', type=int, default=3000, 
                      help='Maximum tokens to keep in cache')
    parser.add_argument('--max_seq_len', type=int, default=None, 
                      help='Maximum sequence length for the model (if None, will use model default)')
    
    args = parser.parse_args()
    
    # Run evaluation with all specified parameters
    results = evaluate_model(
        args.dictionary, 
        args.questions, 
        args.model, 
        args.quantization,
        args.max_cache_tokens,
        args.max_seq_len
    )
    
    # Sample if requested
    if args.sample_size and args.sample_size < len(results):
        results_sample = results.sample(args.sample_size, random_state=42)
    else:
        results_sample = results
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"Full results saved to {args.output}")
    
    # Print sample results
    print("\nSample Results:")
    for i in range(min(5, len(results_sample))):
        print(f"Question: {results_sample.iloc[i]['question']}")
        print(f"Gold Answer: {results_sample.iloc[i]['gold_answer']}")
        print(f"Predicted Answer: {results_sample.iloc[i]['predicted_answer']}")
        print("-" * 80)

if __name__ == "__main__":
    main()