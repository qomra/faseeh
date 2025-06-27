import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np

class InfiniRetri:
    def __init__(self, 
                 chunk_size: int = 512,
                 phrase_token_num: int = 3,
                 top_k: int = 5,
                 model=None):
        """
        Initialize InfiniRetri processor
        
        Args:
            chunk_size: Size of each text chunk
            phrase_token_num: Size of convolutional kernel for attention aggregation
            top_k: Number of tokens to retrieve
            model: LLM model instance
        """
        self.chunk_size = chunk_size
        self.phrase_token_num = phrase_token_num
        self.top_k = top_k
        self.model = model
        self.cache = []  # Store token IDs of relevant sentences
        
    def chunk_text(self, text: str) -> List[str]:
        """Step 1: Segment text into chunks based on sentence boundaries"""
        # Split text into sentences
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def merge_with_cache(self, chunk: torch.Tensor) -> torch.Tensor:
        """Step 2: Merge current chunk with cached tokens"""
        if not self.cache:
            return chunk
        
        # Convert cache to tensor if needed and concatenate
        cache_tensor = torch.tensor(self.cache)
        return torch.cat([cache_tensor, chunk])
    
    def compute_attention_scores(self, 
                               query: torch.Tensor, 
                               key: torch.Tensor) -> torch.Tensor:
        """Step 3: Compute standard attention scores"""
        # QK^T / sqrt(d)
        d = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (d ** 0.5)
        return F.softmax(attention_scores, dim=-1)
    
    def retrieve_relevant_tokens(self, 
                               attention_scores: torch.Tensor,
                               context_tokens: torch.Tensor) -> List[int]:
        """Step 4: Retrieve most relevant tokens based on attention scores"""
        # Eq. 2: Sum attention scores across heads
        if len(attention_scores.shape) > 2:
            attention_sum = attention_scores.sum(dim=1)  # Sum across heads
        
        # Eq. 3: 1D convolution for phrase-level features
        kernel = torch.ones(1, 1, self.phrase_token_num)
        padded_attention = F.pad(attention_sum.unsqueeze(0), 
                               (0, self.phrase_token_num-1), 
                               mode='constant', 
                               value=0)
        conv_result = F.conv1d(padded_attention, kernel).squeeze(0)
        
        # Eq. 4: Sum across query dimension for token importance
        importance_scores = conv_result.sum(dim=0)
        
        # Eq. 5: Select top-k tokens
        _, top_k_indices = torch.topk(importance_scores, self.top_k)
        
        return top_k_indices.tolist()
    
    def update_cache(self, 
                    context_tokens: torch.Tensor, 
                    relevant_indices: List[int], 
                    original_text: str):
        """Step 5: Cache sentence-level tokens"""
        # Convert text to sentences
        sentences = original_text.split('.')
        token_positions = np.cumsum([0] + [len(s.split()) for s in sentences[:-1]])
        
        # Find sentences containing relevant tokens
        relevant_sentences = set()
        for idx in relevant_indices:
            sentence_idx = np.searchsorted(token_positions, idx, side='right') - 1
            relevant_sentences.add(sentences[sentence_idx].strip() + '.')
        
        # Update cache with token IDs from relevant sentences
        self.cache = []
        for sentence in relevant_sentences:
            # Here we're storing the text - in practice you'd store token IDs
            self.cache.extend(sentence.split())
    
    def process_long_text(self, 
                         text: str, 
                         query: str) -> str:
        """Main processing pipeline"""
        # Chunk the input text
        chunks = self.chunk_text(text)
        response = ""
        
        for chunk in chunks:
            # Convert to tensors (in practice, this would be tokenized input)
            chunk_tensor = torch.tensor([ord(c) for c in chunk])  # Simplified representation
            query_tensor = torch.tensor([ord(c) for c in query])
            
            # Merge with cache
            input_tensor = self.merge_with_cache(chunk_tensor)
            
            # Model inference (simplified)
            if self.model:
                # Assuming model returns attention scores and output
                attention_scores, output = self.model(input_tensor, query_tensor)
            else:
                # Mock attention scores for demonstration
                attention_scores = self.compute_attention_scores(
                    query_tensor.unsqueeze(0), 
                    input_tensor.unsqueeze(0)
                )
            
            # Retrieve relevant tokens
            relevant_indices = self.retrieve_relevant_tokens(
                attention_scores, 
                input_tensor
            )
            
            # Update cache
            self.update_cache(input_tensor, relevant_indices, chunk)
            
            # Generate response (simplified)
            response += chunk[:50] + "..."  # Simplified output
            
        return response

# Example usage
if __name__ == "__main__":
    processor = InfiniRetri(chunk_size=512, phrase_token_num=3, top_k=5)
    
    # Sample long text
    sample_text = """This is a long text document. It contains multiple sentences 
    that need to be processed. The goal is to extract relevant information based 
    on a query. This method helps process texts beyond the context window."""
    
    query = "What is the goal?"
    
    result = processor.process_long_text(sample_text, query)
    print(f"Processed result: {result}")
    print(f"Cached tokens: {processor.cache[:10]}...")