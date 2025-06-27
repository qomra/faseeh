import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from transformers import AutoTokenizer, AutoModel
import torch
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
from nltk.tokenize import sent_tokenize

class AttentionVisualizer:
    def __init__(self, model_name="distilbert-base-uncased", device="cpu", phrase_token_num=3):
        """
        Initialize the attention visualizer.
        
        Args:
            model_name (str): The pretrained model to use for attention visualization
            device (str): Device to run the model on ('cpu' or 'cuda')
            phrase_token_num (int): Convolution kernel size for phrase-level features
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
        self.model.eval()
        self.phrase_token_num = phrase_token_num
    
    def get_attention_scores(self, query, context):
        """
        Calculate attention scores between query and context tokens.
        
        Args:
            query (str): Query text (e.g., a question)
            context (str): Context text to analyze
            
        Returns:
            tuple: (query_tokens, context_tokens, attention_scores, all_tokens)
        """
        # Process the combined text (query + context)
        combined_text = query + " " + context
        
        # Tokenize
        inputs = self.tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # Get the token IDs and tokens
        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Tokenize query separately to find its length
        query_tokens_ids = self.tokenizer(query, return_tensors="pt").input_ids[0]
        query_length = len(query_tokens_ids)
        
        # Define the query and context token indices
        # Approximation: We'll consider the first query_length tokens as the query
        # and the rest as the context
        query_indices = list(range(min(query_length, len(all_tokens))))
        context_indices = list(range(min(query_length, len(all_tokens)), len(all_tokens)))
        
        # If we don't have enough tokens for both, make a reasonable division
        if not context_indices:
            mid_point = len(all_tokens) // 2
            query_indices = list(range(min(mid_point, len(all_tokens))))
            context_indices = list(range(min(mid_point, len(all_tokens)), len(all_tokens)))
        
        # Run inference to get attention scores
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the attention weights from the last layer
        attentions = outputs.attentions[-1].cpu().numpy()[0]  # [num_heads, seq_len, seq_len]
        
        # Average across attention heads
        avg_attention = np.mean(attentions, axis=0)  # [seq_len, seq_len]
        
        # Extract query-to-context attention
        query_tokens = [all_tokens[i] for i in query_indices]
        context_tokens = [all_tokens[i] for i in context_indices]
        
        # Extract the submatrix of attention from query tokens to context tokens
        attention_slice = avg_attention[query_indices, :][:, context_indices]
        
        return query_tokens, context_tokens, attention_slice, all_tokens
    
    def calculate_importance_scores(self, attention_matrix):
        """
        Calculate the importance scores for tokens using the 1D convolution method
        described in the paper.
        
        Args:
            attention_matrix: Attention scores from query to context tokens
            
        Returns:
            numpy.ndarray: Importance scores for context tokens
        """
        # If matrix is empty, return empty array
        if attention_matrix.size == 0:
            return np.array([])
        
        # Get dimensions
        num_query_tokens, num_context_tokens = attention_matrix.shape
        
        # Define kernel size for convolution
        kernel_size = min(self.phrase_token_num, num_context_tokens)
        if kernel_size <= 0:
            return np.zeros(num_context_tokens)
        
        # Initialize feature importance matrix
        feature_importance = np.zeros_like(attention_matrix)
        
        # Apply 1D convolution for each query token
        for i in range(num_query_tokens):
            # For each valid position
            for j in range(num_context_tokens - kernel_size + 1):
                # Apply convolution: sum attention over a window of tokens
                feature_importance[i, j] = np.sum(attention_matrix[i, j:j+kernel_size])
        
        # Handle the edge cases near the end of the sequence
        for i in range(num_query_tokens):
            for j in range(num_context_tokens - kernel_size + 1, num_context_tokens):
                # Use whatever part of the kernel fits
                remaining = num_context_tokens - j
                feature_importance[i, j] = np.sum(attention_matrix[i, j:j+remaining])
        
        # Sum along the query dimension to get overall importance for each context token
        token_importance = np.sum(feature_importance, axis=0)
        
        return token_importance
    
    def visualize_token_attention(self, query, context, threshold=0.7, figsize=(14, 10)):
        """
        Visualize token attention scores between query and context.
        
        Args:
            query (str): Query text (e.g., a question)
            context (str): Context text to analyze
            threshold (float): Threshold for highlighting important tokens
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get attention scores
        query_tokens, context_tokens, attention_matrix, _ = self.get_attention_scores(query, context)
        
        # Calculate token importance
        token_importance = self.calculate_importance_scores(attention_matrix)
        
        # Handle empty data case
        if len(token_importance) == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Not enough tokens to visualize attention", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Normalize importance scores to [0, 1]
        min_score = np.min(token_importance)
        max_score = np.max(token_importance)
        if max_score > min_score:
            normalized_scores = (token_importance - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(token_importance)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # 1. Plot token importance scores
        bars = ax1.bar(range(len(context_tokens)), normalized_scores, color='skyblue')
        
        # Highlight important tokens based on threshold
        for i, score in enumerate(normalized_scores):
            if score > threshold:
                bars[i].set_color('darkblue')
        
        # Add threshold line
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
        
        # Set x-axis labels
        # If too many tokens, limit the display
        max_display = 30
        if len(context_tokens) > max_display:
            step = len(context_tokens) // max_display
            x_ticks = range(0, len(context_tokens), step)
            ax1.set_xticks(x_ticks)
            ax1.set_xticklabels([context_tokens[i] for i in x_ticks], rotation=45, ha='right')
        else:
            ax1.set_xticks(range(len(context_tokens)))
            ax1.set_xticklabels(context_tokens, rotation=45, ha='right')
        
        # Set labels and title
        ax1.set_xlabel('Context Tokens')
        ax1.set_ylabel('Normalized Importance Score')
        ax1.set_title('Token Importance Scores in InfiniRetri (1D Convolution Method)')
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # 2. Visualization of context tokens with highlighting based on importance
        token_colors = []
        for score in normalized_scores:
            if score <= threshold:
                # Light blue for less important tokens
                intensity = score / threshold
                token_colors.append((0.7, 0.7 + 0.3 * intensity, 1.0))
            else:
                # Dark blue gradient for important tokens
                intensity = (score - threshold) / (1 - threshold)
                token_colors.append((0, 0, 1.0 - 0.5 * intensity))
        
        # Create colored rectangles for each token
        max_display_tokens = min(40, len(context_tokens))
        display_step = max(1, len(context_tokens) // max_display_tokens)
        
        for i in range(0, len(context_tokens), display_step):
            position = i // display_step
            token = context_tokens[i]
            color = token_colors[i]
            
            rect = patches.Rectangle((position - 0.4, 0), 0.8, 1, linewidth=1, 
                                   edgecolor='black', facecolor=color, alpha=0.7)
            ax2.add_patch(rect)
            
            ax2.text(position, 0.5, token, ha='center', va='center', 
                   color='black' if sum(color) > 1.5 else 'white',
                   fontsize=9, fontweight='bold' if sum(color) < 1.5 else 'normal')
        
        # Set axis for token visualization
        ax2.set_xlim(-0.5, max_display_tokens - 0.5)
        ax2.set_ylim(0, 1)
        ax2.set_title('Context Tokens with Importance Highlighting')
        ax2.axis('off')
        
        # 3. Plot heatmap of attention
        # Limit dimensions for visualization clarity
        max_query = min(10, len(query_tokens))
        max_context = min(30, len(context_tokens))
        
        display_attn = attention_matrix[:max_query, :max_context]
        im = ax3.imshow(display_attn, cmap='Blues', aspect='auto')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label('Attention Weight')
        
        # Set axis labels
        ax3.set_xlabel('Context Tokens')
        ax3.set_ylabel('Query Tokens')
        
        # Set tick labels
        ax3.set_xticks(range(max_context))
        ax3.set_xticklabels(context_tokens[:max_context], rotation=45, ha='right')
        ax3.set_yticks(range(max_query))
        ax3.set_yticklabels(query_tokens[:max_query])
        
        ax3.set_title('Attention From Query to Context Tokens')
        
        plt.tight_layout()
        return fig
    
    def visualize_attention_matrix(self, query, context, figsize=(10, 8)):
        """
        Visualize the attention matrix between query and context tokens.
        
        Args:
            query (str): Query text (e.g., a question)
            context (str): Context text to analyze
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get attention scores
        query_tokens, context_tokens, attention_matrix, _ = self.get_attention_scores(query, context)
        
        # Handle empty data case
        if attention_matrix.size == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Not enough tokens to visualize attention matrix", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap of attention
        im = ax.imshow(attention_matrix, cmap='Blues')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        # Set axis labels
        ax.set_xlabel('Context Tokens')
        ax.set_ylabel('Query Tokens')
        
        # Set tick labels
        # If too many tokens, limit the display
        max_query_display = min(15, len(query_tokens))
        max_context_display = min(20, len(context_tokens))
        
        if len(query_tokens) > max_query_display:
            step = len(query_tokens) // max_query_display
            y_ticks = range(0, len(query_tokens), step)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([query_tokens[i] for i in y_ticks])
        else:
            ax.set_yticks(range(len(query_tokens)))
            ax.set_yticklabels(query_tokens)
            
        if len(context_tokens) > max_context_display:
            step = len(context_tokens) // max_context_display
            x_ticks = range(0, len(context_tokens), step)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([context_tokens[i] for i in x_ticks], rotation=45, ha='right')
        else:
            ax.set_xticks(range(len(context_tokens)))
            ax.set_xticklabels(context_tokens, rotation=45, ha='right')
        
        ax.set_title('Query-to-Context Attention Matrix')
        
        plt.tight_layout()
        return fig
        
    def visualize_extraction_process(self, query, context, top_k=20, figsize=(14, 12)):
        """
        Visualize the InfiniRetri extraction process.
        
        Args:
            query (str): Query text (e.g., a question)
            context (str): Context text to analyze
            top_k (int): Number of top tokens to highlight
            figsize (tuple): Figure size
            
        Returns:
            tuple: (figure, extracted_sentences)
        """
        # Get attention scores
        query_tokens, context_tokens, attention_matrix, all_tokens = self.get_attention_scores(query, context)
        
        # Calculate token importance using the 1D convolution approach
        token_importance = self.calculate_importance_scores(attention_matrix)
        
        # Handle empty data case
        if len(token_importance) == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Not enough tokens to visualize extraction process", 
                   horizontalalignment='center', verticalalignment='center')
            return fig, []
        
        # Get top-k important tokens
        top_k_actual = min(top_k, len(token_importance))
        if top_k_actual <= 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No important tokens found", 
                   horizontalalignment='center', verticalalignment='center')
            return fig, []
            
        top_k_indices = np.argsort(token_importance)[::-1][:top_k_actual]
        
        # For visualization, identify which sentences these tokens belong to
        # Tokenize the context by sentences
        sentences = sent_tokenize(context)
        
        # Extract sentences containing important tokens
        # This is a simplified approach - in a real implementation we would map tokens back to sentences more accurately
        # Here we'll just use the first half of the important tokens to identify important sentences
        
        # Create a simple mapping from tokens to character positions
        # This is an approximation
        important_sentences = set()
        
        # For demo purposes, just select a few sentences
        num_sentences = len(sentences)
        if num_sentences > 0:
            # Use modulo to select sentences based on token indices
            for idx in top_k_indices:
                sent_idx = idx % num_sentences
                important_sentences.add(sent_idx)
        
        important_sentence_texts = [sentences[i] for i in important_sentences if i < len(sentences)]
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                          gridspec_kw={'height_ratios': [3, 2, 2]})
        
        # 1. Top plot: Token importance scores with top-k highlighted
        normalized_scores = np.zeros_like(token_importance)
        if np.max(token_importance) > np.min(token_importance):
            normalized_scores = (token_importance - np.min(token_importance)) / (np.max(token_importance) - np.min(token_importance))
            
        bars = ax1.bar(range(len(context_tokens)), normalized_scores, color='lightgray')
        
        # Highlight top-k tokens
        for idx in top_k_indices:
            if idx < len(bars):
                bars[idx].set_color('darkblue')
        
        # Set x-axis labels
        if len(context_tokens) > 40:
            step = len(context_tokens) // 40
            x_ticks = range(0, len(context_tokens), step)
            ax1.set_xticks(x_ticks)
            ax1.set_xticklabels([context_tokens[i] for i in x_ticks], rotation=45, ha='right')
        else:
            ax1.set_xticks(range(len(context_tokens)))
            ax1.set_xticklabels(context_tokens, rotation=45, ha='right')
        
        ax1.set_xlabel('Context Tokens')
        ax1.set_ylabel('Importance Score')
        ax1.set_title(f'Token Importance Scores with Top {top_k} Highlighted')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # 2. Middle plot: Attention heatmap (summary view)
        # Show attention from query to context with top-k highlighted
        im = ax2.imshow(attention_matrix, cmap='Blues', aspect='auto')
        
        # Mark top-k tokens on the heatmap
        for idx in top_k_indices:
            if idx < attention_matrix.shape[1]:
                # Draw a red rectangle around the column for this token
                rect = patches.Rectangle((idx-0.5, -0.5), 1, attention_matrix.shape[0], 
                                       edgecolor='red', facecolor='none', linewidth=1)
                ax2.add_patch(rect)
        
        # Set axis labels and ticks
        ax2.set_xlabel('Context Tokens')
        ax2.set_ylabel('Query Tokens')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Attention Weight')
        
        # Limit the number of ticks for readability
        max_q_ticks = min(10, len(query_tokens))
        q_step = max(1, len(query_tokens) // max_q_ticks)
        q_indices = range(0, len(query_tokens), q_step)
        ax2.set_yticks(q_indices)
        ax2.set_yticklabels([query_tokens[i] for i in q_indices])
        
        max_c_ticks = min(20, len(context_tokens))
        c_step = max(1, len(context_tokens) // max_c_ticks)
        c_indices = range(0, len(context_tokens), c_step)
        ax2.set_xticks(c_indices)
        ax2.set_xticklabels([context_tokens[i] for i in c_indices], rotation=45, ha='right')
        
        ax2.set_title('Attention from Query to Context with Top-K Important Tokens Highlighted')
        
        # 3. Bottom plot: Extracted sentences
        ax3.axis('off')
        ax3.set_title('Extracted Sentences Based on Top-K Tokens')
        
        # Format and display the extracted sentences
        extracted_text = ""
        for i, sentence in enumerate(important_sentence_texts):
            extracted_text += f"{i+1}. {sentence}\n\n"
        
        if not extracted_text:
            extracted_text = "No sentences extracted. Try adjusting parameters."
            
        ax3.text(0, 0.9, extracted_text, wrap=True, verticalalignment='top', 
               fontsize=10, transform=ax3.transAxes)
        
        plt.tight_layout()
        return fig, important_sentence_texts