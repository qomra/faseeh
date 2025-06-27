import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

class InfiniRetri:
    def __init__(self, 
                 model_name="/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/mysam/oryx-3.0-Base",
                 device="cuda", chunk_size=512, phrase_token_num=3, top_k=20):
        """
        Initialize the InfiniRetri system.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_attentions=True, 
            output_hidden_states=True,
            attn_implementation="eager"
        ).to(self.device)
        self.generator = self.model
        self.model.eval()
        self.chunk_size = chunk_size
        self.phrase_token_num = phrase_token_num
        self.top_k = top_k
        self.cache_segments = []  # Now stores segment texts, not tokens
        self.original_segments = []
        self.segment_embeddings = []
        self.last_query = None
        
    def segment_text(self, text):
        """Segment text into chunks based on sentence boundaries."""
        sentence_endings = {'.', '!', '?'}
        text = ' '.join(text.split())
        segments = []
        current_segment = ""
        current_pos = 0
        
        while current_pos < len(text):
            next_boundary = len(text)
            for end_char in sentence_endings:
                pos = text.find(end_char + ' ', current_pos)
                if pos != -1 and pos < next_boundary:
                    next_boundary = pos + 1
            
            if next_boundary == len(text) and current_pos < next_boundary:
                sentence = text[current_pos:].strip()
            else:
                sentence = text[current_pos:next_boundary].strip()
            
            if not sentence:
                current_pos = next_boundary + 1
                continue
            
            temp_segment = current_segment + " " + sentence if current_segment else sentence
            tokens = self.tokenizer.encode(temp_segment, add_special_tokens=True)
            
            if len(tokens) <= self.chunk_size:
                current_segment = temp_segment
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                    current_segment = ""
                sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=True)
                if len(sentence_tokens) <= self.chunk_size:
                    current_segment = sentence
                else:
                    remaining = sentence
                    while remaining:
                        tokens = self.tokenizer.encode(remaining, add_special_tokens=True)
                        if len(tokens) <= self.chunk_size:
                            segments.append(remaining.strip())
                            remaining = ""
                        else:
                            encoded = self.tokenizer.encode(remaining[:self.chunk_size], add_special_tokens=False)
                            decoded = self.tokenizer.decode(encoded[:self.chunk_size-1], skip_special_tokens=True)
                            last_space = decoded.rfind(' ')
                            split_point = last_space if last_space > len(decoded) // 2 else len(decoded)
                            segment = remaining[:split_point].strip()
                            segments.append(segment)
                            remaining = remaining[split_point:].strip()
            
            current_pos = next_boundary + 1
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def generate_embedding(self, text):
        """Generate embedding for text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.chunk_size).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
    
    def merge_with_cache(self, segment, query=None):
        """Merge current segment with cached segments, keeping top-K most relevant."""
        if not self.cache_segments:
            return segment
        
        # Combine current cache and new segment
        print(segment)
        all_segments = self.cache_segments + [segment]
        merged_text = "\n\n".join(all_segments)
        encoded = self.tokenizer.encode(merged_text, add_special_tokens=True)
        
        # if len(encoded) <= self.chunk_size:
        #     print("No need to merge, fits within chunk_size.")
        #     return merged_text
        
        # Update query embedding if needed
        if query and query != self.last_query:
            self.last_query = query
            self.query_embedding = self.generate_embedding(query)
        
        # If no query, fall back to simple truncation
        if not hasattr(self, 'query_embedding'):
            excess = len(encoded) - self.chunk_size
            merged_tokens = encoded[:self.chunk_size]  # Simple truncation
            return self.tokenizer.decode(merged_tokens, skip_special_tokens=True)
        
        # Rank all segments by relevance
        segment_relevances = []
        for seg in all_segments:
            seg_embedding = self.generate_embedding(seg)
            relevance = cosine_similarity(self.query_embedding, seg_embedding)[0][0]
            print(relevance)
            segment_relevances.append((seg, relevance))
        
        # Sort by relevance and take top-K
        segment_relevances.sort(key=lambda x: x[1], reverse=True)
        top_k_segments = [seg for seg, _ in segment_relevances]
        
        # Merge top-K segments
        merged_text = "\n\n".join(top_k_segments)
        encoded = self.tokenizer.encode(merged_text, add_special_tokens=True)
        
        # Ensure it fits chunk_size
        if len(encoded) > self.chunk_size:
            excess = len(encoded) - self.chunk_size
            merged_tokens = encoded[:self.chunk_size]
            merged_text = self.tokenizer.decode(merged_tokens, skip_special_tokens=True)
        
        print(f"Cache updated with {len(top_k_segments)} segments, merged length: {len(encoded)}")
        return merged_text
        
    def calculate_attention_scores(self, query, context):
        """Calculate attention scores between query and context."""
        combined_text = query + " [SEP] " + context
        inputs = self.tokenizer(combined_text, return_tensors="pt", truncation=True, 
                               max_length=self.chunk_size).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions[-1]
        avg_attentions = torch.mean(attentions, dim=1)[0]
        
        query_tokens = self.tokenizer.encode(query, add_special_tokens=True)
        sep_tokens = self.tokenizer.encode(" [SEP] ", add_special_tokens=False)
        query_end = len(query_tokens) + len(sep_tokens) - 1
        
        query_indices = list(range(len(query_tokens)))
        context_indices = list(range(query_end, inputs.input_ids.shape[1]))
        
        query_to_context = avg_attentions[query_indices, :][:, context_indices]
        
        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        query_tokens = all_tokens[:query_end]
        context_tokens = all_tokens[query_end:]
        
        return query_tokens, context_tokens, query_to_context.cpu().numpy()
        
    def retrieve_important_segments(self, query, context):
        """Retrieve the most relevant segments from context based on attention scores."""
        # Split context into sentences to approximate segments
        sentences = sent_tokenize(context)
        if not sentences:
            return [context]  # Fallback if no sentence boundaries
        
        # Calculate attention scores
        _, _, attention_scores = self.calculate_attention_scores(query, context)
        
        # Tokenize the full context
        tokenized = self.tokenizer.encode_plus(context, return_offsets_mapping=True, add_special_tokens=False)
        tokens = tokenized['input_ids']
        offsets = tokenized['offset_mapping']
        
        # Map sentences to token ranges
        sentence_boundaries = []
        current_pos = 0
        for sentence in sentences:
            sentence_len = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            sentence_boundaries.append((current_pos, current_pos + sentence_len))
            current_pos += sentence_len
        
        # Compute average attention score per sentence
        sentence_importance = []
        for i, (start, end) in enumerate(sentence_boundaries):
            if end > len(attention_scores[0]):
                end = len(attention_scores[0])  # Handle truncation
            if start >= end:
                continue
            sentence_scores = attention_scores[:, start:end].mean(axis=1).sum()  # Sum across query tokens
            sentence_importance.append((sentences[i], sentence_scores))
        
        # Sort by importance and take top-K
        sentence_importance.sort(key=lambda x: x[1], reverse=True)
        top_k_segments = [seg for seg, _ in sentence_importance[:self.top_k]]
        
        print(f"Retrieved {len(top_k_segments)} important segments from context")
        return top_k_segments
    
    def process_document(self, document, query, progress_callback=None):
        """Process a document using InfiniRetri."""
        self.cache_segments = []
        self.original_segments = []
        self.segment_embeddings = []
        
        if isinstance(document, str):
            try:
                import json
                json_data = json.loads(document)
                segments = list(json_data['لسان العرب'].values())
            except (json.JSONDecodeError, KeyError):
                segments = self.segment_text(document)
        else:
            segments = self.segment_text(str(document))
        
        self.original_segments = segments
        
        for i, segment in enumerate(segments):
            if progress_callback:
                progress_callback((i + 1) / len(segments), f"Processing segment {i+1}/{len(segments)}")
            print(f"Processing segment {i+1} of {len(segments)}")
            merged_segment = self.merge_with_cache(segment, query)
            print(f"Merged segment: {merged_segment}")
            important_segments = self.retrieve_important_segments(query, merged_segment)
            self.cache_segments = important_segments  # Update cache with top-K segments
            embedding = self.generate_embedding(segment)
            self.segment_embeddings.append(embedding)
    
    def retrieve_relevant_segments(self, query, top_k=3):
        """Retrieve most relevant segments for a query."""
        if not self.segment_embeddings:
            return []
        
        query_embedding = self.generate_embedding(query)
        similarities = [(i, cosine_similarity(query_embedding, emb)[0][0]) 
                       for i, emb in enumerate(self.segment_embeddings)]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [(self.original_segments[i], score) for i, score in similarities[:top_k]]
    
    def answer_question(self, question, model_for_answering=None, top_k=3, prompt_template=None):
        """Answer a question based on processed document."""
        if not self.original_segments:
            return "No document processed."
        
        cache_text = "\n###########\n".join(self.cache_segments)
        relevant_segments = self.retrieve_relevant_segments(question, top_k)
        context = cache_text
        if len(cache_text.split()) < 50:
            segment_texts = "###########\n".join([seg for seg, _ in relevant_segments])
            context += "\n\n" + segment_texts
        
        if prompt_template is None:
            prompt_template = """Given the following context and question, provide a concise answer:

Context: {context}

Question: {question}

Answer:"""
        
        input_text = prompt_template.format(context=context, question=question)
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=500,
                return_dict_in_generate=True,
                output_attentions=True
            )
        
        full_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        print("Full output for debugging:")
        print(full_output)
        print("-" * 50)
        
        answer_start_marker = "Answer:"
        answer_end_marker = None
        
        answer_start = full_output.find(answer_start_marker) + len(answer_start_marker) if answer_start_marker in full_output else len(input_text)
        answer_end = full_output.find(answer_end_marker) if answer_end_marker and answer_end_marker in full_output else len(full_output)
        answer = full_output[answer_start:answer_end].strip()
        
        return answer

# Example usage
def demo_infinitri():
    retriever = InfiniRetri(chunk_size=512, phrase_token_num=3, top_k=2)  # top_k now refers to segments
    
    document = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by humans or other animals. 
    Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, 
    as well as other mappings of inputs.
    
    AI applications include advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon and Netflix), 
    understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), 
    automated decision-making and competing at the highest level in strategic game systems (such as chess and Go).
    
    This is unrelated content about cooking recipes and gardening tips which should be less relevant to the AI query.
    """
    
    question = "What are the applications of AI?"
    
    retriever.process_document(document, question)
    answer = retriever.answer_question(question)
    
    return answer

if __name__ == "__main__":
    result = demo_infinitri()
    print("Answer:", result)