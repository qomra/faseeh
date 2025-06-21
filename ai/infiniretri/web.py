import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import time
import PyPDF2  # For PDF reading

# Import the InfiniRetri implementation and AttentionVisualizer
from infinitri_implementation import InfiniRetri as OriginalInfiniRetri
from attention_visualization import AttentionVisualizer

# Extend InfiniRetri to include progress reporting
class InfiniRetri(OriginalInfiniRetri):
    def process_document(self, document, query, progress_callback=None):
        """
        Process a document with progress reporting, caching top-K relevant segments.
        
        Args:
            document (str): The document to process (JSON string)
            query (str): The query or question
            progress_callback (callable, optional): Function to report progress
        """
        self.cache_segments = []  # Store segment texts instead of tokens
        self.original_segments = []
        self.segment_embeddings = []
        
        # Parse JSON document
        import json
        json_data = json.loads(document)
        segments = json_data['لسان العرب']
        gt = [segments["كرشب"]]  # Guaranteed segment
        segments = list(segments.values())[:100]  # Limit to first 50 segments
        # Filter short segments
        #segments = [segment for segment in segments]
        segments = segments + gt  # Add guaranteed segment
        # Shuffle segments
        import random
        random.shuffle(segments)

        self.original_segments = segments
        
        total_segments = len(segments)
        print(f"Total segments to process: {total_segments}")
        
        # Process each segment sequentially with progress updates
        for i, segment in enumerate(segments):
            # Merge with cache, passing query for relevance
            merged_segment = self.merge_with_cache(segment, query)
            print(f"Merged segment {i+1}: {merged_segment}")
            
            # Retrieve top-K important segments from merged context
            top_segments = self.retrieve_important_segments(query, merged_segment)
            self.cache_segments = top_segments  # Update cache with top-K segments
            
            # Generate embedding for the original segment
            embedding = self.generate_embedding(segment)
            self.segment_embeddings.append(embedding)
            
            # # Report progress if callback is provided
            # if progress_callback:
            #     progress_callback((i + 1) / total_segments, f"Processed segment {i+1}/{total_segments}")
            print(f"Processed segment {i+1}/{total_segments}")

        

# Initialize InfiniRetri and visualizer
infinitri = InfiniRetri(chunk_size=4096, phrase_token_num=20, top_k=20,
                        model_name="/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/mysam/oryx-2.0-1B-Base-Maajim-4")
visualizer = AttentionVisualizer(phrase_token_num=3)

# LLaMA 3.2 prompt template
LLAMA_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant. Answer questions based only on the provided context. If the answer is not in the context, say you don't know.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context: {context}

Question: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

def fig_to_image(fig):
    """Convert matplotlib figure to image for display"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return Image.open(buf)

def read_file(file):
    """Read content from a file (supports .txt and .pdf)"""
    if file is None:
        return None
    if file.name.endswith('.txt') or file.name.endswith('.json'):
        with open(file.name, 'r', encoding='utf-8') as f:
            return f.read()
    elif file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file.name)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        raise ValueError("Unsupported file format. Please upload a .txt or .pdf file.")

def process_document(document_text, document_file, query, phrase_token_num=3, top_k=20, use_llama_template=True, progress=gr.Progress()):
    """
    Process a document with InfiniRetri and answer the initial query, with progress bar.
    
    Args:
        document_text (str): Document text from textbox
        document_file (File): Uploaded file object
        query (str): Initial query to guide the attention focus
        phrase_token_num (int): Phrase token number for attention calculation
        top_k (int): Number of top segments to consider
        use_llama_template (bool): Whether to use the LLaMA template
        progress (gr.Progress): Gradio progress object
        
    Returns:
        tuple: (stats, segments_info, visualization image, answer)
    """
    # Use file content if provided, otherwise use textbox content
    if document_file is not None:
        try:
            document = read_file(document_file)
        except Exception as e:
            return f"Error reading file: {str(e)}", "", None, "Cannot generate answer due to file error."
    else:
        document = document_text

    if not document or not query:
        return "Error: Please provide both document (via text or file) and query.", "", None, "Cannot generate answer without document and query."

    # Set parameters
    infinitri.phrase_token_num = phrase_token_num
    infinitri.top_k = top_k
    
    # Process document with progress
    start_time = time.time()
    infinitri.process_document(document, query, progress_callback=progress)
    processing_time = time.time() - start_time
    
    # Get document stats
    doc_size = len(document)
    cache_text = " ".join(infinitri.cache_segments)  # Join cached segments into a single string
    cache_size = len(cache_text)  # Length of the cached text in characters
    stats = f"""
    Document Processing Statistics:
    - Processing time: {processing_time:.2f} seconds
    - Document size: {doc_size} characters
    - Cache size: {cache_size} characters
    - Compression ratio: {cache_size/doc_size:.2%}
    - Number of segments: {len(infinitri.original_segments)}
    - Phrase token num: {phrase_token_num}
    - Top-K segments: {top_k}
    """
    
    # Show segment info
    segments_info = "Document segments:\n"
    for i, segment in enumerate(infinitri.original_segments[:3]):
        segments_info += f"Segment {i+1} ({len(segment)} chars): {segment[:100]}...\n\n"

    if len(infinitri.original_segments) > 3:
        segments_info += f"...and {len(infinitri.original_segments)-3} more segments\n\n"

    # Show cached content entirely with newlines between segments
    segments_info += f"Cached content ({len(cache_text)} chars):\n\n" + "\n\n".join(infinitri.cache_segments)
    
    # Generate visualization
    if document:
        sample_text = document[:min(500, len(document))]
        fig, _ = visualizer.visualize_extraction_process(query, sample_text, top_k=top_k)
        vis_img = fig_to_image(fig)
        plt.close(fig)
    else:
        vis_img = None
    
    # Generate answer to initial query using InfiniRetri's built-in function
    start_answer_time = time.time()
    
    if use_llama_template:
        answer = infinitri.answer_question(query, prompt_template=LLAMA_TEMPLATE)
    else:
        answer = infinitri.answer_question(query)
        
    answer_time = time.time() - start_answer_time
    
    formatted_answer = f"""Query: {query}

Answer (using LLaMA 3.2 template):
{answer}

Answer generated in {answer_time:.2f} seconds
"""
    
    return stats, segments_info, vis_img, formatted_answer

def answer_question(question, top_k=3, use_llama_template=True):
    # ... (keeping existing implementation unchanged)
    if not infinitri.original_segments:
        return "No document has been processed yet. Please process a document first.", ""
    
    start_time = time.time()
    
    if use_llama_template:
        answer = infinitri.answer_question(question, top_k=top_k, prompt_template=LLAMA_TEMPLATE)
    else:
        answer = infinitri.answer_question(question, top_k=top_k)
        
    answer_time = time.time() - start_time
    
    relevant_segments = infinitri.retrieve_relevant_segments(question, top_k=top_k)
    
    debug_info = f"Answer generated in {answer_time:.2f} seconds using top {top_k} segments.\n\n"
    
    if use_llama_template:
        cache_text = infinitri.tokenizer.decode(infinitri.cache_tokens)
        first_segment = relevant_segments[0][0] if relevant_segments else ""
        context_preview = (cache_text + "\n" + first_segment)[:100] + "..."
        
        prompt_example = LLAMA_TEMPLATE.format(context=context_preview, question=question)
        debug_info += f"Using LLaMA 3.2 template:\n{prompt_example}\n\n"
    
    debug_info += "Relevant segments:\n"
    for i, (segment, score) in enumerate(relevant_segments):
        debug_info += f"Segment {i+1} (similarity: {score:.4f}):\n{segment[:200]}...\n\n"
    
    cache_text = infinitri.tokenizer.decode(infinitri.cache_tokens)
    debug_info += f"Cached content:\n{cache_text[:200]}..."
    
    return answer, debug_info

def visualize_attention(query, context, phrase_token_num=3):
    # ... (keeping existing implementation unchanged)
    if not query or not context:
        return None
    
    visualizer.phrase_token_num = phrase_token_num
    fig = visualizer.visualize_token_attention(query, context)
    vis_img = fig_to_image(fig)
    plt.close(fig)
    
    return vis_img

# Create the Gradio interface
with gr.Blocks(title="InfiniRetri Demo") as demo:
    gr.Markdown("""
    # InfiniRetri Demo with LLaMA 3.2 Template
    
    This demo shows how InfiniRetri works to process long texts efficiently using attention mechanisms.
    It uses LLaMA 3.2's prompt template to format context and questions.
    
    ## How to use:
    1. In the "Document Processing" tab, either:
       - Enter text directly in the Document field, OR
       - Upload a .txt or .pdf file
    2. Enter a query to process the document and get an initial answer
    3. Watch the progress bar as segments are processed
    4. Ask follow-up questions in the "Follow-up Questions" tab
    """)
    
    with gr.Tab("Document Processing"):
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    document_input = gr.Textbox(label="Document Text", lines=15, placeholder="Enter document text here...")
                    document_file = gr.File(label="Or Upload Document File (.txt or .pdf)", file_types=[".txt", ".pdf",".json"])
                query_input = gr.Textbox(label="Initial Query", lines=2, placeholder="Enter a query to guide attention...")
                
                with gr.Row():
                    phrase_token_num = gr.Slider(minimum=1, maximum=40, value=3, step=1, 
                                             label="Phrase Token Num (kernel size)")
                    top_k_tokens = gr.Slider(minimum=5, maximum=10, value=20, step=5, 
                                          label="Top-K tokens")
                
                use_llama_template_checkbox = gr.Checkbox(label="Use LLaMA 3.2 Template", value=True)
                process_btn = gr.Button("Process Document")
            
            with gr.Column(scale=2):
                stats_output = gr.Textbox(label="Processing Stats", lines=6)
                segments_output = gr.Textbox(label="Segments Info", lines=10)
        
        with gr.Row():
            with gr.Column():
                attention_vis = gr.Image(label="Attention Visualization")
            
            with gr.Column():
                answer_output = gr.Textbox(label="Answer to Initial Query", lines=10)
    
    with gr.Tab("Follow-up Questions"):
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(label="Question", lines=2, placeholder="Ask a follow-up question...")
                top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, 
                               label="Top-K segments to retrieve")
                use_llama_template_followup = gr.Checkbox(label="Use LLaMA 3.2 Template", value=True)
                answer_btn = gr.Button("Get Answer")
            
            with gr.Column(scale=3):
                followup_answer = gr.Textbox(label="Answer", lines=10)
                debug_info = gr.Textbox(label="Debug Info", lines=10)
    
    with gr.Tab("Attention Visualization"):
        with gr.Row():
            with gr.Column(scale=2):
                viz_query = gr.Textbox(label="Query", lines=2, placeholder="Enter a query...")
                viz_context = gr.Textbox(label="Context", lines=5, placeholder="Enter context to analyze...")
                viz_phrase_num = gr.Slider(minimum=1, maximum=10, value=3, step=1, 
                                        label="Phrase Token Num")
                viz_btn = gr.Button("Visualize Attention")
            
            with gr.Column(scale=3):
                attention_matrix_vis = gr.Image(label="Attention Visualization")
    
    with gr.Tab("LLaMA 3.2 Template Preview"):
        gr.Markdown(f"""
        ## LLaMA 3.2 Template
        
        This is the template used to format context and questions:
        
        {LLAMA_TEMPLATE}
        
        The template follows LLaMA 3.2's chat format with system, user, and assistant messages.
        """)
    # Set up event handlers
    process_btn.click(
    process_document,
    inputs=[document_input, document_file, query_input, phrase_token_num, top_k_tokens, use_llama_template_checkbox],
    outputs=[stats_output, segments_output, attention_vis, answer_output]
    )

    answer_btn.click(
    answer_question,
    inputs=[question_input, top_k, use_llama_template_followup],
    outputs=[followup_answer, debug_info]
    )

    viz_btn.click(
    visualize_attention,
    inputs=[viz_query, viz_context, viz_phrase_num],
    outputs=[attention_matrix_vis]
    )

    # Run the app
if __name__ == "__main__":
    demo.launch()