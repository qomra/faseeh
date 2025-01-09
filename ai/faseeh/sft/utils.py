
import torch
from typing import Dict, Any, List



def copy_weights_to_hf_model(source_model, target_model):
    """
    Copy weights from source model to HuggingFace model
    """
    # Copy base model weights
    target_model.model.embed_tokens.weight.data.copy_(source_model.tok_embeddings.weight.data)
    target_model.model.norm.weight.data.copy_(source_model.norm.weight.data)
    
    for i in range(len(source_model.layers)):
        src_layer = source_model.layers[i]
        tgt_layer = target_model.model.layers[i]
        
        # Copy attention weights
        tgt_layer.self_attn.q_proj.weight.data.copy_(src_layer.attention.wq.weight.data)
        tgt_layer.self_attn.k_proj.weight.data.copy_(src_layer.attention.wk.weight.data)
        tgt_layer.self_attn.v_proj.weight.data.copy_(src_layer.attention.wv.weight.data)
        tgt_layer.self_attn.o_proj.weight.data.copy_(src_layer.attention.wo.weight.data)
        
        # Copy MLP weights
        tgt_layer.mlp.gate_proj.weight.data.copy_(src_layer.feed_forward.w1.weight.data)
        tgt_layer.mlp.down_proj.weight.data.copy_(src_layer.feed_forward.w2.weight.data)
        tgt_layer.mlp.up_proj.weight.data.copy_(src_layer.feed_forward.w3.weight.data)
        
        # Copy layer norms
        tgt_layer.input_layernorm.weight.data.copy_(src_layer.attention_norm.weight.data)
        tgt_layer.post_attention_layernorm.weight.data.copy_(src_layer.ffn_norm.weight.data)
    
    # Copy output layer
    target_model.lm_head.weight.data.copy_(source_model.output.weight.data)

def formatting_prompts_func(example: Dict[str, Any]) -> List[str]:
    """
    Format conversations using Llama3 template format.
    
    Args:
        example: Dictionary containing conversation data
        
    Returns:
        List of formatted conversation strings
    """
    system_prompt = "أنت مساعد مفيد ومحترم. تجيب دائماً بشكل مباشر ودقيق."  # Customize this
    output_texts = []
    
    for conversation in example["conversation"]:
        # Format each conversation turn using Llama3 template
        formatted_text = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>"
            f"{system_prompt}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>"
            f"{conversation[1]['content'].strip()}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
            f"{conversation[2]['content'].strip()}"
            "<|eot_id|>"
        )
        output_texts.append(formatted_text)
    
    return output_texts

def get_instruction_template(tokenizer):
    """
    Get the instruction template tokens for the data collator
    """
    # This will match everything up to the assistant's response
    instruction_template = tokenizer.encode(
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>"
        "أنت مساعد مفيد ومحترم. تجيب دائماً بشكل مباشر ودقيق."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>",
        add_special_tokens=False
    )
    
    return instruction_template

def get_response_template(tokenizer):
    """
    Get the response template tokens for the data collator
    """
    # This will match the start of the assistant's response
    response_template = tokenizer.encode(
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        add_special_tokens=False
    )
    
    return response_template

            