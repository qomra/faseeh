import os
import torch
import argparse
import uvicorn
import json
import time
import asyncio
from typing import List, Optional, Dict, Any, Union, AsyncIterator, Callable
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager, contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TextIteratorStreamer
from threading import Thread
from peft import PeftModel, PeftConfig

# Define OpenAI API compatible request/response models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None

# Define global variables for models and tokenizer
base_model = None
analyzer_model = None
memorizer_model = None
tokenizer = None
model_config = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for loading and unloading models."""
    global base_model, analyzer_model, memorizer_model, answer_model, tokenizer, model_config
    
    # Load configuration
    model_config = ModelConfig()
    
    # Load base model and tokenizer
    print("Loading models...")
    base_model, tokenizer = load_base_model(model_config)
    
    # Load LoRA models
    print(f"Loading root analyzer from {model_config.root_analyzer_path}")
    analyzer_model = PeftModel.from_pretrained(base_model, model_config.root_analyzer_path)
    
    # Create a new instance for the memorizer to avoid conflicts
    base_model_copy, _ = load_base_model(model_config)
    
    print(f"Loading memorizer from {model_config.memorizer_path}")
    memorizer_model = PeftModel.from_pretrained(base_model_copy, model_config.memorizer_path)
    
    # Load a separate model for answer generation
    print("Loading answer model...")
    answer_model, _ = load_base_model(model_config)
    
    print("All models loaded successfully. API server is ready.")
    
    yield

app = FastAPI(title="Arabic Root Analysis API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelConfig:
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME", "/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model/meta-llama/Llama-3.1-8B-Instruct/")
        self.root_analyzer_path = os.environ.get("ROOT_ANALYZER_PATH", "models/rooter")
        self.memorizer_path = os.environ.get("MEMORIZER_PATH", "models/memorizer")
        self.max_length = int(os.environ.get("MAX_LENGTH", "4096"))
        self.load_8bit = os.environ.get("LOAD_8BIT", "False").lower() == "true"
        self.load_4bit = os.environ.get("LOAD_4BIT", "False").lower() == "true"

def load_base_model(config):
    """Load the base model with appropriate quantization settings"""
    print(f"Loading base model: {config.model_name}")
    
    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure model loading based on precision arguments
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    
    if config.load_8bit:
        model_kwargs["load_in_8bit"] = True
    elif config.load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        model_kwargs["quantization_config"] = quantization_config
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    return base_model, tokenizer

async def run_model_with_streamer(model, prompt, tokenizer, gen_kwargs):
    """Run model with TextIteratorStreamer for proper streaming"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Create a TextIteratorStreamer with a small timeout
    streamer = TextIteratorStreamer(tokenizer, timeout=10, skip_prompt=True, skip_special_tokens=True)
    
    # Set up generation parameters with streamer as callback
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "streamer": streamer,
        **gen_kwargs
    }
    
    # Create a thread to run the generation
    thread = Thread(target=lambda: model.generate(**generation_kwargs))
    thread.start()
    
    # Stream the output token by token
    for new_text in streamer:
        yield new_text
    
    thread.join()

# We're now handling the streaming and non-streaming cases directly in the chat_completion endpoint
# These standalone streaming functions are no longer needed, but we'll keep
# the run_model_with_streamer helper function for the new implementation

# The run_model_with_streamer function is unchanged

def count_tokens(text):
    """Approximate token count"""
    return len(tokenizer.encode(text))

async def stream_response(response_id: str, model: str, content_iterator, role: str = "assistant"):
    """Stream the response in OpenAI-compatible chunks."""
    # Initial chunk with role
    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': role}, 'finish_reason': None}]})}\n\n"
    
    # Stream content chunks
    try:
        buffer = ""
        async for content_chunk in content_iterator:
            if content_chunk:
                # Buffer the content to avoid sending too many small chunks
                buffer += content_chunk
                
                # Send the buffer when it reaches a certain size or contains a complete word
                if len(buffer) > 5 or " " in buffer or "\n" in buffer:
                    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': buffer}, 'finish_reason': None}]})}\n\n"
                    buffer = ""
                    
                    # Small delay to ensure the event loop can process other tasks
                    await asyncio.sleep(0.001)
        
        # Send any remaining buffered content
        if buffer:
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': buffer}, 'finish_reason': None}]})}\n\n"
    
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        # Send an error message to the client
        error_message = "\\n\\nError: " + str(e)
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': error_message}, 'finish_reason': 'error'}]})}\n\n"
    
    finally:
        # Always send the final completion message
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    """Unified endpoint for both streaming and non-streaming responses."""
    global base_model, analyzer_model, memorizer_model, answer_model, tokenizer, model_config
    
    try:
        # Parse request body
        body = await request.json()
        request_obj = ChatCompletionRequest(**body)
        
        # Extract the last user message
        user_messages = [msg for msg in request_obj.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        question = user_messages[-1].content
        response_id = f"chatcmpl-{torch.randint(0, 10000, (1,)).item()}"
        
        # Handle streaming requests
        if request_obj.stream:
            # Create an async generator for streaming all components
            async def content_stream():
                try:
                    # Variables to store the collected outputs
                    analysis_text = ""
                    lisan_text = ""
                    
                    # Yield opening think tag
                    yield "<think>"
                    
                    # 1. Stream and collect the analyzer output
                    system_prompt = "أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية."
                    prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                    
                    gen_kwargs = {
                        "max_new_tokens": 1024,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "do_sample": True,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    
                    async for chunk in run_model_with_streamer(analyzer_model, prompt, tokenizer, gen_kwargs):
                        analysis_text += chunk
                        yield chunk
                    
                    # 2. Stream and collect the memorizer output
                    system_prompt = "أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية ومعانيها من معجم لسان العرب."
                    prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{analysis_text}\n</analyze>\n\n<lisan alarab>\n"
                    
                    gen_kwargs = {
                        "max_new_tokens": 256,
                        "temperature": 0.01,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "do_sample": True,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    
                    yield "<lisan alarab>\n"
                    async for chunk in run_model_with_streamer(memorizer_model, prompt, tokenizer, gen_kwargs):
                        lisan_text += chunk
                        yield chunk
                    yield "\n</lisan alarab>\n"
                    
                    # Close the thinking section
                    yield "</think>"
                    
                    # 3. Stream the final answer
                    system_prompt = "أنت خبير في علم اللغة العربية متخصص في تفسير معاني الكلمات العربية. أجب على السؤال باستخدام السياق المقدم لك"
                    
                    # Create a prompt that includes the analysis and Lisan Al-Arab content
                    prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>السياق: {lisan_text}\n\nالسؤال: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                    
                    gen_kwargs = {
                        "max_new_tokens": request_obj.max_tokens,
                        "temperature": request_obj.temperature,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "do_sample": True,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    
                    async for chunk in run_model_with_streamer(answer_model, prompt, tokenizer, gen_kwargs):
                        yield chunk
                        
                except Exception as e:
                    print(f"Error in content_stream: {str(e)}")
                    yield f"\n\nError occurred: {str(e)}"
            
            # Return a streaming response
            return StreamingResponse(
                stream_response(response_id, request_obj.model, content_stream()),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        else:
            try:
                # Variables to store the collected outputs
                analysis_text = ""
                lisan_text = ""
                
                # 1. Run analyzer
                system_prompt = "أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية."
                prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = analyzer_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=1024,
                        temperature=0.1,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                prompt_length = inputs["input_ids"].shape[1]
                analysis_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
                
                # 2. Run memorizer
                system_prompt = "أنت خبير في علم اللغة العربية متخصص في تحديد جذور الكلمات العربية ومعانيها من معجم لسان العرب."
                prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|><analyze>\n{analysis_text}\n</analyze>\n\n<lisan alarab>\n"
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = memorizer_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=256,
                        temperature=0.1,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                prompt_length = inputs["input_ids"].shape[1]
                lisan_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
                
                # 3. Generate the complete answer
                system_prompt = "أنت خبير في علم اللغة العربية متخصص في تفسير معاني الكلمات العربية. أجب على السؤال باستخدام السياق المقدم لك"
                
                prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>السياق: {lisan_text}\n\nالسؤال: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = answer_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=request_obj.max_tokens,
                        temperature=request_obj.temperature,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                prompt_length = inputs["input_ids"].shape[1]
                final_answer = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
                
                # Create a response with thinking steps in <think> tags
                thinking = f"<think>\n<analyze>\n{analysis_text}\n</analyze>\n\n<lisan alarab>\n{lisan_text}\n</lisan alarab>\n</think>"
                complete_response = f"{thinking}\n\n{final_answer}"
                
                # Count tokens (approximate)
                prompt_tokens = sum(count_tokens(msg.content) for msg in request_obj.messages)
                completion_tokens = count_tokens(complete_response)
                total_tokens = prompt_tokens + completion_tokens
                
                # Create and return the non-streaming response
                return ChatCompletionResponse(
                    id=response_id,
                    created=int(time.time()),
                    model=request_obj.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=Message(
                                role="assistant",
                                content=complete_response
                            )
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    )
                )
            except Exception as e:
                print(f"Non-streaming error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        print(f"Request processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_args():
    parser = argparse.ArgumentParser(description='Run Arabic root analysis FastAPI server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to run the server on')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Use the current filename instead of "main"
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)