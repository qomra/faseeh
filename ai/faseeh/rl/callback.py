import os
import json
from transformers import TrainerCallback

class SimpleJsonlLogger(TrainerCallback):
    """
    Simple callback to log metrics to a JSONL file.
    """
    def __init__(self, output_dir):
        """
        Initialize the logger.
        
        Args:
            output_dir: Directory to save the log file
        """
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "training_log.jsonl")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when the trainer logs metrics.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            logs: Dictionary of logged values
        """
        if logs is None:
            return
            
        # Append the logs to the JSONL file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(logs, ensure_ascii=False) + '\n')


class DebugLoggingCallback(TrainerCallback):
    """
    Callback to log detailed debugging information for GRPO training.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.debug_file = os.path.join(output_dir, "debug_samples.jsonl")
        self.metrics_file = os.path.join(output_dir, "training_log.jsonl")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear previous debug file
        with open(self.debug_file, 'w') as f:
            f.write("")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log regular metrics"""
        if logs is None:
            return
            
        # Append the logs to the JSONL file
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(logs, ensure_ascii=False) + '\n')
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log detailed sample information for debugging"""
        # Extract trainer from kwargs
        trainer = kwargs.get("trainer", None)
        if trainer is None or not hasattr(trainer, "_last_prompt") or not hasattr(trainer, "_last_completions"):
            return
        
        # Get the last prompt and completions
        last_prompt = trainer._last_prompt if hasattr(trainer, "_last_prompt") else None
        last_completions = trainer._last_completions if hasattr(trainer, "_last_completions") else None
        last_rewards = trainer._last_rewards if hasattr(trainer, "_last_rewards") else None
        
        if not last_prompt or not last_completions:
            return
        
        # Create debug info
        debug_info = {
            "step": state.global_step,
            "prompt": last_prompt[0] if last_prompt and len(last_prompt) > 0 else None,
            "completion": last_completions[0][0]["content"] if last_completions and len(last_completions) > 0 else None,
            "rewards": last_rewards
        }
        
        # Append to debug file
        with open(self.debug_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(debug_info, ensure_ascii=False) + '\n')

