
import os
import logging
import numpy as np
from tqdm import tqdm
from maknaz import pull
from transformers import AutoTokenizer


from .sft import FaseehSFTTrainer
from .tokenizer import FaseehTokenizer,FaseehTokenizer4
from .utils import load_yaml,save_yaml,full_or_augment

class FaseehProject:
    def __init__(self,config_path):
        self.config_path = config_path  
        self.configuration = load_yaml(config_path)
        self.root_path  = os.path.dirname(os.path.abspath(config_path))
        devices = self.configuration.get("devices",None)
        if devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in devices])
        self.dataset_name = self.configuration["dataset"]
        self.dataset = None
        self.action_ids = [a["id"] for a in self.configuration.get("actions",[])]
        self.actions = {a["id"]:a for a in self.configuration.get("actions",[])}
        self.current_action = 0
        self.action_outputs = {}

    def _update_status(self,status):
        action_id = self.action_ids[self.current_action]
        NO_UPDATE = ["always","ignore","failed"]
        if self.actions[action_id]["status"] not in NO_UPDATE:
            self.actions[action_id]["status"] = status
    
    def _assign_output(self,output):
        action_id = self.action_ids[self.current_action]
        logging.info(f"Assigning output to action {action_id}")
        self.action_outputs[action_id] = output

    def execute_current_action(self):
        
        action_id = self.action_ids[self.current_action]
        action = self.actions[action_id]
        if action["status"] == "done":
            logging.info(f"Skipping action {action_id} as it is already done")
            return True
        elif action["status"] == "ignore":
            logging.info(f"Skipping action {action_id} as it is ignored")
            return True
        
        # get_action_function from action["type"]
        action_function = getattr(self,action["type"])
        # execute the action
        success = action_function(**action)
        # update status of the action
        if success:
            self._update_status("done")
            return True
        else:
            self._update_status("failed")
            return False      
      
    def load_dataset(self,path=None,split="train",shuffle=False,**kwargs):
        if path is not None:
            full_path = full_or_augment(path, self.root_path)
            logging.info(f"Loading dataset from {full_path} with split {split}")
            import datasets
            if os.path.exists(full_path):
                self.dataset = datasets.load_from_disk(full_path)
            else:
                logging.error(f"Dataset path {full_path} does not exist.")
                self._update_status("failed")
                return False
            return True
        if self.dataset is None:
            self.dataset = pull(self.dataset_name)
            if split is not None:
                self.dataset = self.dataset[split]
            if shuffle:
                self.dataset = self.dataset.shuffle(seed=42)
            self._update_status("done")
            return True
    
    def train_load_tokenizer(self, vocab_size, path, **kwargs):
        full_path = full_or_augment(path, self.root_path)

        kinds = {
            "faseeh": FaseehTokenizer,
            "auto": AutoTokenizer,
            "faseeh4": FaseehTokenizer4
        }

        selected_kind_str = kwargs.get("kind", "faseeh")
        selected_kind_class = kinds.get(selected_kind_str)

        if not selected_kind_class:
            logging.error(f"Unknown tokenizer kind specified: {selected_kind_str}")
            self._update_status("failed")
            return False

        tokenizer_exists = os.path.exists(os.path.join(full_path, "tokenizer.json"))

        try:
            if not tokenizer_exists:
                logging.info(f"Tokenizer not found at {full_path}. Training a new {selected_kind_str} tokenizer with vocab size {vocab_size}...")
                self.load_dataset()

                if self.dataset is None:
                     logging.error("Dataset not loaded for tokenizer training.")
                     self._update_status("failed")
                     return False

                # Create an iterator over the 'content' column for memory efficiency
                # sample 1000 examples for training the tokenizer
                # sample = self.dataset.shuffle(seed=42).select(range(1000)) if len(self.dataset) > 1000 else self.dataset
                text_iterator = (example["content"] for example in self.dataset if "content" in example)

                if selected_kind_str == "faseeh4":
                    tokenizer = FaseehTokenizer4.train(full_path, vocab_size, text_iterator)
                elif selected_kind_str == "faseeh":
                    tokenizer = FaseehTokenizer.train(full_path, vocab_size, text_iterator)
                else:
                    logging.error(f"Training of {selected_kind_str} tokenizer is not implemented in this method.")
                    self._update_status("failed")
                    return False

                if tokenizer is None:
                    logging.error(f"Tokenizer training failed for {selected_kind_str}.")
                    self._update_status("failed")
                    return False

                logging.info(f"Saving trained tokenizer to {full_path}")
                tokenizer.save_pretrained(full_path) # Saves tokenizer.json, tokenizer_config.json, etc.

            else:
                logging.info(f"Loading {selected_kind_str} tokenizer from {full_path}")
                tokenizer = selected_kind_class.from_pretrained(full_path)

            # Post-loading/training adjustments
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logging.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token_id}).")
            elif tokenizer.pad_token is None:
                logging.warning("Tokenizer has no pad_token or eos_token. Consider setting one explicitly if padding is needed.")

            logging.info(f"Tokenizer model max length: {tokenizer.model_max_length}")
            logging.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
            self._assign_output(tokenizer)
            self._update_status("done")
            return True

        except Exception as e:
            logging.error(f"Failed to train/load tokenizer: {e}", exc_info=True)
            self._update_status("failed")
            return False
        
    def pre_tokenize_data(self,
                          path=None,
                          tokenizer_id=None,
                          sample_size=-1,
                          shuffle=True,
                          min_seq_len=-1,
                          kind="faseeh",
                          **kwargs):
       
        path = full_or_augment(path,self.root_path)
        if tokenizer_id not in self.action_outputs:
            logging.error(f"Tokenizer {tokenizer} not found")
            self._update_status("failed")
            return False
        
        
        dataset = self.dataset
        
        tokenizer = self.action_outputs[tokenizer_id]
        if kind == "faseeh":
            processed_dataset =  tokenizer.tokenize_dataset(dataset,path,sample_size,min_seq_len)
            if processed_dataset is None:
                logging.error(f"Failed to pre-tokenize dataset")
                return False
        else:
            # check if .arrow file exists
            if os.path.exists(f"{path}/dataset_info.json"):
                # load dataset
                from datasets import Dataset
                logging.info(f"Loading pre-tokenized dataset from {path}")
                processed_dataset = Dataset.load_from_disk(path)
            elif kind == "hf4":
                from .tokenizer.hf4 import pre_tokenize_dataset
                block_size = kwargs.get("block_size",8192)
                logging.info(f"Pre-tokenizing dataset with block size {block_size} {tokenizer.model_max_length}")
                processed_dataset = pre_tokenize_dataset(dataset,tokenizer,sample_size,block_size)
                if processed_dataset is None:
                    logging.error(f"Failed to pre-tokenize dataset")
                    return False
                logging.info(f"Saving pre-tokenized dataset to {path}")
                processed_dataset.save_to_disk(path)
            else:
                from .tokenizer.hf import pre_tokenize_dataset
                block_size = kwargs.get("block_size",8192)
                logging.info(f"Pre-tokenizing dataset with block size {block_size} {tokenizer.model_max_length}")
                processed_dataset = pre_tokenize_dataset(dataset,tokenizer,sample_size,block_size)
                if processed_dataset is None:
                    logging.error(f"Failed to pre-tokenize dataset")
                    return False
                logging.info(f"Saving pre-tokenized dataset to {path}")
                processed_dataset.save_to_disk(path)
        if shuffle:
            processed_dataset = processed_dataset.shuffle(seed=42)
        
        self.dataset = processed_dataset
        self._assign_output(processed_dataset)
        return True
    
    def pretrain(self,
                 path,
                 base_model_type="faseeh",
                 base_model_name=None,
                 data_source=None,
                 params=None,
                 num_train_epochs=1,
                 **kwargs):
        path = full_or_augment(path, self.root_path)

        if base_model_type == "faseeh":
            from .pretrain import Pretrainer # This is your original Faseeh Pretrainer
            data_source = full_or_augment(data_source, self.root_path)
            pretrainer = Pretrainer(path, vocab_source=data_source, **params)
            pretrainer.train(data_source)
        elif base_model_type == "hf":
            # This branch uses your original HF Pretrainer (e.g., pretrain_hf.py)
            from .pretrain.pretrain_hf import Pretrainer, PretrainerConfig # Assuming it has its own PretrainerConfig

            tokenizer = self.action_outputs.get(kwargs.get("tokenizer_id"), None)
            if tokenizer is None:
                logging.error(f"Tokenizer with ID '{kwargs.get('tokenizer_id')}' not found for HF Pretrainer.")
                self._update_status("failed")
                return False

            pretrainer_config_instance = PretrainerConfig(**(params if params is not None else {}))
            pretrainer = Pretrainer(
                model_name_or_path=base_model_name, # Original HF Pretrainer still takes base_model_name
                tokenizer=tokenizer,
                output_path=path,
                num_train_epochs=num_train_epochs,
                config=pretrainer_config_instance
            )
            pretrainer.train(self.dataset)
        elif base_model_type == "hf4": # New branch for Llama 4 Pretrainer
            from faseeh.pretrain.pretrain_hf4 import Pretrainer, PretrainerConfig # Import from the new file

            tokenizer = self.action_outputs.get(kwargs.get("tokenizer_id"), None)
            if tokenizer is None:
                logging.error(f"Tokenizer with ID '{kwargs.get('tokenizer_id')}' not found for HF4 Pretrainer.")
                self._update_status("failed")
                return False

            pretrainer_config_instance = PretrainerConfig(**(params if params is not None else {}))
            pretrainer = Pretrainer(
                tokenizer=tokenizer, # HF4 Pretrainer doesn't take base_model_name
                output_path=path,
                config=pretrainer_config_instance
            )
            pretrainer.train(self.dataset)
        else:
            logging.error(f"Unsupported base_model_type: {base_model_type}")
            self._update_status("failed")
            return False

        self._update_status("done")
        return True

    def sft(self,
            pretrained_model_ckpt,
            sft_config,
            tokenizer_id,
            path,
            llama_config=None,
            sample_size=-1,
            dataset_name=None,
            lora_config=None,
            **kwargs):
        # load dataset
        if dataset_name is not None:
            logging.info(f"Pulling sft dataset {dataset_name}")
            dataset = pull(dataset_name)
        else:
            dataset = self.dataset  
        # pretrained model path
        pretrained_model_ckpt = full_or_augment(
                pretrained_model_ckpt,
                self.root_path)
        # sft model path
        sft_model_path = full_or_augment(
                path,
                self.root_path)
        
        # output dir must be passed to sft_config, although we write the model in a separate call
        sft_config["output_dir"] = sft_model_path

        # tokenizer 
        tokenizer = self.action_outputs[tokenizer_id]
        tokenizer.pad_token_id = 8 
        
        pretrained_model_kind = kwargs.get("pretrained_model_kind","faseeh")
        # sft-trainer 
        sft_trainer = FaseehSFTTrainer(
            sft_config,
            tokenizer,
            pretrained_model_kind,
            pretrained_model_ckpt,
            sft_model_path,
            llama_config = llama_config,
            sample_size = sample_size,
            lora_config = lora_config
        )
     
        sft_trainer.train(dataset)

        return True

    def generate_chat_completion(self,
                                 model_name,
                                 file_name,
                                 lora_adapter=None,
                                 max_new_tokens=200,
                                 temprature=0.7,
                                 top_k=50,
                                 top_p=0.9,
                                 **kwargs):
        from .generator.hf import HuggingFaceWrapper
        full_path = full_or_augment(file_name,self.root_path)
        logging.info(f"Generating completions using model {model_name}")
        model = HuggingFaceWrapper(model_name,lora_adapter=lora_adapter,tokenizer=self.action_outputs.get(kwargs.get("tokenizer_id"),None))
        completions = model.generate(self.dataset,
                                     max_new_tokens,
                                     temprature,
                                     top_k,
                                     top_p)

        # store completions into jsonl file
        import json
        logging.info(f"Saving completions to {full_path}")
        results = []
        with open(full_path,"w",encoding="utf-8") as f:
            for index,completion in enumerate(completions):
                # f.write(json.dumps({"index":index,"completion":completion},indent=4,ensure_ascii=False) + "\n")
                item = {
                    "index": index,
                    "completion": completion[0]["generated_text"],  # Assuming completion is a list of dicts with 'generated_text'
                    "prompt": self.dataset["conversation"][index][1]["content"],
                    "ground_truth": self.dataset["conversation"][index][2]["content"]
                }
                results.append(item)
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info(f"Completions saved to {full_path}")
        return True

    def generate_pretrained_completion(self,
                                        model_name,
                                        file_name,
                                        dataset_id=None,
                                        max_new_tokens=100, # This parameter needs to be passed
                                        **kwargs):
        from .generator.hf import HuggingFacePretrainedCompletionWrapper
        full_path = full_or_augment(file_name,self.root_path)
        logging.info(f"Generating completions using model {model_name}")
        
        tokenizer = self.action_outputs.get(kwargs.get("tokenizer_id"),None)
        if tokenizer is None: # Added check for tokenizer
            logging.error(f"Tokenizer with ID '{kwargs.get('tokenizer_id')}' not found for generation.")
            self._update_status("failed")
            return False

        model_wrapper = HuggingFacePretrainedCompletionWrapper(model_name,tokenizer)
        
        dataset_for_generation = self.dataset # Default to self.dataset if dataset_id is None
        if dataset_id is not None:
            # If dataset_id is provided, pull that specific dataset for generation
            logging.info(f"Pulling dataset '{dataset_id}' for generation.")
            dataset_for_generation = pull(dataset_id)
            if dataset_for_generation is None:
                logging.error(f"Failed to pull dataset '{dataset_id}' for generation.")
                self._update_status("failed")
                return False
            # Ensure it's the 'train' split if pulling a DatasetDict
            if isinstance(dataset_for_generation, dict) and "train" in dataset_for_generation:
                dataset_for_generation = dataset_for_generation["train"]
        
        if dataset_for_generation is None:
            logging.error("No dataset available for generation.")
            self._update_status("failed")
            return False

        # FIX: Pass max_new_tokens to the generate method
        completions_data = model_wrapper.generate(dataset_for_generation, max_new_tokens=max_new_tokens,**kwargs)
        
        if not completions_data: # Handle case where generate returns empty due to error or no prompts
            logging.warning("No completions generated. Skipping saving file.")
            self._update_status("failed")
            return False

        # store completions into jsonl file
        logging.info(f"Saving completions to {full_path}")
        os.makedirs(os.path.dirname(full_path), exist_ok=True) # Ensure output directory exists
        import json

        # with open(full_path,"w",encoding="utf-8") as f:
        #     # FIX: Simplify logic - completions_data is now always a list of (prompt, completion, ground_truth) tuples
        #     for index,(prompt,completion,ground_truth) in enumerate(completions_data):
        #         f.write(json.dumps({
        #             "index": index,
        #             "prompt": prompt,
        #             "completion": completion,
        #             "ground_truth": ground_truth # Directly use ground_truth as provided
        #         }, indent=4, ensure_ascii=False) + "\n")
        # write the list to json file as [dicts]
        data = []
        for index, (prompt, completion, ground_truth) in enumerate(completions_data):
            data.append({
                "index": index,
                "prompt": prompt,
                "completion": completion,
                "ground_truth": ground_truth
            })
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Completions saved to {full_path}")
        
        self._update_status("done")
        return True

    def train_reward_model(self,base_model,tokenizer_id,output_dir,batch_size=2,**kwargs):
        from .rl import FaseehRewardTrainer
        tokenizer = self.action_outputs[tokenizer_id]
        full_output_dir = full_or_augment(output_dir,self.root_path)    
        trainer = FaseehRewardTrainer(
            tokenizer,
            base_model,
            full_output_dir,
            batch_size=batch_size
        )
        # if kwargs contains dataset_id, load dataset
        dataset_id = kwargs.get("dataset_id",None)
        if dataset_id:
            dataset = pull(dataset_id)
            trainer.train(dataset["train"])
        else:
            trainer.train(self.dataset)
        return True
    
    def train_policy_model(self,tokenizer_id,reward_model,ref_policy_model,policy_model,output_dir,**kwargs):
        from .rl import FaseehRLTrainer
        tokenizer = self.action_outputs[tokenizer_id]
        full_output_dir = full_or_augment(output_dir,self.root_path)
        reward_model = full_or_augment(reward_model,self.root_path)
        trainer = FaseehRLTrainer(
            tokenizer,
            reward_model,
            ref_policy_model,
            policy_model,
            full_output_dir
        )
        # if kwargs contains dataset_id, load dataset
        dataset_id = kwargs.get("dataset_id",None)
        if dataset_id:
            dataset = pull(dataset_id)
            trainer.train(dataset["train"])
        else:
            trainer.train(self.dataset)
        return True

    def train_dpo_model(self,tokenizer_id,base_model,output_dir,**kwargs):
        from .rl import FaseehDPOTrainer
        tokenizer = self.action_outputs[tokenizer_id]
        full_output_dir = full_or_augment(output_dir,self.root_path)
        trainer = FaseehDPOTrainer(
            tokenizer,
            base_model,
            full_output_dir
        )
        # if kwargs contains dataset_id, load dataset
        dataset_id = kwargs.get("dataset_id",None)
        if dataset_id:
            dataset = pull(dataset_id)
            trainer.train(dataset["train"])
        else:
            trainer.train(self.dataset)
        return True

    def train_grpo_model(self,tokenizer_id,base_model,output_dir,**kwargs):
        trainer_type =  kwargs.get("trainer_type","general") 
        if trainer_type == "general":
            from .rl import FaseehGRPOTrainer
        elif trainer_type == "arabic":
            from .rl.rl_ar import FaseehGRPOTrainer
        elif trainer_type == "llama":
            from .rl.rl_llama import FaseehGRPOTrainer
        elif trainer_type == "poetry":
            from .rl.rl_poetry import FaseehPoetryGRPOTrainer as FaseehGRPOTrainer
        
        tokenizer = self.action_outputs[tokenizer_id]
        full_output_dir = full_or_augment(output_dir,self.root_path)
        trainer = FaseehGRPOTrainer(
            tokenizer,
            base_model,
            full_output_dir
        )
        # if kwargs contains dataset_id, load dataset
        dataset_id = kwargs.get("dataset_id",None)
        if dataset_id:
            dataset = pull(dataset_id)
            trainer.train(dataset["train"])
        else:
            trainer.train(self.dataset)
        return True

    def execute(self):
        self.current_action = 0
        while self.current_action < len(self.actions):
            logging.info(f"Executing action {self.action_ids[self.current_action]}")
            status = self.execute_current_action()
            if not status:
                logging.error(f"Failed to execute action {self.current_action}")
                break
            # update current yaml file
            save_yaml(self.configuration,self.config_path)
            self.current_action += 1
        
        