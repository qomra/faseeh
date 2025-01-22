
import os
import logging
import numpy as np
from tqdm import tqdm
from maknaz import pull
from transformers import AutoTokenizer


from .sft import FaseehSFTTrainer
from .tokenizer import FaseehTokenizer
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
      
    def load_dataset(self,**kwargs):
        if self.dataset is None:
            self.dataset = pull(self.dataset_name)["train"]
            self._update_status("done")
            return True
            
    def train_load_tokenizer(self,vocab_size,path,**kwargs):
        path = full_or_augment(path,self.root_path)
        kinds = {"faseeh":FaseehTokenizer,"auto":AutoTokenizer}
        try:
            if not os.path.exists(f"{path}/tokenizer.json"):
                self.load_dataset()
                tokenizer = FaseehTokenizer.train(path,vocab_size,self.dataset["content"])
                # make sure the directory exists  
                logging.info(f"Saving tokenizer to {path}")
                tokenizer.save_pretrained(path)
            else:
                logging.info(f"Loading tokenizer from {path}")
                kind = kinds.get(kwargs.get("kind","faseeh"))
                tokenizer = kind.from_pretrained(path,legacy=False)  
                # use pad token as eos token
                tokenizer.pad_token = tokenizer.eos_token
                logging.info(f"Tokenizer model max length: {tokenizer.model_max_length}")
            self._assign_output(tokenizer)
            return True    
        except Exception as e:
            logging.error(f"Failed to train/load tokenizer: {e}")
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

        if shuffle:
            dataset = self.dataset.shuffle(seed=42)
        
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

        self._assign_output(processed_dataset)
        return True

    def pretrain(self,
                 path,
                 base_model_type="faseeh",
                 base_model_name=None,
                 data_source=None,
                 params=None,
                 **kwargs):
        path = full_or_augment(path,self.root_path)

        if base_model_type == "faseeh":  
            from .pretrain import Pretrainer  
            data_source = full_or_augment(data_source,self.root_path)
            pretrainer = Pretrainer(path,vocab_source=data_source,**params)
            pretrainer.train(data_source)
        elif base_model_type == "hf":
            from .pretrain.pretrain_hf import Pretrainer
            # load tokenizer
            tokenizer = self.action_outputs.get(kwargs.get("tokenizer_id"),None)
            if tokenizer is None:
                logging.error(f"Tokenizer {tokenizer} not found")
                return False
            
            pretrainer = Pretrainer(base_model_name,tokenizer,path)
            pretrainer.train(self.dataset)
        return True

    def sft(self,
            dataset_name,
            pretrained_model_ckpt,
            sft_config,
            tokenizer_id,
            path,
            llama_config=None,
            sample_size=-1,
            **kwargs):
        # load dataset
        logging.info(f"Pulling sft dataset {dataset_name}")
        dataset = pull(dataset_name)

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
        
        pretrained_model_kind = kwargs.get("pretrained_model_kind","faseeh")
        # sft-trainer 
        sft_trainer = FaseehSFTTrainer(
            sft_config,
            tokenizer,
            pretrained_model_kind,
            pretrained_model_ckpt,
            sft_model_path,
            llama_config = llama_config,
            sample_size = sample_size
        )

        # train
        sft_trainer.train(dataset["train"])

        return True

    def generate_chat_completion(self,
                                 model_name,
                                 file_name):
        from .generator.hf import HuggingFaceWrapper
        full_path = full_or_augment(file_name,self.root_path)
        logging.info(f"Generating completions using model {model_name}")
        model = HuggingFaceWrapper(model_name)
        completions = model.generate(self.dataset)

        # store completions into jsonl file
        import json
        logging.info(f"Saving completions to {full_path}")
        with open(full_path,"w") as f:
            for index,completion in enumerate(completions):
                f.write(json.dumps({"index":index,"completion":completion}) + "\n")
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
        
        