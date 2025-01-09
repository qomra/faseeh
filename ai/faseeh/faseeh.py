
import os
import json
import numpy as np
from tqdm import tqdm

from .pretrain import Pretrainer
from .tokenizer import FaseehTokenizer
from .utils import load_yaml,save_yaml,full_or_augment

from maknaz import pull

import logging

class FaseehProject:
    def __init__(self,config_path):
        self.config_path = config_path  
        self.configuration = load_yaml(config_path)
        self.root_path  = os.path.dirname(os.path.abspath(config_path))

        self.dataset_name = self.configuration["dataset"]
        self.dataset = None
        self.action_ids = [a["id"] for a in self.configuration.get("actions",[])]
        self.actions = {a["id"]:a for a in self.configuration.get("actions",[])}
        self.current_action = 0

    def _update_status(self,status):
        action_id = self.action_ids[self.current_action]
        if self.actions[action_id]["status"] != "always":
            self.actions[action_id]["status"] = status
        if status != "failed":
            self.configuration["actions"][self.current_action]["status"] = status
    
    def execute_next(self):
        if self.current_action < len(self.actions):
            action_id = self.action_ids[self.current_action]
            action = self.actions[action_id]
            if action["status"] == "done":
                self.current_action += 1
                logging.info(f"Skipping action {action_id} as it is already done")
                return True
            elif action["status"] == "ignore":
                self.current_action += 1
                logging.info(f"Skipping action {action_id} as it is ignored")
                return True
            
            # get_action_function from action["type"]
            action_function = getattr(self,action["type"])
            # execute the action
            success = action_function(**action)
            # update status of the action
            if success:
                self._update_status("done")
                self.current_action += 1
                return True
            else:
                self._update_status("failed")
                return False      
        else:
            logging.info("All actions executed")
        
    def load_dataset(self,**kwargs):
        if self.dataset is None:
            self.dataset = pull(self.dataset_name)["train"]
            self._update_status("done")
            return True
            
    def train_load_tokenizer(self,vocab_size,path,**kwargs):
        
        try:
            if not os.path.exists(path):
                self.load_dataset()
                tokenizer = FaseehTokenizer.train(path,vocab_size,self.dataset["content"])
                # make sure the directory exists
                os.makedirs(path,exist_ok=True)
                logging.info(f"Saving tokenizer to {path}")
                path = full_or_augment(path,self.root_path)
                tokenizer.save_pretrained(path)
            else:
                tokenizer = FaseehTokenizer.from_pretrained(path)  
            self.actions[self.current_action]["output"] = tokenizer
            self._update_status("done")
            return True    
        except Exception as e:
            logging.error(f"Failed to train/load tokenizer: {e}")
            self._update_status("failed")
            return False
        
    def pre_tokenize_data(self,
                          path=None,
                          tokenizer=None,
                          sample_size=-1,
                          shuffle=True,
                          **kwargs):
        path = full_or_augment(path,self.root_path)
        if tokenizer not in self.actions:
            logging.error(f"Tokenizer {tokenizer} not found")
            self._update_status("failed")
            return False
        
        all_tokens = []
        dataset = self.dataset
        if sample_size > 0:
            if shuffle:
                dataset = self.dataset.shuffle(seed=42).select(range(sample_size))
            else:
                dataset = self.dataset.select(range(sample_size))
        
        tokenizer = self.actions[tokenizer]["output"]
        try:
            for example in tqdm(dataset):
                text = f"{example['root']}:{example['content']}"
                text = text.strip()  # get rid of leading/trailing whitespace
                tokens = tokenizer.encode(text, add_special_tokens=True)  # encode the text, use BOS
                all_tokens.extend(tokens)
            
            # convert to uint16 nparray
            all_tokens = np.array(all_tokens, dtype=np.uint16)

            # create the directory if it does not exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # write the bytes
            with open(path, "wb") as f:
                f.write(all_tokens.tobytes())
            # calculate the average sequence length (they are separated by BOS=1)
            avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
            logging.info(f"Saved {path}, average seqlen: {avg_seq_len:.2f}")

            logging.info("Done.")
            self._update_status("done")
            self.actions[self.current_action]["output"] = path
            return True
        except:
            self._update_status("failed")
            return False

    def pretrain(self,
                 path,
                 params,
                 data_source,
                 **kwargs):
        path = full_or_augment(path,self.root_path)
        data_source = full_or_augment(self.actions[data_source]["output"],self.root_path)
        params["vocab_source"] = data_source
        pretrainer = Pretrainer(path,**params)
        pretrainer.train(data_source)

    def execute(self):
        while self.current_action < len(self.actions):
            logging.info(f"Executing action {self.action_ids[self.current_action]}")
            status = self.execute_next()
            if not status:
                logging.error(f"Failed to execute action {self.current_action}")
                break
            # update current yaml file
            save_yaml(self.configuration,self.config_path)
        
        