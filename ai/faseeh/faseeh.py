
import os
import logging
import numpy as np
from tqdm import tqdm
from maknaz import pull

from .pretrain import Pretrainer
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
        try:
            if not os.path.exists(f"{path}/tokenizer.json"):
                self.load_dataset()
                tokenizer = FaseehTokenizer.train(path,vocab_size,self.dataset["content"])
                # make sure the directory exists  
                logging.info(f"Saving tokenizer to {path}")
                tokenizer.save_pretrained(path)
            else:
                logging.info(f"Loading tokenizer from {path}")
                tokenizer = FaseehTokenizer.from_pretrained(path,legacy=False)  
            self._update_status("done")
            self._assign_output(tokenizer)
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
                          min_seq_len=-1,
                          **kwargs):
       
        path = full_or_augment(path,self.root_path)
        if tokenizer not in self.action_outputs:
            logging.error(f"Tokenizer {tokenizer} not found")
            self._update_status("failed")
            return False
        
        all_tokens = []
        dataset = self.dataset

        if shuffle:
            dataset = self.dataset.shuffle(seed=42)
        
        tokenizer = self.action_outputs[tokenizer]
        logging.info(f"Pre-tokenizing dataset with sample size {sample_size} and min_seq_len {min_seq_len}")
        try:
            for index, example in enumerate(tqdm(dataset)):
                text = f"{example['root']}:{example['content']}"
                text = text.strip()  # get rid of leading/trailing whitespace
                tokens = tokenizer.encode(text, add_special_tokens=True)  # encode the text, use BOS
                all_tokens.extend(tokens)

                if min_seq_len > 0 and sample_size > 0 and len(all_tokens) > min_seq_len and index > sample_size:
                    logging.info(f"Reached min_seq_len {len(all_tokens)} > {min_seq_len} and sample_size {sample_size}")
                    break

            
            # convert to uint16 nparray
            all_tokens = np.array(all_tokens, dtype=np.uint16)
            logging.info(f"Pre-tokenized {len(all_tokens)} tokens")

            # create the directory if it does not exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # write the bytes
            with open(path, "wb") as f:
                f.write(all_tokens.tobytes())
            # calculate the average sequence length (they are separated by BOS=1)
            avg_seq_len = all_tokens.size / ((all_tokens == tokenizer.bos_token_id).sum())
            logging.info(f"Saved {path}, average seqlen: {avg_seq_len:.2f}")

            logging.info("Done.")
            self._update_status("done")
            self._assign_output(path)
            
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
        data_source = full_or_augment(data_source,self.root_path)
        
        pretrainer = Pretrainer(path,vocab_source=data_source,**params)
        pretrainer.train(data_source)
        return True

    def sft(self,
            dataset_name,
            pretrained_model_ckpt,
            llama_config,
            sft_config,
            tokenizer_id,
            path,
            **kwargs):
        # load dataset
        logging.info(f"Pulling sft dataset {dataset_name}")
        dataset = pull(dataset_name)

        # pretrained model path
        pretrained_model_ckpt = full_or_augment(
                pretrained_model_ckpt,
                self.root_path)

        # tokenizer 
        tokenizer = self.action_outputs[tokenizer_id]

        # sft-trainer 
        sft_trainer = FaseehSFTTrainer(
            sft_config,
            llama_config,
            tokenizer,
            pretrained_model_ckpt,
            path
        )

        # train
        sft_trainer.train(dataset["train"])

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
        
        