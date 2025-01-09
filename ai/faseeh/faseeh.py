
import os
import json
import numpy as np
from tqdm import tqdm

from .tokenizer import FaseehTokenizer
from .pretrain import Pretrainer
import utils 
from maknaz import pull

import logging

class FaseehProject:
    def __init__(self,path):
        self.path = path
        # load configuration
        with open(os.path.join(self.path,"configuration.json"),"r") as f:
            self.configuration = json.load(f)
        self.dataset_name = self.configuration["dataset"]
        self.dataset = None
        self.tokenizer_vocab_size = self.configuration["tokenizer_vocab_size"]
        
    def load_tokenizer(self):
        # load tokenizer if exists
        if os.path.exists(os.path.join(self.path,"tokenizer")):
            self.tokenizer = FaseehTokenizer.from_pretrained(os.path.join(self.path,"tokenizer"))
        else:
            self.train_tokenizer()
    
    def load_dataset(self):
        if self.dataset is None:
            self.dataset = pull(self.dataset_name)["train"]
            
    def train_tokenizer(self):
        # if tokenizer folder existst, remove it
        if os.path.exists(os.path.join(self.path,"tokenizer")):
            import shutil
            shutil.rmtree(os.path.join(self.path,"tokenizer"))
        # train tokenizer
        self.load_dataset()
        self.tokenizer = FaseehTokenizer.train(self.tokenizer_vocab_size,self.dataset["content"])

    def pre_tokenize(self,experiment_path=None,sample_size=-1,shuffle=True):
        path = os.path.join(self.path,"data", f"data.bin")
        if experiment_path:
            path = os.path.join(self.path,experiment_path,"data", f"data.bin")

        if os.path.exists(path):
            return 
        
        self.load_tokenizer()

        all_tokens = []
        dataset = self.dataset
        if sample_size > 0:
            if shuffle:
                dataset = self.dataset.shuffle(seed=42).select(range(sample_size))
            else:
                dataset = self.dataset.select(range(sample_size))

        for example in tqdm(dataset):
            text = f"{example['root']}:{example['content']}"
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = self.tokenizer.encode(text, add_special_tokens=True)  # encode the text, use BOS
            all_tokens.extend(tokens)
        
        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        # write the bytes
        with open(path, "wb") as f:
            f.write(all_tokens.tobytes())
        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
        logging.info(f"Saved {path}, average seqlen: {avg_seq_len:.2f}")

        logging.info("Done.")

    def pretrain(self,experiment_config):
        # check current number of experiments
        experiments_root = os.path.join(self.path,"experiments")
        experiment_count = 0
        if os.path.exists(experiments_root):
            experiment_count = utils.count_folders(experiments_root)

        # padded 3 zeros
        experiment_id = "{:03d}".format(experiment_count) 
        experiment_path = os.path.join(experiments_root,experiment_id)
        # create folder
        os.makedirs(experiment_path,exist_ok=True)

        with open(f"{experiment_path}/config.json","w") as f:
            f.write(experiment_config)
        
        pretrainer = Pretrainer(**experiment_config)

        sample_size = -1
        shuffle = True
        if "dataset" in experiment_config:
            sample_size = experiment_config["dataset"].get("sample_size",-1)
            shuffle = experiment_config["dataset"].get("shuffle",-1)

        self.pre_tokenize(experiment_path,sample_size,shuffle)

        data_source = os.path.join(experiment_path,"data")
        pretrainer.train(data_source)
    

    @staticmethod
    def create(path,dataset,tokenizer_vocab_size):
        os.makedirs(path,exist_ok=True)
        # create configuration file if not exists
        configuration = {
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "dataset": dataset
        }
        with open(os.path.join(path,"configuration.json"),"w") as f:
            json.dump(configuration,f)
        
        return FaseehProject(path)
    
        
        