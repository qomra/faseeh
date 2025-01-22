
import os
import logging
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizerFast
from tokenizers import SentencePieceBPETokenizer


class FaseehTokenizer(LlamaTokenizerFast):
    special_tokens_encoder = {
        "<|begin_of_text|>":1,
        "<|eot_id|>":2,
        "<|start_header_id|>":3,
        "<|end_header_id|>":4,
        "system":5,
        "user":6,
        "assistant":7,
        "<unk>":8, 
        "<pad>":9,
        "<mask>":10
    }
    def __init__(self, **kwargs):
        if "tokenizer_file" in kwargs:
            sentence_pieace_file_name = kwargs["tokenizer_file"]
            path = os.path.dirname(sentence_pieace_file_name)
            max_model_input_sizes = kwargs.get("max_model_input_sizes", 512)
            args = dict(
                tokenizer_file=sentence_pieace_file_name,
                name_or_path=path,
                unk_token="<unk>",
                unk_token_id=FaseehTokenizer.special_tokens_encoder["<unk>"],
                bos_token="<|begin_of_text|>",
                bos_token_id=FaseehTokenizer.special_tokens_encoder["<|begin_of_text|>"],
                eos_token="<|eot_id|>",
                eos_token_id=FaseehTokenizer.special_tokens_encoder["<|eot_id|>"],
                pad_token="<pad>",
                pad_token_id=FaseehTokenizer.special_tokens_encoder["<pad>"],
                padding_side="right",
                max_model_input_sizes=max_model_input_sizes)
            super().__init__(legacy=False,**args)
        else:
            super().__init__(legacy=False,**kwargs)
        
    @staticmethod
    def train(path,vocab_size,dataset_iterator):
        
        logging.info(f"Training tokenizer with vocab size {vocab_size}")
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train_from_iterator(
            iterator=dataset_iterator,
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens= list(FaseehTokenizer.special_tokens_encoder.keys()))
        
        sentence_piece_file = os.path.join(path, "sentence_piece.json")
        logging.info(f"Saving sentence piece tokenizer to {sentence_piece_file}")
        tokenizer.save(sentence_piece_file, pretty=True)
        
        #f_tokenizer =  FaseehTokenizer(tokenizer_file=sentence_piece_file,max_model_input_sizes=vocab_size)
        args = dict(
                tokenizer_file=sentence_piece_file,
                name_or_path=path,
                unk_token="<unk>",
                unk_token_id=FaseehTokenizer.special_tokens_encoder["<unk>"],
                bos_token="<|begin_of_text|>",
                bos_token_id=FaseehTokenizer.special_tokens_encoder["<|begin_of_text|>"],
                eos_token="<|eot_id|>",
                eos_token_id=FaseehTokenizer.special_tokens_encoder["<|eot_id|>"],
                pad_token="<pad>",
                pad_token_id=FaseehTokenizer.special_tokens_encoder["<pad>"],
                padding_side="right",
                max_model_input_sizes=vocab_size)
        f_tokenizer = LlamaTokenizerFast(legacy=False,**args)

        # remove the temp file
        return f_tokenizer
    
    def tokenize_dataset(self,dataset, path, sample_size=-1, min_seq_len=-1):
        all_tokens = []
        logging.info(f"Pre-tokenizing dataset with sample size {sample_size} and min_seq_len {min_seq_len}")
        try:
            for index, example in enumerate(tqdm(dataset)):
                text = f"{example['root']}:{example['content']}"
                text = text.strip()  # get rid of leading/trailing whitespace
                tokens = self.encode(text, add_special_tokens=True)  # encode the text, use BOS
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
            avg_seq_len = all_tokens.size / ((all_tokens == self.bos_token_id).sum())
            logging.info(f"Saved {path}, average seqlen: {avg_seq_len:.2f}")
            return all_tokens
        except:
            return None