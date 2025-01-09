
import os
import logging
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
            max_model_input_sizes = 512
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
        
        f_tokenizer =  FaseehTokenizer(path,sentence_piece_file,vocab_size)
        # remove the temp file
        return f_tokenizer
    