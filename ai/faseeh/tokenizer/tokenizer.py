# import temp file for temprary file creation
import tempfile
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
    def __init__(self, sentence_pieace_file_name,vocab_size, **kwargs):
        
        args = dict(tokenizer_file=sentence_pieace_file_name,
            name_or_path=self.path,
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
        super().__init__(**args)
        
    @staticmethod
    def train(vocab_size,dataset_iterator):
        # save the trainer into a temp file
        with tempfile.NamedTemporaryFile() as temp_file:
            tokenizer = SentencePieceBPETokenizer()
            tokenizer.train_from_iterator(
                iterator=dataset_iterator,
                vocab_size=vocab_size,
                show_progress=True,
                special_tokens= list(FaseehTokenizer.special_tokens_encoder.keys()))
            tokenizer.save_model(temp_file.name)
            # copy the temp file to the path
            f_tokenizer =  FaseehTokenizer(temp_file.name,vocab_size)
            # remove the temp file
            return f_tokenizer