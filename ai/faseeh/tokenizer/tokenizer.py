import os
import logging
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizerFast
from tokenizers import SentencePieceBPETokenizer
from datasets import load_dataset # Make sure this is correctly imported
import json # Ensure json is imported for tokenizer_config.json

# --- Your KotobDataset Class (from previous response) ---
# This part should be correctly defined and accessible.
# import json
# import glob
# import datasets
# import os
# import logging
# from maknaz.config import LOCAL_MAKNAZ_DIR # Assuming this points to your base data directory
# HUB = os.environ.get("MAKNAZ_MODULES_CACHE", LOCAL_MAKNAZ_DIR)
# class KotobDataset(datasets.GeneratorBasedBuilder):
#     ... (full definition from previous response) ...
# --- End of KotobDataset Class ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaseehTokenizer(LlamaTokenizerFast):
    # This class might still be used if you have 'faseeh' kind in your config
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

        return f_tokenizer

    def tokenize_dataset(self,dataset, path, sample_size=-1, min_seq_len=-1):
        all_tokens = []
        logging.info(f"Pre-tokenizing dataset with sample size {sample_size} and min_seq_len {min_seq_len}")
        try:
            for index, example in enumerate(tqdm(dataset)):
                text = f"{example['root']}:{example['content']}" # This 'root' might still be an issue from earlier
                text = text.strip()
                tokens = self.encode(text, add_special_tokens=True)
                all_tokens.extend(tokens)

                if min_seq_len > 0 and sample_size > 0 and len(all_tokens) > min_seq_len and index > sample_size:
                    logging.info(f"Reached min_seq_len {len(all_tokens)} > {min_seq_len} and sample_size {sample_size}")
                    break

            all_tokens = np.array(all_tokens, dtype=np.uint16)
            logging.info(f"Pre-tokenized {len(all_tokens)} tokens")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "wb") as f:
                f.write(all_tokens.tobytes())
            avg_seq_len = all_tokens.size / ((all_tokens == self.bos_token_id).sum())
            logging.info(f"Saved {path}, average seqlen: {avg_seq_len:.2f}")
            return all_tokens
        except:
            return None

class FaseehTokenizer4(LlamaTokenizerFast):
    """
    A custom Llama-like tokenizer for Arabic text, based on SentencePieceBPE,
    aligned with Llama 4's expected special tokens.
    """
    special_tokens_encoder = {
        "<|begin_of_text|>": 1,
        "<|eot|>": 2,
        "<|header_start|>": 3,
        "<|header_end|>": 4,
        "system": 5,
        "user": 6,
        "assistant": 7,
        "tool": 8,
        "<unk>": 9,
        "<pad>": 10,
        "<mask>": 11
    }

    special_tokens_decoder = {v: k for k, v in special_tokens_encoder.items()}

    def __init__(self, **kwargs):
        tokenizer_file = kwargs.pop("tokenizer_file", None)
        model_max_length = kwargs.pop("model_max_length", 8192)

        # Determine 'path' and safely remove 'name_or_path' from kwargs if it exists
        # This prevents the TypeError.
        path_from_kwargs = kwargs.pop("name_or_path", None)
        path = os.path.dirname(tokenizer_file) if tokenizer_file else path_from_kwargs
        if "bos_token" in kwargs:
            args = dict(
                tokenizer_file=tokenizer_file,
                name_or_path=path, # Now `path` is determined without conflicting with kwargs
                **kwargs # Now kwargs no longer contains 'name_or_path'
            )
        else:
            args = dict(
                tokenizer_file=tokenizer_file,
                name_or_path=path, # Now `path` is determined without conflicting with kwargs
                unk_token=FaseehTokenizer4.special_tokens_decoder.get(FaseehTokenizer4.special_tokens_encoder["<unk>"], "<unk>"),
                bos_token=FaseehTokenizer4.special_tokens_decoder.get(FaseehTokenizer4.special_tokens_encoder["<|begin_of_text|>"], "<|begin_of_text|>"),
                eos_token=FaseehTokenizer4.special_tokens_decoder.get(FaseehTokenizer4.special_tokens_encoder["<|eot|>"], "<|eot|>"),
                pad_token=FaseehTokenizer4.special_tokens_decoder.get(FaseehTokenizer4.special_tokens_encoder["<pad>"], "<pad>"),
                padding_side="right",
                model_max_length=model_max_length,
                **kwargs # Now kwargs no longer contains 'name_or_path'
            )
        if "legacy" in args:
            # remove legacy keyword argument if it exists
            args.pop("legacy", None)

        super().__init__(legacy=False, **args)

        current_added_tokens = self.added_tokens_encoder.keys()
        new_special_tokens_to_add = [
            token_str for token_str in FaseehTokenizer4.special_tokens_encoder.keys()
            if token_str not in self.vocab and token_str not in current_added_tokens
        ]

        if new_special_tokens_to_add:
            logging.info(f"Adding new special tokens: {new_special_tokens_to_add}")
            self.add_special_tokens({"additional_special_tokens": new_special_tokens_to_add})

        for token_str, expected_id in FaseehTokenizer4.special_tokens_encoder.items():
            actual_id = self.convert_tokens_to_ids(token_str)
            if actual_id is None:
                logging.error(f"Special token '{token_str}' could not be converted to an ID. Check tokenizer training/loading.")
            elif actual_id != expected_id:
                logging.warning(f"Special token '{token_str}' assigned ID {actual_id} "
                                f"instead of desired {expected_id}. Model embedding layer "
                                f"might need to be adjusted or tokenizer ID remapping applied.")

    @staticmethod
    def train(output_path, vocab_size, dataset_iterator):
        logging.info(f"Training tokenizer with vocab size {vocab_size}")
        tokenizer = SentencePieceBPETokenizer()

        special_tokens_list_for_training = sorted(
            FaseehTokenizer4.special_tokens_encoder.keys(), # Corrected here
            key=lambda x: FaseehTokenizer4.special_tokens_encoder[x] # Corrected here
        )

        tokenizer.train_from_iterator(
            iterator=dataset_iterator,
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=special_tokens_list_for_training,
        )

        os.makedirs(output_path, exist_ok=True)
        sentence_piece_file = os.path.join(output_path, "tokenizer.json")
        logging.info(f"Saving sentence piece tokenizer to {sentence_piece_file}")
        tokenizer.save(sentence_piece_file, pretty=True)

        tokenizer_config_path = os.path.join(output_path, "tokenizer_config.json")
        tokenizer_config = {
            "tokenizer_class": "FaseehTokenizer4", # Ensure this matches the class name
            "model_max_length": 8192,
            "padding_side": "right",
            "unk_token": FaseehTokenizer4.special_tokens_decoder.get(FaseehTokenizer4.special_tokens_encoder["<unk>"], "<unk>"), # Corrected here
            "bos_token": FaseehTokenizer4.special_tokens_decoder.get(FaseehTokenizer4.special_tokens_encoder["<|begin_of_text|>"], "<|begin_of_text|>"), # Corrected here
            "eos_token": FaseehTokenizer4.special_tokens_decoder.get(FaseehTokenizer4.special_tokens_encoder["<|eot|>"], "<|eot|>"), # Corrected here
            "pad_token": FaseehTokenizer4.special_tokens_decoder.get(FaseehTokenizer4.special_tokens_encoder["<pad>"], "<pad>"), # Corrected here
            "clean_up_tokenization_spaces": False,
            "add_prefix_space": False,
        }
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved tokenizer_config.json to {tokenizer_config_path}")

        f_tokenizer = FaseehTokenizer4( # Corrected here
            tokenizer_file=sentence_piece_file,
            name_or_path=output_path,
            model_max_length=8192
        )

        logging.info("Verifying special tokens after training and loading:")
        for token_str, expected_id in FaseehTokenizer4.special_tokens_encoder.items(): # Corrected here
            actual_id = f_tokenizer.convert_tokens_to_ids(token_str)
            logging.info(f"Token: '{token_str}', Desired ID: {expected_id}, Actual ID: {actual_id}")

        return f_tokenizer

    def tokenize_dataset(self, dataset, path, sample_size=-1, min_seq_len=-1):
        all_tokens = []
        logging.info(f"Pre-tokenizing dataset with sample size {sample_size} and min_seq_len {min_seq_len}")
        count_processed_examples = 0
        try:
            for index, example in enumerate(tqdm(dataset)):
                # Corrected here: Use example.get for safe access
                text = f"{example.get('title', '')} {example.get('name', '')} {example.get('content', '')}"
                text = text.strip()

                if not text:
                    logging.debug(f"Skipping empty text for example {index}")
                    continue

                tokens = self.encode(text, add_special_tokens=True)
                all_tokens.extend(tokens)

                count_processed_examples += 1
                if sample_size > 0 and count_processed_examples >= sample_size:
                    if min_seq_len <= 0 or len(all_tokens) >= min_seq_len:
                        logging.info(f"Reached sample_size {sample_size} and token count {len(all_tokens)}. Stopping pre-tokenization.")
                        break
                elif min_seq_len > 0 and len(all_tokens) >= min_seq_len:
                    logging.info(f"Reached min_seq_len {len(all_tokens)}. Stopping pre-tokenization.")
                    break

            if self.vocab_size < 2**16:
                dtype = np.uint16
            else:
                dtype = np.uint32

            all_tokens_np = np.array(all_tokens, dtype=dtype)
            logging.info(f"Pre-tokenized {len(all_tokens_np)} tokens.")

            output_dir = os.path.dirname(path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(path, "wb") as f:
                f.write(all_tokens_np.tobytes())
            logging.info(f"Saved pre-tokenized data to {path}")

            if count_processed_examples > 0:
                avg_seq_len = len(all_tokens_np) / count_processed_examples
                logging.info(f"Average tokens per processed example: {avg_seq_len:.2f}")
            else:
                logging.info("No examples processed for average sequence length calculation.")

            return all_tokens_np
        except Exception as e:
            logging.error(f"Error during dataset tokenization: {e}", exc_info=True)
            return None