import logging
from itertools import chain
import os

logger = logging.getLogger(__name__)

def _raw_text_processing_for_tokenization(examples):
    concatenated_strings = []
    # `examples` here will contain 'kitab_id', 'title', 'name', 'volume_num', 'content'
    # where 'content' is now the full text of a single volume.
    # The title is already prepended within the 'content' field by KotobDataset.
    for text_content in examples["content"]:
        if text_content: # Ensure it's not empty
            concatenated_strings.append(text_content.strip())
    return {"text_for_tokenization": concatenated_strings}


def tokenize_and_group_texts(examples, tokenizer, block_size):
    # This function processes a batch of raw texts (each representing a volume's content)
    # and groups them into fixed-size token blocks.

    tokenized_examples = tokenizer(
        examples["text_for_tokenization"],
        add_special_tokens=False, # Crucial: False, as we are chaining all text from this batch
        return_attention_mask=False,
        return_special_tokens_mask=False,
    )

    all_blocks_from_batch = []
    # `tokenized_examples["input_ids"]` is a list of lists (one inner list per volume in the batch).
    # Chain all tokens from all volumes in this batch into one long stream.
    # This is safe because each volume is now a smaller unit, reducing the chance of OOM from a single large book.
    concatenated_tokens_from_batch = list(chain(*tokenized_examples["input_ids"]))

    # Now, chunk this large stream of tokens into fixed-size blocks
    total_length = (len(concatenated_tokens_from_batch) // block_size) * block_size
    
    for i in range(0, total_length, block_size):
        all_blocks_from_batch.append(concatenated_tokens_from_batch[i : i + block_size])
    
    result = {
        "input_ids": all_blocks_from_batch,
        "labels": all_blocks_from_batch # Labels are the same as input_ids for CLM
    }
    return result

def pre_tokenize_dataset(dataset, tokenizer, sample_size=-1, block_size=8192):
    try:
        if sample_size > 0:
            dataset = dataset.select(range(sample_size))

        logger.info("Step 1/2: Preparing raw text content for tokenization...")
        temp_dataset_for_text = dataset.map(
            _raw_text_processing_for_tokenization,
            batched=True,
            # batch_size for Step 1 can be higher as it's primarily string operations.
            batch_size=1000,
            remove_columns=dataset.column_names,
            desc="Preparing text for tokenization",
            num_proc=16 # Use multiple cores for CPU-bound string concatenation
        )

        logger.info(f"Step 2/2: Tokenizing and grouping texts into blocks of {block_size}...")
        processed_dataset = temp_dataset_for_text.map(
            tokenize_and_group_texts,
            batched=True,
            # batch_size for Step 2: Now that inputs are volumes (smaller), you can increase this.
            # Start with 100 or 500. If OOM, reduce. If it works, increase for speed.
            batch_size=10, # Increased batch_size for better performance with smaller units
            remove_columns=["text_for_tokenization"],
            desc=f"Tokenizing and grouping into {block_size} blocks",
            fn_kwargs={"tokenizer": tokenizer, "block_size": block_size},
            num_proc=4, # Try using all CPUs again, as chunks are smaller
            load_from_cache_file=True
        )
        logger.info(f"Final pre-tokenized dataset has {len(processed_dataset)} examples.")
        return processed_dataset
    except Exception as e:
        logger.error(f"Failed to preprocess dataset: {e}", exc_info=True)
        return None