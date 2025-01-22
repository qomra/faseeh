
import logging

def tokenize_function(examples, tokenizer, block_size,**kwargs):
    outputs = tokenizer(
        [str(item) for item in examples["content"]],
        truncation=True,
        max_length=block_size,
        return_attention_mask=True,
        add_special_tokens=True
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
        "labels": outputs["input_ids"]
    }

def pre_tokenize_dataset(dataset, tokenizer, sample_size=-1, block_size=8192):
    try:
        if sample_size > 0:
            dataset = dataset.select(range(sample_size))
        
        processed_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing dataset",
                batch_size=100,
                fn_kwargs={"tokenizer": tokenizer, "block_size": block_size}
            )
        return processed_dataset
    except Exception as e:
        logging.error(f"Failed to preprocess dataset: {e}")
        return None