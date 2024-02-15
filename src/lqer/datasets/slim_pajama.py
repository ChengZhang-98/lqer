import datasets as hf_datasets


def get_raw_data_module_slim_pajama_6b() -> hf_datasets.DatasetDict:
    """
    A sampled version of the cerebras/SlimPajama-627B dataset.

    https://huggingface.co/datasets/DKYoon/SlimPajama-6B?row=5
    """
    dataset_dict = hf_datasets.load_dataset("DKYoon/SlimPajama-6B")
    return dataset_dict


def preprocess_data_module_slim_pajama_6b(
    raw_data_module,
    tokenizer,
    max_length,
    num_proc: int,
) -> hf_datasets.DatasetDict:
    if tokenizer.pad_token in ["<unk>", None]:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(["\n\n".join(examples["text"])])

    encodings = raw_data_module.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_data_module["train"].column_names,
        desc="Running tokenizer on dataset",
        num_proc=num_proc,
    )

    def group_texts(examples):
        # Concatenate all texts.
        # >>> sum([[1,2,3],[4,5,6]],[])
        # [1, 2, 3, 4, 5, 6]
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    preprocessed = encodings.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping texts",
    )

    return preprocessed
