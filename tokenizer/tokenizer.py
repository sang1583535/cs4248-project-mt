from datasets import Dataset
from transformers import MT5Tokenizer, AutoTokenizer
import os

def load_parallel_dataset(src_file, tgt_file, src_lang="zh", tgt_lang="en"):
    with open(src_file, encoding='utf-8') as f_src, open(tgt_file, encoding='utf-8') as f_tgt:
        src_lines = [line.strip() for line in f_src if line.strip()]
        tgt_lines = [line.strip() for line in f_tgt if line.strip()]

    assert len(src_lines) == len(tgt_lines), "Source and target files must have the same number of lines."

    data = [
        {"translation": {src_lang: src, tgt_lang: tgt}}
        for src, tgt in zip(src_lines, tgt_lines)
    ]
    return Dataset.from_list(data)

# extract source and target sentences
def preprocess_data(example, src_lang="zh", tgt_lang="en"):
    source = example["translation"][src_lang].strip()
    target = example["translation"][tgt_lang].strip()
    return {"source": source, "target": target}

# tokenize and store subwords
def tokenize_by_subwords(example, tokenizer):
    model_inputs = tokenizer(
        example["source"],
        max_length=128,
        truncation=True,
        padding="longest"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target"],
            max_length=128,
            truncation=True,
            padding="longest"
        )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["source_tokens"] = tokenizer.tokenize(example["source"])
    model_inputs["target_tokens"] = tokenizer.tokenize(example["target"])

    return model_inputs


def main():
    # file paths
    src_path = "./dataset/train.zh-en.zh"
    tgt_path = "./dataset/train.zh-en.en"
    save_dir = "./tokenized_dataset/ALMA_Human_Parallel"

    # load and preprocess data
    dataset = load_parallel_dataset(src_path, tgt_path)
    dataset = dataset.map(preprocess_data)
    dataset = dataset.filter(lambda x: len(x["source"]) > 0 and len(x["target"]) > 0)

    # load tokenizer
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

    # tokenize dataset
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_by_subwords(ex, tokenizer),
        batched=False,
        remove_columns=dataset.column_names
    )

    # testing
    sample = tokenized_dataset[0]
    print("Source sentence:", tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
    print("Target sentence:", tokenizer.decode(sample["labels"], skip_special_tokens=True))
    print("Source subword tokens:", sample["source_tokens"])
    print("Target subword tokens:", sample["target_tokens"])

    # export to directory
    os.makedirs(save_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(save_dir)
    print("Tokenized dataset saved.")


if __name__ == "__main__":
    main()