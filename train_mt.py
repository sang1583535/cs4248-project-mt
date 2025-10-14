import torch
from datasets import load_from_disk
import argparse
import yaml
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


def main():
    parser = argparse.ArgumentParser(description="Train NMT Model")
    parser.add_argument("--config", type=str, default="./configs/mT5-small-training.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    train_dataset_path = config['training_dataset_path']
    train_dataset = load_from_disk(train_dataset_path)

    model_name = config['model_name']

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        use_cache=False,  # Disable cache during training
        # dtype=torch.float16,  # Use float16 for reduced memory usage
        # device_map="auto"  # Enable automatic device mapping
    )
    model.gradient_checkpointing_enable()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = config['training_args']
    seq2seq_training_args = Seq2SeqTrainingArguments(**training_args)

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=seq2seq_training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()

