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
from peft import LoraConfig, get_peft_model


def main():
    parser = argparse.ArgumentParser(description="Train NMT Model")
    parser.add_argument("--config", type=str, default="./configs/mT5-small-training.yaml", help="Path to config file")
    parser.add_argument("--multi-gpu", action='store_true', help="Use multiple GPUs if available")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    train_dataset_paths = config['training_dataset_paths']

    # Load and concatenate datasets with multiple paths are provided
    train_datasets = [load_from_disk(path) for path in train_dataset_paths]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    model_name = config['model_name']

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    
    if config['use_lora']:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["q", "v", "k", "o"]
        )
        model = get_peft_model(model, lora_config)

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

