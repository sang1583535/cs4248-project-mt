import torch
from datasets import load_from_disk
import argparse
import yaml
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
import evaluate


def main():
    metric = evaluate.load("sacrebleu")

    def preprocess_logits_for_metrics(logits, labels):
        """
        Reduce memory usage by only keeping the argmax of logits instead of full distributions.
        This significantly reduces memory consumption during evaluation.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    parser = argparse.ArgumentParser(description="Train NMT Model")
    parser.add_argument("--config", type=str, default="./configs/mT5-small-training.yaml", help="Path to config file")
    parser.add_argument("--multi-gpu", action='store_true', help="Use multiple GPUs if available")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    train_dataset_paths = config['training_dataset_paths']
    eval_dataset_paths = config['evaluation_dataset_paths']

    # Load and concatenate datasets with multiple paths are provided
    train_datasets = [load_from_disk(path) for path in train_dataset_paths]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    eval_datasets = [load_from_disk(path) for path in eval_dataset_paths]
    eval_dataset = torch.utils.data.ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]

    model_name = config['model_name']

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    
    if config['use_lora']:
        lora_config = LoraConfig(**config['lora_args'])
        model = get_peft_model(model, lora_config)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = config['training_args']
    seq2seq_training_args = Seq2SeqTrainingArguments(**training_args)

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Load metric
    metric = evaluate.load("sacrebleu")

    trainer = Seq2SeqTrainer(
        model=model,
        args=seq2seq_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()

