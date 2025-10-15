from transformers import MT5ForConditionalGeneration, AutoTokenizer
import torch
import argparse
from typing import List
import os


def batch_translate_text(model, tokenizer, texts: List[str], batch_size=32, max_length=512, skip_special_tokens=True, num_beams=4):
    translations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=max_length).to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=num_beams,  # Add beam search for better translations
                length_penalty=0.6,  # Penalize very long outputs
                early_stopping=True  # Stop when all beams are finished
            )

        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)
        translations.extend(batch_translations)

    return translations


def translate_text(model, tokenizer, text, max_length=512, skip_special_tokens=True, num_beams=4):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    return tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference on NMT Model")
    parser.add_argument("--input-text", type=str, required=False, help="Input text for translation")

    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model")

    parser.add_argument("--input-file", type=str, required=False, help="Path to input file with texts to translate")
    parser.add_argument("--output-file", type=str, required=False, help="Path to output file to save translations")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing multiple texts")
    parser.add_argument("--force-generate", action='store_true', help="Force generation even if output file exists")

    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for inputs and outputs")
    parser.add_argument("--num-beams", type=int, default=4, help="Number of beams for beam search")
    
    args = parser.parse_args()

    model_path = args.model_path
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Assume we also use GPU for inference
    model.to('cuda')

    force_generate = args.force_generate
    if args.input_file and args.output_file:
        generate = False
        if force_generate:
            if os.path.exists(args.output_file):
                os.remove(args.output_file)
            generate = True
        else:
            if not os.path.exists(args.output_file):
                generate = True
            else:
                print(f"Output file {args.output_file} already exists. Use --force-generate to overwrite.")

        if generate:
            with open(args.input_file, 'r', encoding='utf-8') as infile:
                texts = [line.strip() for line in infile if line.strip()]
            
            translations = batch_translate_text(model, tokenizer, 
                                                texts, batch_size=args.batch_size, 
                                                max_length=args.max_length, num_beams=args.num_beams)
            
            with open(args.output_file, 'w', encoding='utf-8') as outfile:
                for translation in translations:
                    outfile.write(translation + '\n')
        
            print(f"Translations saved to {args.output_file}")

    if args.input_text:
        input_text = args.input_text.strip()
        translation = translate_text(model, tokenizer, input_text, 
                                    batch_size=args.batch_size, max_length=args.max_length, 
                                    num_beams=args.num_beams)
        print(f"Input: {args.input_text}")
        print(f"Translation: {translation}")
