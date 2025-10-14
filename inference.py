from transformers import MT5ForConditionalGeneration, AutoTokenizer
import torch
import argparse

def translate_text(model, tokenizer, text, max_length=512, skip_special_tokens=False):
    inputs = tokenizer(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference on NMT Model")
    parser.add_argument("--input-text", type=str, required=False, help="Input text for translation")

    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model")

    parser.add_argument("--input-file", type=str, required=False, help="Path to input file with texts to translate")
    parser.add_argument("--output-file", type=str, required=False, help="Path to output file to save translations")
    
    args = parser.parse_args()

    model_path = args.model_path
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Assume we also use GPU for inference
    model.to('cuda')

    if args.input_file and args.output_file:
        with open(args.input_file, 'r', encoding='utf-8') as infile, open(args.output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                translation = translate_text(model, tokenizer, line)
                outfile.write(translation + '\n')
        print(f"Translations saved to {args.output_file}")

    if args.input_text:
        inputs = tokenizer(args.input_text, return_tensors='pt').to('cuda')
        translation = translate_text(model, tokenizer, args.input_text)
        print(f"Input: {args.input_text}")
        print(f"Translation: {translation}")
