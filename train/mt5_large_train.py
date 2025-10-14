import torch
from transformers import MT5Tokenizer, AutoModelForSeq2SeqLM


def get_best_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Get the best device for your use case
device = get_best_device()
print(f"\nUsing device: {device}")

if __name__ == "__main__":
    # Load the tokenizer and model
    model_name = "google/mt5-large"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Move the model to the selected device
    model.to(device)

    # Example input text
    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode the generated tokens to text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Output: {output_text}")
