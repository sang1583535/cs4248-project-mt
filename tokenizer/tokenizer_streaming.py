import argparse
from datasets import Dataset
from transformers import MT5Tokenizer
import os
import gc
from multiprocessing import cpu_count

def process_file_in_chunks(src_file, tgt_file, tokenizer, save_dir, max_length=128, chunk_size=50000):
    """Process large files in chunks to avoid memory issues"""
    
    os.makedirs(save_dir, exist_ok=True)
    chunk_num = 0
    
    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        
        while True:
            print(f"Processing chunk {chunk_num}...")
            
            # Read chunk
            src_lines = []
            tgt_lines = []
            
            for _ in range(chunk_size):
                src_line = f_src.readline()
                tgt_line = f_tgt.readline()
                
                if not src_line or not tgt_line:
                    break
                    
                src_lines.append(src_line.strip())
                tgt_lines.append(tgt_line.strip())
            
            if not src_lines:
                break
            
            # Filter empty lines
            filtered_data = [
                (src, tgt) for src, tgt in zip(src_lines, tgt_lines)
                if src.strip() and tgt.strip()
            ]
            
            if not filtered_data:
                continue
                
            sources, targets = zip(*filtered_data)
            
            print(f"Chunk {chunk_num}: {len(sources)} examples")
            
            # Tokenize in smaller batches
            batch_size = 1000
            tokenized_examples = []
            
            for i in range(0, len(sources), batch_size):
                batch_src = sources[i:i+batch_size]
                batch_tgt = targets[i:i+batch_size]
                
                # Tokenize sources
                model_inputs = tokenizer(
                    list(batch_src),
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None
                )
                
                # Tokenize targets
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        list(batch_tgt),
                        max_length=max_length,
                        truncation=True,
                        padding=False,
                        return_tensors=None
                    )
                
                # Combine
                for j in range(len(batch_src)):
                    tokenized_examples.append({
                        "input_ids": model_inputs["input_ids"][j],
                        "attention_mask": model_inputs["attention_mask"][j],
                        "labels": labels["input_ids"][j]
                    })
            
            # Save chunk
            chunk_dataset = Dataset.from_list(tokenized_examples)
            chunk_path = os.path.join(save_dir, f"chunk_{chunk_num:04d}")
            chunk_dataset.save_to_disk(chunk_path)
            
            print(f"Saved chunk {chunk_num} to {chunk_path}")
            
            # Clean up memory
            del tokenized_examples, chunk_dataset, sources, targets
            gc.collect()
            
            chunk_num += 1

def main(src_path, tgt_path, save_dir, max_length=128, chunk_size=50000):
    print(f"Processing files: {src_path}, {tgt_path}")
    print(f"Chunk size: {chunk_size}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-large")
    
    # Process file in chunks
    print("Processing dataset in chunks...")
    process_file_in_chunks(src_path, tgt_path, tokenizer, save_dir, max_length, chunk_size)
    
    print("Processing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-efficient tokenizer for large parallel datasets.")
    parser.add_argument("--src-path", type=str, required=True, help="Path to source language file")
    parser.add_argument("--tgt-path", type=str, required=True, help="Path to target language file")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save tokenized dataset")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Number of examples per chunk")
    args = parser.parse_args()

    main(args.src_path, args.tgt_path, args.save_dir, args.max_length, args.chunk_size)