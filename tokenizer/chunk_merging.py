import os
from datasets import Dataset, concatenate_datasets, load_from_disk
import argparse

def merge_chunks(chunks_dir, output_dir):
    """Merge all chunks into a single dataset"""
    
    chunk_paths = []
    for item in sorted(os.listdir(chunks_dir)):
        chunk_path = os.path.join(chunks_dir, item)
        if os.path.isdir(chunk_path):
            chunk_paths.append(chunk_path)
    
    print(f"Found {len(chunk_paths)} chunks")
    
    datasets = []
    for chunk_path in chunk_paths:
        print(f"Loading {chunk_path}...")
        chunk_dataset = load_from_disk(chunk_path)
        datasets.append(chunk_dataset)
    
    print("Concatenating datasets...")
    merged_dataset = concatenate_datasets(datasets)
    
    print(f"Total examples: {len(merged_dataset)}")
    
    # Save merged dataset
    os.makedirs(output_dir, exist_ok=True)
    merged_dataset.save_to_disk(output_dir)
    print(f"Merged dataset saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    merge_chunks(args.chunks_dir, args.output_dir)