from datasets import load_from_disk
import argparse
import os
import numpy as np


def create_subset(dataset_path, output_path, subset_size=None, subset_percentage=None, random_seed=42):
    """
    Create a subset from a large dataset.
    
    Args:
        dataset_path: Path to the original dataset
        output_path: Path where the subset will be saved
        subset_size: Number of samples to include (e.g., 100000)
        subset_percentage: Percentage of dataset to include (e.g., 0.1 for 10%)
        random_seed: Random seed for shuffling
    """
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    total_size = len(dataset)
    print(f"Original dataset size: {total_size:,} samples")
    
    # Determine subset size
    if subset_size is not None:
        subset_size = min(subset_size, total_size)
        print(f"Creating subset with {subset_size:,} samples ({subset_size/total_size*100:.2f}%)")
    elif subset_percentage is not None:
        subset_size = int(total_size * subset_percentage)
        print(f"Creating subset with {subset_size:,} samples ({subset_percentage*100:.2f}%)")
    else:
        raise ValueError("Either subset_size or subset_percentage must be provided")
    
    # Generate random indices for sampling (avoids permission issues with shuffle)
    # This ensures random sampling without requiring write access to input directory
    rng = np.random.default_rng(seed=random_seed)
    random_indices = rng.choice(total_size, size=subset_size, replace=False)
    random_indices = sorted(random_indices)  # Sort for better performance when selecting
    
    print(f"Selected {len(random_indices):,} random indices")
    print("Selecting samples...")
    subset = dataset.select(random_indices)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save subset
    print(f"Saving subset to: {output_path}")
    subset.save_to_disk(output_path)
    print(f"✓ Subset saved successfully!")
    print(f"  - Samples: {len(subset):,}")
    print(f"  - Location: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a subset from a tokenized dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create subset with 1 million samples
  python create_dataset_subset.py --input ./tokenized_dataset/WMT22_Train_Merged --output ./tokenized_dataset/WMT22_Train_Merged_1M --size 1000000

  # Create subset with 10% of data
  python create_dataset_subset.py --input ./tokenized_dataset/WMT22_Train_Merged --output ./tokenized_dataset/WMT22_Train_Merged_10pct --percentage 0.1

  # Create subset with 5% of data
  python create_dataset_subset.py --input ./tokenized_dataset/WMT22_Train_Merged --output ./tokenized_dataset/WMT22_Train_Merged_5pct --percentage 0.05
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the subset will be saved"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Number of samples in the subset (e.g., 1000000)"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=None,
        help="Percentage of dataset to include (e.g., 0.1 for 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.size is None and args.percentage is None:
        parser.error("Either --size or --percentage must be provided")
    
    if args.size is not None and args.percentage is not None:
        parser.error("Provide either --size OR --percentage, not both")
    
    create_subset(
        dataset_path=args.input,
        output_path=args.output,
        subset_size=args.size,
        subset_percentage=args.percentage,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()

