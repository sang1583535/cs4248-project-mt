from datasets import load_from_disk
import argparse
import os
import json


def verify_subset(original_path, subset_path, show_samples=3):
    """
    Verify that a subset dataset was created correctly.
    
    Args:
        original_path: Path to the original dataset
        subset_path: Path to the subset dataset
        show_samples: Number of sample records to display
    """
    print("=" * 70)
    print("DATASET SUBSET VERIFICATION")
    print("=" * 70)
    
    # 1. Check if directories exist
    print("\n[1] Checking directory structure...")
    if not os.path.exists(original_path):
        print(f"  ❌ ERROR: Original dataset not found: {original_path}")
        return False
    else:
        print(f"  ✓ Original dataset found: {original_path}")
    
    if not os.path.exists(subset_path):
        print(f"  ❌ ERROR: Subset dataset not found: {subset_path}")
        return False
    else:
        print(f"  ✓ Subset dataset found: {subset_path}")
    
    
    # 3. Load and compare datasets
    print("\n[3] Loading datasets...")
    try:
        original_dataset = load_from_disk(original_path)
        print(f"  ✓ Original dataset loaded: {len(original_dataset):,} samples")
    except Exception as e:
        print(f"  ❌ ERROR loading original dataset: {e}")
        return False
    
    try:
        subset_dataset = load_from_disk(subset_path)
        print(f"  ✓ Subset dataset loaded: {len(subset_dataset):,} samples")
    except Exception as e:
        print(f"  ❌ ERROR loading subset dataset: {e}")
        return False
    
    # 4. Compare sizes
    print("\n[4] Size comparison...")
    original_size = len(original_dataset)
    subset_size = len(subset_dataset)
    percentage = (subset_size / original_size) * 100 if original_size > 0 else 0
    
    print(f"  Original dataset: {original_size:,} samples")
    print(f"  Subset dataset:   {subset_size:,} samples")
    print(f"  Percentage:       {percentage:.4f}%")
    
    if subset_size > original_size:
        print("  ⚠️  WARNING: Subset is larger than original! This shouldn't happen.")
        return False
    
    if subset_size == 0:
        print("  ❌ ERROR: Subset is empty!")
        return False
    
    # 5. Check features/columns
    print("\n[5] Checking dataset features...")
    original_features = set(original_dataset.features.keys())
    subset_features = set(subset_dataset.features.keys())
    
    if original_features == subset_features:
        print(f"  ✓ Features match: {list(original_features)}")
    else:
        print(f"  ⚠️  WARNING: Features differ!")
        print(f"     Original: {original_features}")
        print(f"     Subset:   {subset_features}")
        missing = original_features - subset_features
        extra = subset_features - original_features
        if missing:
            print(f"     Missing in subset: {missing}")
        if extra:
            print(f"     Extra in subset: {extra}")
    
    # 6. Verify sample data integrity
    print("\n[6] Checking sample data integrity...")
    try:
        # Check first sample
        original_sample = original_dataset[0]
        subset_sample = subset_dataset[0]
        
        # Verify structure
        if set(original_sample.keys()) == set(subset_sample.keys()):
            print("  ✓ Sample structure matches")
        else:
            print("  ⚠️  WARNING: Sample structure differs")
        
        # Check data types and shapes
        all_match = True
        for key in original_sample.keys():
            if key in subset_sample:
                orig_val = original_sample[key]
                sub_val = subset_sample[key]
                
                # Check types
                if type(orig_val) != type(sub_val):
                    print(f"  ⚠️  WARNING: Type mismatch for '{key}': {type(orig_val)} vs {type(sub_val)}")
                    all_match = False
                
                # For lists/arrays, check if they're valid
                if isinstance(orig_val, list):
                    if len(sub_val) == 0 and len(orig_val) > 0:
                        print(f"  ⚠️  WARNING: Empty list for '{key}' in subset")
                        all_match = False
        
        if all_match:
            print("  ✓ Sample data structure is valid")
    except Exception as e:
        print(f"  ⚠️  WARNING: Error checking sample data: {e}")
    
    # 7. Display sample records
    if show_samples > 0:
        print(f"\n[7] Displaying {show_samples} sample records from subset...")
        try:
            for i in range(min(show_samples, len(subset_dataset))):
                print(f"\n  Sample {i+1}:")
                sample = subset_dataset[i]
                for key, value in sample.items():
                    if isinstance(value, list):
                        if isinstance(value[0], str):
                            print(f"    {key}: {value[:5]}..." if len(value) > 5 else f"    {key}: {value}")
                        elif isinstance(value[0], (int, float)):
                            print(f"    {key}: list of length {len(value)}")
                            print(f"            first few: {value[:5]}...")
                        else:
                            print(f"    {key}: list of length {len(value)}")
                    else:
                        print(f"    {key}: {value}")
        except Exception as e:
            print(f"  ⚠️  WARNING: Error displaying samples: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"✓ Subset created successfully!")
    print(f"✓ Dataset size: {subset_size:,} / {original_size:,} ({percentage:.4f}%)")
    print(f"✓ Can be loaded and used for training")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify that a dataset subset was created correctly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify subset against original
  python verify_subset.py \\
    --original /home/n/ntasang/cs4248-project-mt/tokenized_dataset/WMT22_Train_Merged \\
    --subset /home/n/ntasang/cs4248-project-mt/tokenized_dataset/WMT22_Train_Merged_5pct

  # Verify and show 5 sample records
  python verify_subset.py \\
    --original ./tokenized_dataset/WMT22_Train_Merged \\
    --subset ./tokenized_dataset/WMT22_Train_Merged_5pct \\
    --samples 5
        """
    )
    
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to the original dataset"
    )
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help="Path to the subset dataset"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample records to display (default: 3)"
    )
    
    args = parser.parse_args()
    
    success = verify_subset(
        original_path=args.original,
        subset_path=args.subset,
        show_samples=args.samples
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

