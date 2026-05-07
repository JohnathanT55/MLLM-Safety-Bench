"""
HuggingFace Cache Checker
Check cache status for the MM-SafetyBench dataset
"""

import os
from huggingface_hub import scan_cache_dir


def default_cache_dir() -> str:
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')


def check_hf_cache():
    """Check HuggingFace cache"""
    print("=" * 60)
    print("HuggingFace Cache Checker")
    print("=" * 60)
    
    cache_info = scan_cache_dir()
    
    print(f"\nTotal disk usage: {cache_info.size_on_disk_str}")
    print(f"Repository count: {len(cache_info.repos)}")
    
    # Find MM-SafetyBench-related cache entries
    mm_safety_repos = [
        repo for repo in cache_info.repos 
        if 'MM-SafetyBench' in repo.repo_id
    ]
    
    if not mm_safety_repos:
        print("\n[INFO] No MM-SafetyBench cache found")
        print("The dataset will be downloaded and cached automatically on first run")
        print(f"\nDefault cache location: {default_cache_dir()}")
        return None
    
    print(f"\nFound {len(mm_safety_repos)} MM-SafetyBench-related cache entries:\n")
    
    for repo in mm_safety_repos:
        print(f"Dataset: {repo.repo_id}")
        print(f"  Disk usage: {repo.size_on_disk / (1024**2):.2f} MB")
        print()
    
    print("=" * 60)
    print("Cache location:")
    print(f"  {default_cache_dir()}")
    print("\nNotes:")
    print("  - The dataset will be downloaded automatically on first run")
    print("  - Subsequent runs will use the cache and avoid re-downloading")
    print("  - You can run `huggingface-cli delete-cache` to clear cache")
    
    return cache_info


def verify_dataset_cache(
    dataset_name: str = "PKU-Alignment/MM-SafetyBench",
    config: str = "Illegal_Activitiy",
    split: str = "TYPO"
):
    """Verify a specific dataset configuration"""
    print(f"\nVerifying dataset: {dataset_name}/{config} (split: {split})")
    print("-" * 60)
    
    try:
        from datasets import load_dataset
        
        print("Attempting to load dataset...")
        dataset = load_dataset(
            dataset_name,
            config,
            split=split,
            streaming=True
        )
        
        first_sample = next(iter(dataset))
        print(f"[OK] Dataset is accessible")
        print(f"  Sample fields: {list(first_sample.keys())}")
        
        if 'image' in first_sample or 'Image' in first_sample:
            print(f"  Contains images: Yes")
        
        return True
        
    except Exception as e:
        print(f"[INFO] Dataset will be downloaded on first access: {type(e).__name__}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="HuggingFace cache checker")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify MM-SafetyBench dataset cache"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Illegal_Activitiy",
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="TYPO",
        help="Dataset split"
    )
    
    args = parser.parse_args()
    
    check_hf_cache()
    
    if args.verify:
        verify_dataset_cache(
            config=args.config,
            split=args.split
        )


if __name__ == "__main__":
    main()
