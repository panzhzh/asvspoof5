#!/usr/bin/env python3
"""
ASVspoof5 Dataset Download Script
Downloads the dataset from HuggingFace and organizes it for the baseline models
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
import argparse

def download_asvspoof5(output_dir="./data/ASVspoof5"):
    """
    Download ASVspoof5 dataset from HuggingFace
    
    Args:
        output_dir (str): Directory to save the dataset
    """
    print("🔄 Starting ASVspoof5 dataset download from HuggingFace...")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download dataset
        print("📥 Downloading dataset from jungjee/asvspoof5...")
        dataset = load_dataset("jungjee/asvspoof5", cache_dir=str(output_path / "cache"))
        
        print("✅ Download completed!")
        print(f"📊 Dataset info:")
        print(f"   - Splits available: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"   - {split_name}: {len(split_data)} samples")
        
        # Save dataset to local directory
        print(f"💾 Saving dataset to {output_dir}...")
        dataset.save_to_disk(str(output_path / "asvspoof5_dataset"))
        
        print("🎉 Dataset download and save completed!")
        print(f"📝 Next steps:")
        print(f"   1. Check the dataset structure in: {output_dir}")
        print(f"   2. Read LICENSE.txt and README.txt for usage terms")
        print(f"   3. Update config files with the correct database_path")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Possible solutions:")
        print("   - Check internet connection")
        print("   - Install required packages: pip install datasets")
        print("   - Try downloading from official ASVspoof website instead")
        return False

def update_config_files(dataset_path):
    """
    Update configuration files with the correct dataset path
    
    Args:
        dataset_path (str): Path to the downloaded dataset
    """
    print("🔧 Updating configuration files...")
    
    # Update AASIST config
    aasist_config = Path("./Baseline-AASIST/config/AASIST_ASVspoof5.conf")
    if aasist_config.exists():
        print(f"📝 Please manually update {aasist_config}")
        print(f"   Change 'database_path' to: {os.path.abspath(dataset_path)}")
    
    # Note for RawNet2
    print(f"📝 For RawNet2, use --database_path={os.path.abspath(dataset_path)} when running")

def main():
    parser = argparse.ArgumentParser(description="Download ASVspoof5 dataset from HuggingFace")
    parser.add_argument("--output_dir", "-o", 
                       default="./data/ASVspoof5",
                       help="Output directory for dataset (default: ./data/ASVspoof5)")
    parser.add_argument("--update_config", "-u", 
                       action="store_true",
                       help="Update configuration files with dataset path")
    
    args = parser.parse_args()
    
    # Check if datasets library is installed
    try:
        import datasets
    except ImportError:
        print("❌ Required package 'datasets' not found!")
        print("💡 Install it with: pip install datasets")
        sys.exit(1)
    
    # Download dataset
    success = download_asvspoof5(args.output_dir)
    
    if success and args.update_config:
        update_config_files(args.output_dir)
    
    if success:
        print("\n🚀 Ready to start training!")
        print("🔥 For AASIST: cd Baseline-AASIST && python main.py --config ./config/AASIST_ASVspoof5.conf")
    else:
        print("\n🌐 Consider downloading from official website: https://www.asvspoof.org/database")

if __name__ == "__main__":
    main()