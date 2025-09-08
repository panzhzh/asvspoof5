#!/usr/bin/env python3
"""
ASVspoof5 Training Configuration

This configuration file contains all hyperparameters and paths
for training the ASVspoof5 anti-spoofing model.
"""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# Data Paths Configuration
# =============================================================================
config = {
    # Dataset paths
    "database_path": str(PROJECT_ROOT / "data" / "ASVspoof5"),
    "feature_path": str(PROJECT_ROOT / "data" / "ASVspoof5" / "features"),
    
    # Model paths
    "model_path": str(PROJECT_ROOT / "models" / "weights" / "best.pth"),
    
    # =============================================================================
    # Training Hyperparameters
    # =============================================================================
    # Batch sizes: train/dev/test separated (fallbacks provided in code)
    "batch_size_train": 128,
    "batch_size_dev": 32,
    "batch_size_test": 32,
    "num_epochs": 20,
    "loss": "CCE",  # Cross-Entropy Loss
    "track": "LA",  # Logical Access track
    
    # Evaluation settings
    "eval_all_best": True,
    "eval_output": "eval_scores_using_best_dev_model.txt",

    # PyTorch settings for reproducibility and performance
    "cudnn_deterministic_toggle": True,
    "cudnn_benchmark_toggle": True,

    # DataLoader worker settings (tune for throughput vs. stability)
    "num_workers_train": 0,
    "num_workers_dev": 0,
    "num_workers_test": 0,

    # I/O optimization (ragged memmap): block-wise sequential read with per-epoch block shuffle
    "io_block_shuffle": True,
    "io_block_mb": 512,

    # I/O cropping on load: read only needed time window (training)
    # If True, collate reads first `target_frames` frames directly from disk (per sample)
    # to reduce disk traffic. Set False to read full sample then crop on GPU.
    "io_crop_on_load": True,
    # Limit number of layers to read on training (e.g., 6 to save I/O). None -> read all.
    "io_train_layers": None,

    # Data augmentation
    "freq_aug": False,  # Frequency masking augmentation
    
    # =============================================================================
    # Model Architecture Configuration
    # =============================================================================
    "model_config": {
        "architecture": "model",  # Model class name in src.models.model
        
        # Input configuration
        "nb_samp": 64600,  # Number of samples (~4 seconds at 16kHz)
        
        # Convolutional layers
        "first_conv": 128,  # First conv layer filters
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],  # Filter configurations
        
        # Graph Attention Network
        "gat_dims": [64, 32],  # GAT layer dimensions
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],  # Pooling ratios for each layer
        "temperatures": [2.0, 2.0, 100.0, 100.0],  # Temperature parameters for attention
    },
    
    # =============================================================================
    # Optimizer Configuration
    # =============================================================================
    "optim_config": {
        "optimizer": "adam",  # Optimizer type
        "amsgrad": False,     # Use AMSGrad variant of Adam
        
        # Learning rate settings
        "base_lr": 1e-4,      # Base learning rate
        "lr_min": 5e-6,       # Minimum learning rate for cosine annealing
        
        # Adam parameters
        "betas": [0.9, 0.999],    # Adam beta parameters
        "weight_decay": 1e-4,     # L2 regularization
        
        # Learning rate scheduler
        "scheduler": "cosine",    # Cosine annealing scheduler
    }
}


def get_config():
    """
    Get the configuration dictionary.
    
    Returns:
        dict: Complete configuration dictionary
    """
    return config


def update_config(**kwargs):
    """
    Update configuration with custom parameters.
    
    Args:
        **kwargs: Key-value pairs to update in config
        
    Example:
        update_config(batch_size=32, num_epochs=50)
    """
    config.update(kwargs)


def get_paths():
    """
    Get all path configurations.
    
    Returns:
        dict: Dictionary containing all path configurations
    """
    return {
        "database_path": config["database_path"],
        "feature_path": config["feature_path"],
        "model_path": config["model_path"],
    }


def validate_paths():
    """
    Validate that all required paths exist.
    
    Raises:
        FileNotFoundError: If any required path doesn't exist
    """
    paths_to_check = [
        ("database_path", config["database_path"]),
        ("feature_path", config["feature_path"]),
    ]
    
    for name, path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")
    
    print("✅ All paths validated successfully!")


if __name__ == "__main__":
    # Print configuration when run directly
    import json
    
    print("=" * 60)
    print("ASVspoof5 Training Configuration")
    print("=" * 60)
    
    # Validate paths
    try:
        validate_paths()
    except FileNotFoundError as e:
        print(f"⚠️  Path validation failed: {e}")
    
    # Pretty print configuration
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    print("\nProject paths:")
    print(f"Database: {config['database_path']}")
    print(f"Features: {config['feature_path']}")
    print(f"Model: {config['model_path']}")
