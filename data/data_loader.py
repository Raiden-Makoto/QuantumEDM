from torch_geometric.datasets import QM9 # type: ignore
from torch_geometric.loader import DataLoader # type: ignore
import os
import torch
import random # type: ignore

def get_data(batch_size=32, dataset_percent=0.05, validation_split=0.2):
    """
    Load QM9 dataset with flexible configuration.
    
    Args:
        batch_size: Batch size for data loaders
        dataset_percent: Percentage of full dataset to sample BEFORE train/val split (default: 0.05 = 5%)
        validation_split: Validation split ratio (default: 0.2 = 20% validation, 80% training). 
                          Set to 0 to disable validation and return only train_loader.
    
    Returns:
        If validation_split == 0: single train_loader
        Otherwise: (train_loader, val_loader)
    """
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Load QM9 (automatically downloads to data/qm9)
    # Filter: Only molecules with <= 9 heavy atoms (Standard "QM9-small" subset)
    dataset = QM9(root=os.path.join(os.getcwd(), 'data', 'qm9'))
    indices = [i for i, data in enumerate(dataset) if data.z.size(0) <= 18]  # 18 includes Hydrogens
    dataset = dataset.index_select(indices)
    
    total_size = len(dataset)
    
    # Sample percentage of dataset (default: 5% with seed 42)
    if dataset_percent is not None and 0 < dataset_percent < 1.0:
        sample_size = max(1, int(dataset_percent * total_size))
        all_indices = list(range(total_size))
        random.shuffle(all_indices)
        sampled_indices = all_indices[:sample_size]
        dataset = dataset.index_select(sampled_indices)
        print(f"Sampled {len(dataset)} samples ({100*dataset_percent:.1f}% of filtered dataset, seed=42)")
    elif dataset_percent == 1.0:
        print(f"Using full filtered dataset: {len(dataset)} samples")
    
    # Split into train/val if validation_split > 0
    if validation_split is not None and validation_split > 0:
        # Split into train and validation
        dataset_size = len(dataset)
        train_size = int((1 - validation_split) * dataset_size)
        
        all_indices = list(range(dataset_size))
        random.shuffle(all_indices)
        
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]
        
        train_dataset = dataset.index_select(train_indices)
        val_dataset = dataset.index_select(val_indices)
        
        print(f"Training: {len(train_dataset)} samples ({100*len(train_dataset)/dataset_size:.1f}% of sampled dataset, seed=42)")
        print(f"Validation: {len(val_dataset)} samples ({100*len(val_dataset)/dataset_size:.1f}% of sampled dataset, seed=42)")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader
    else:
        # Single train loader (no validation) - when validation_split=0 or None
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"Using full sampled dataset: {len(dataset)} samples (no validation split)")
        return train_loader

if __name__ == "__main__":
    # Test with default settings: 5% dataset sample, 80/20 train/val split
    train_loader, val_loader = get_data(dataset_percent=0.05, validation_split=0.2)
    train_sample = next(iter(train_loader))
    val_sample = next(iter(val_loader))
    print(f"Train Batch Loaded: {train_sample}")
    print(f"Train Positions (x,y,z): {train_sample.pos.shape}")
    print(f"Train Atom Types: {train_sample.z.shape}")
    print(f"Val Batch Loaded: {val_sample}")
    print(f"Val Positions (x,y,z): {val_sample.pos.shape}")
    print(f"Val Atom Types: {val_sample.z.shape}")
