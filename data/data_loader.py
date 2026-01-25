from torch_geometric.datasets import QM9 # type: ignore
from torch_geometric.loader import DataLoader # type: ignore
import os
import torch
import random # type: ignore

def get_data(batch_size=32, dataset_percent=None, validation_split=None):
    """
    Load QM9 dataset with flexible configuration.
    
    Args:
        batch_size: Batch size for data loaders
        dataset_percent: If specified (0.0-1.0), use this percentage of training set
        validation_split: If specified (0.0-1.0), split dataset into train/val with this ratio
    
    Returns:
        If validation_split is None: single train_loader
        If validation_split is specified: (train_loader, val_loader)
    """
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Load QM9 (automatically downloads to data/qm9)
    # Filter: Only molecules with <= 9 heavy atoms (Standard "QM9-small" subset)
    dataset = QM9(root=os.path.join(os.getcwd(), 'data', 'qm9'))
    indices = [i for i, data in enumerate(dataset) if data.z.size(0) <= 18]  # 18 includes Hydrogens
    dataset = dataset.index_select(indices)
    
    if validation_split is not None:
        # Split into train and validation
        total_size = len(dataset)
        train_size = int((1 - validation_split) * total_size)
        
        all_indices = list(range(total_size))
        random.shuffle(all_indices)
        
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]
        
        train_dataset = dataset.index_select(train_indices)
        val_dataset = dataset.index_select(val_indices)
        
        # Sample a percentage of training set if specified
        if dataset_percent is not None and dataset_percent < 1.0:
            sample_size = max(1, int(dataset_percent * len(train_dataset)))
            sample_indices = torch.randperm(len(train_dataset))[:sample_size].tolist()
            train_dataset = train_dataset.index_select(sample_indices)
            print(f"Using {len(train_dataset)} training samples ({100*len(train_dataset)/train_size:.1f}% of training set)")
        
        print(f"Training: {len(train_dataset)} samples ({100*len(train_dataset)/total_size:.2f}% of filtered dataset)")
        print(f"Validation: {len(val_dataset)} samples ({100*len(val_dataset)/total_size:.2f}% of filtered dataset)")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader
    else:
        # Single train loader (no validation)
        train_size = int(0.8 * len(dataset))
        train_set = dataset[:train_size]
        
        # Sample a percentage of training set if specified
        if dataset_percent is not None and dataset_percent < 1.0:
            sample_size = max(1, int(dataset_percent * len(train_set)))
            sample_indices = torch.randperm(len(train_set))[:sample_size].tolist()
            train_set = train_set.index_select(sample_indices)
            print(f"Using {len(train_set)} samples ({100*len(train_set)/train_size:.1f}% of training set)")
        else:
            print(f"Using full training set: {len(train_set)} samples")
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        return train_loader

if __name__ == "__main__":
    # Test with validation split
    train_loader, val_loader = get_data(validation_split=0.2)
    train_sample = next(iter(train_loader))
    val_sample = next(iter(val_loader))
    print(f"Train Batch Loaded: {train_sample}")
    print(f"Train Positions (x,y,z): {train_sample.pos.shape}")
    print(f"Train Atom Types: {train_sample.z.shape}")
    print(f"Val Batch Loaded: {val_sample}")
    print(f"Val Positions (x,y,z): {val_sample.pos.shape}")
    print(f"Val Atom Types: {val_sample.z.shape}")
