from torch_geometric.datasets import QM9 # type: ignore
from torch_geometric.loader import DataLoader # type: ignore
import os
import torch

def get_data(batch_size=32, dataset_percent=None):
    # Load QM9 (automatically downloads to /tmp/qm9)
    # Filter: Only molecules with <= 9 heavy atoms (Standard "QM9-small" subset)
    dataset = QM9(root=os.path.join(os.getcwd(), 'data', 'qm9'))
    indices = [i for i, data in enumerate(dataset) if data.z.size(0) <= 18] # 18 includes Hydrogens
    dataset = dataset.index_select(indices)
    # 80-20 Split
    train_size = int(0.8 * len(dataset))
    train_set = dataset[:train_size]
    
    # Sample a percentage of training set if specified
    if dataset_percent is not None and dataset_percent < 1.0:
        torch.manual_seed(42)  # For reproducibility
        sample_size = max(1, int(dataset_percent * len(train_set)))
        sample_indices = torch.randperm(len(train_set))[:sample_size].tolist()
        train_set = train_set.index_select(sample_indices)
        print(f"Using {len(train_set)} samples ({100*len(train_set)/train_size:.1f}% of training set)")
    else:
        print(f"Using full training set: {len(train_set)} samples")
    
    # Num_workers=0 is often safer on M3 for debugging
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader

if __name__ == "__main__":
    loader = get_data()
    sample = next(iter(loader))
    print(f"Batch Loaded: {sample}")
    print(f"Positions (x,y,z): {sample.pos.shape}") # Should be [Total_Atoms, 3]
    print(f"Atom Types (One-Hot): {sample.z.shape}")