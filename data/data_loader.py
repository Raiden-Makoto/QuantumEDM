from torch_geometric.datasets import QM9 # type: ignore
from torch_geometric.loader import DataLoader # type: ignore
import os

def get_data(batch_size=32):
    # Load QM9 (automatically downloads to /tmp/qm9)
    # Filter: Only molecules with <= 9 heavy atoms (Standard "QM9-small" subset)
    dataset = QM9(root=os.path.join(os.getcwd(), 'data', 'qm9'))
    indices = [i for i, data in enumerate(dataset) if data.z.size(0) <= 18] # 18 includes Hydrogens
    dataset = dataset.index_select(indices)
    # 80-20 Split
    train_size = int(0.8 * len(dataset))
    train_set = dataset[:train_size]
    # Num_workers=0 is often safer on M3 for debugging
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader

if __name__ == "__main__":
    loader = get_data()
    sample = next(iter(loader))
    print(f"Batch Loaded: {sample}")
    print(f"Positions (x,y,z): {sample.pos.shape}") # Should be [Total_Atoms, 3]
    print(f"Atom Types (One-Hot): {sample.z.shape}")