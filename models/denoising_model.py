import torch # type: ignore
import torch.nn as nn # type: ignore
from layers.egg import EGNNLayer # type: ignore

class DenoisingEGNN(nn.Module):
    def __init__(self, num_atom_types=10, hidden_dim=128, num_layers=4, n_qubits=4):
        super().__init__()
        
        # 1. Atom Embedding: Maps integer (e.g., Carbon=6) to Vector [128]
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        
        # 2. Stack of Hybrid Quantum Layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, n_qubits=n_qubits) 
            for _ in range(num_layers)
        ])
        
    def forward(self, h, pos, edge_index):
        # Input: Atom types (h), Positions (pos), Graph connectivity (edge_index)
        
        # Embed atoms
        h = self.embedding(h)
        
        # Keep track of initial position to calculate noise displacement
        pos_input = pos.clone()
        
        # Pass through layers
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
            
        # The network output is the predicted noise (displacement)
        epsilon_pred = pos - pos_input
        
        return epsilon_pred