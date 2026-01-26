import torch # type: ignore
import torch.nn as nn # type: ignore
import math # type: ignore
from layers import ClassicalEGNNLayer, QuantumEGNNLayer, HybridDeepAttentionLayer # type: ignore

class DenoisingEGNN(nn.Module):
    """
    Denoising EGNN model with two modes:
    
    Mode 1 (use_quantum=False): All classical
        [Classical, Classical, Classical, Classical] - 4 layers
    
    Mode 2 (use_quantum=True): Sandwich architecture
        [Classical, Classical, Classical, Quantum] - 3 classical + 1 quantum
    
    Args:
        num_atom_types: Number of atom types (default: 10 for QM9)
        hidden_dim: Hidden dimension size (default: 128)
        use_quantum: If True, use 3 classical + 1 quantum; if False, use all classical (default: True)
        n_qubits: Number of qubits for quantum layer (default: 4, only used if use_quantum=True)
        timesteps: Number of diffusion timesteps (default: 2500, only used if use_quantum=False for timestep embedding)
    """
    def __init__(self, num_atom_types=10, hidden_dim=128, 
                 use_quantum=True, n_qubits=4, timesteps=2500):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_quantum = use_quantum
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        
        # Timestep embedding (only for classical model)
        if not use_quantum:
            self.timestep_embedding_dim = hidden_dim
            self.timestep_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Two modes: All classical OR 3 classical + 1 quantum (always 4 layers total)
        self.layers = nn.ModuleList()
        
        if use_quantum:
            # Mode 2: 3 classical + 1 quantum
            self.layers.append(ClassicalEGNNLayer(hidden_dim))
            self.layers.append(ClassicalEGNNLayer(hidden_dim))
            self.layers.append(ClassicalEGNNLayer(hidden_dim))
            self.layers.append(QuantumEGNNLayer(hidden_dim, n_qubits=n_qubits))
        else:
            # Mode 1: All classical (4 layers)
            self.layers.append(ClassicalEGNNLayer(hidden_dim))
            self.layers.append(ClassicalEGNNLayer(hidden_dim))
            self.layers.append(ClassicalEGNNLayer(hidden_dim))
            self.layers.append(ClassicalEGNNLayer(hidden_dim))
    
    def timestep_embedding(self, timesteps):
        """
        Create sinusoidal timestep embeddings (only for classical model).
        Args:
            timesteps: Tensor of shape [N] with timestep indices
        Returns:
            Tensor of shape [N, hidden_dim]
        """
        half_dim = self.timestep_embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Pad if hidden_dim is odd
        if self.timestep_embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        
        # Project through MLP
        emb = self.timestep_mlp(emb)
        return emb
        
    def forward(self, h, pos, edge_index, t=None):
        """
        Forward pass.
        
        Args:
            h: Atom type indices [N]
            pos: Positions [N, 3]
            edge_index: Edge connectivity [2, E]
            t: Timestep indices [N] - required for classical model, ignored for quantum model
        """
        # Embed atom types
        h = self.embedding(h)
        
        # Add timestep embedding for classical model
        if not self.use_quantum:
            if t is None:
                raise ValueError("Timestep tensor 't' is required for classical model")
            t_emb = self.timestep_embedding(t)
            h = h + t_emb  # Add timestep information to node features
        
        # Keep track of initial position to calculate noise displacement
        pos_input = pos.clone()
        
        # Pass through layers
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        
        # The network output is the predicted noise (displacement)
        epsilon_pred = pos - pos_input
        
        return epsilon_pred
