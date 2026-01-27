import torch # type: ignore
import torch.nn as nn # type: ignore
import math # type: ignore
from layers import ClassicalEGNNLayer, HybridDeepAttentionLayer # type: ignore

class QuantumEDM(nn.Module):
    def __init__(self, num_atom_types=10, hidden_dim=128, num_layers=4, n_qubits=4, timesteps=2500):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        
        # Timestep embedding for quantum model
        self.timestep_embedding_dim = hidden_dim
        self.timestep_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # --- THE SANDWICH STACK ---
        self.layers = nn.ModuleList()
        
        # Layers 1-3: CLASSICAL (Fast Feature Extraction)
        # These build up complex features quickly.
        for _ in range(num_layers - 1):
            self.layers.append(ClassicalEGNNLayer(hidden_dim))
            
        # Layer 4: QUANTUM (The Bottleneck Brain)
        # Uses Data Re-uploading AND Attention to make the final decision.
        self.layers.append(HybridDeepAttentionLayer(hidden_dim, n_qubits=n_qubits))
    
    def timestep_embedding(self, timesteps):
        """
        Create sinusoidal timestep embeddings.
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
        
    def forward(self, h, pos, edge_index, t):
        """
        Forward pass.
        
        Args:
            h: Atom type indices [N]
            pos: Positions [N, 3]
            edge_index: Edge connectivity [2, E]
            t: Timestep indices [N] - required for quantum model
        Returns:
            epsilon_pred: Predicted noise (displacement) [N, 3]
        """
        # Embed atom types
        h = self.embedding(h)
        
        # Add timestep embedding
        if t is None:
            raise ValueError("Timestep tensor 't' is required for quantum model")
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