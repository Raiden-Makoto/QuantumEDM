import torch # type: ignore
from layers.rqa import ReuploadingQuantumAttention
import torch.nn as nn # type: ignore

class HybridDeepAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_qubits=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Message Features (Classical)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. QUANTUM DEEP ATTENTION
        # "Deep" because it uses re-uploading (3 layers)
        self.attn_quantum = ReuploadingQuantumAttention(hidden_dim, n_qubits=n_qubits, n_layers=3)
        
        # 3. Coordinate Update (Classical Muscle)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        
        # Handle empty edges case
        if edge_index.size(1) == 0:
            # No edges: just pass through with node MLP
            num_nodes = h.size(0)
            agg_msg = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
            pos_new = pos
            node_input = torch.cat([h, agg_msg], dim=1)
            h_new = h + self.node_mlp(node_input)
            return h_new, pos_new
        
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True) + 1e-8
        
        # Calculate Message
        raw_msg = torch.cat([h[row], h[col], radial], dim=1)
        msg = self.message_mlp(raw_msg)
        
        # Calculate Quantum Attention
        attn_score = self.attn_quantum(msg)
        
        # Gating: If quantum says 0, the atom ignores this neighbor
        weighted_msg = msg * attn_score
        
        # Update
        coord_weights = self.coord_mlp(weighted_msg)
        trans = coord_diff * coord_weights
        
        num_nodes = h.size(0)
        agg_msg = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        agg_trans = torch.zeros(num_nodes, 3, device=h.device)
        agg_msg.index_add_(0, row, weighted_msg)
        agg_trans.index_add_(0, row, trans)
        
        pos_new = pos + agg_trans
        node_input = torch.cat([h, agg_msg], dim=1)
        h_new = h + self.node_mlp(node_input)
        
        return h_new, pos_new