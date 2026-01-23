import torch # type: ignore
import torch.nn as nn # type: ignore
from .qegnn import QuantumEdgeUpdate

class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, n_qubits=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.pre_quantum = nn.Linear(hidden_dim, n_qubits) # project features to qubits
        self.coord_quantum = QuantumEdgeUpdate(n_qubits=n_qubits, n_layers=2) # apply quantum edge update
        
        self.post_quantum = nn.Linear(1, 1) # rescale to 1 dimension
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(), # swish function for smoother gradients
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h, pos, edge_index):
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True) + 1e-8
        raw_msg = torch.cat([h[row], h[col], radial], dim=1)
        msg = self.message_mlp(raw_msg)
        coord_weights = self.coord_mlp(msg) # predict a scalar weight for the vector (pos_i - pos_j)
        trans = (pos[row] - pos[col]) * coord_weights # update position
        num_nodes = h.size(0)
        agg_msg = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        agg_trans = torch.zeros(num_nodes, 3, device=h.device)
        
        agg_msg.index_add_(0, row, msg)
        agg_trans.index_add_(0, row, trans)
        pos_new = pos + agg_trans
        node_input = torch.cat([h, agg_msg], dim=1)
        h_new = h + self.node_mlp(node_input) # residual connection
        return h_new, pos_new