import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message Block: computes interaction strength between atoms
        # Input: [h_i, h_j, distance_sq]
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate Block: predicts how much to move atoms
        # Input: [message] -> Output: scalar weight
        # Fair Classical: 128 features -> 4 Hidden Neurons -> 1 Scalar Output
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.SiLU(),
            nn.Linear(4, 1) # Outputs a weight, not a vector!
        )
        
        # Node Update Block: updates atom features
        # Input: [h_i, aggregated_messages]
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        
        # Calculate Distances (Invariant)
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True) + 1e-8
        
        # Message Passing
        # Concatenate: Atom1 features + Atom2 features + Distance
        raw_msg = torch.cat([h[row], h[col], radial], dim=1)
        msg = self.message_mlp(raw_msg)
        
        # Coordinate Update (Equivariant)
        # We predict a SCALAR weight for the vector (pos_i - pos_j)
        # This is the "magic" that preserves rotation symmetry
        coord_weights = self.coord_mlp(msg)
        trans = (pos[row] - pos[col]) * coord_weights
        
        # Aggregation: Sum effects of all neighbors
        num_nodes = h.size(0)
        agg_msg = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        agg_trans = torch.zeros(num_nodes, 3, device=h.device)
        
        agg_msg.index_add_(0, row, msg)
        agg_trans.index_add_(0, row, trans)
        
        # Update Position
        pos_new = pos + agg_trans
        
        # Feature Update
        # Concatenate old features + aggregated messages
        node_input = torch.cat([h, agg_msg], dim=1)
        h_new = h + self.node_mlp(node_input) # Residual connection
        
        return h_new, pos_new
