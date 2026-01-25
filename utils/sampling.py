import torch # type: ignore
import numpy as np # type: ignore
import sys # type: ignore
import os # type: ignore

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.denoising_model import DenoisingEGNN
from utils.diffusion_scheduler import DiffusionSchedule

@torch.no_grad()
def sample_molecule(model, scheduler, n_nodes, atom_types=None, n_dim=3, device='cpu'):
    """
    Sample a molecule using the trained diffusion model.
    
    Args:
        model: Trained DenoisingEGNN model
        scheduler: DiffusionSchedule instance (provides betas and alphas_cumprod)
        n_nodes: Number of atoms in the molecule
        atom_types: Tensor of atom type indices (default: all Carbon = 6)
        n_dim: Number of spatial dimensions (default: 3)
        device: Device to run on
    
    Returns:
        Generated coordinates as numpy array [n_nodes, n_dim]
    """
    model.eval()
    
    # 1. Start with pure Gaussian noise (random coordinates)
    # Shape: [N_Nodes, Dimensions] (not batched for EGNN)
    x = torch.randn(n_nodes, n_dim, device=device)
    
    # 2. Create atom types (default: all Carbon = 6)
    if atom_types is None:
        atom_types = torch.full((n_nodes,), 6, dtype=torch.long, device=device)  # All Carbons
    else:
        atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)
    
    # 3. Create fully connected edge_index for the graph
    # EGNN needs edge_index in [2, num_edges] format
    edge_list = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    
    # 4. Get betas and alphas_cumprod from scheduler
    betas = scheduler.betas
    alphas = scheduler.alphas
    alphas_cumprod = scheduler.alphas_cumprod
    
    # 5. Iterate backwards from T to 0
    n_timesteps = scheduler.timesteps
    for i in reversed(range(n_timesteps)):
        # Get alpha and alpha_hat for this timestep
        alpha = alphas[i]
        alpha_hat = alphas_cumprod[i]
        beta = betas[i]
        
        # Predict the noise using the EGNN model
        # Check if model accepts timestep argument (classical) or not (quantum)
        if hasattr(model, 'timestep_embedding'):
            # Classical model with timestep embedding
            t = torch.full((n_nodes,), i, device=device, dtype=torch.long)
            predicted_noise = model(atom_types, x, edge_index, t)
        else:
            # Quantum model without timestep embedding
            predicted_noise = model(atom_types, x, edge_index)
        
        # 6. The Denoising Step (Standard DDPM update)
        # x_{t-1} = (1/sqrt(alpha)) * (x_t - (beta/sqrt(1-alpha_hat)) * predicted_noise)
        mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_hat)) * predicted_noise)
        
        # Add noise back in (Langevin dynamics) except for the very last step
        if i > 0:
            sigma = torch.sqrt(beta)
            z = torch.randn_like(x)
            x = mean + sigma * z
        else:
            x = mean
        
        # Enforce Zero Center of Mass (CoM) constraint to prevent drift
        x = x - x.mean(dim=0, keepdim=True)

    return x.cpu().numpy()

if __name__ == "__main__":
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    CHECKPOINT_PATH = os.path.join(project_root, "best_model.pt")
    
    # Load checkpoint to determine model type
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    use_quantum = checkpoint.get('use_quantum', True)  # Default to quantum if not specified
    
    # Load model (match the type used during training)
    if use_quantum:
        model = DenoisingEGNN(num_atom_types=10, hidden_dim=128, num_layers=3, use_quantum=True, n_qubits=4).to(DEVICE)
    else:
        model = DenoisingEGNN(num_atom_types=10, hidden_dim=128, num_layers=4, use_quantum=False, timesteps=2500).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create scheduler (must match training timesteps)
    scheduler = DiffusionSchedule(timesteps=2500, device=DEVICE)
    
    # Sample a molecule
    generated_coords = sample_molecule(model, scheduler, n_nodes=5, device=DEVICE)
    print("Generated coordinates:")
    print(generated_coords)
