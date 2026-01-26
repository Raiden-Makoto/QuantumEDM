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
import matplotlib.pyplot as plt # type: ignore

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
        # Check if model is quantum (no timestep needed) or classical (needs timestep)
        if model.use_quantum:
            # Quantum model without timestep embedding
            predicted_noise = model(atom_types, x, edge_index)
        else:
            # Classical model with timestep embedding
            t = torch.full((n_nodes,), i, device=device, dtype=torch.long)
            predicted_noise = model(atom_types, x, edge_index, t)
        
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

def plot_molecule(coords, path: str='generated_molecule.png'):
    """
    Plot a 3D molecule structure.
    
    Args:
        coords: numpy array of shape [n_atoms, 3] with atomic coordinates
        path: Output path for the plot (default: 'generated_molecule.png')
    """
    full_path = os.path.join(project_root, path)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    num_atoms = coords.shape[0]
    
    # Plot atoms with labels to ensure all are visible
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=200, c='red', alpha=0.8, edgecolors='black', linewidths=1.5)
    
    # Label each atom with its index to verify all are plotted
    for i in range(num_atoms):
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], f'  {i}', fontsize=10, color='blue')
    
    # Draw "bonds" (simple distance heuristic) to see structure
    # If two atoms are closer than 1.7 Angstroms, draw a line
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 1.7:  # Typical bond length threshold ~1.5 - 1.7 Angstroms
                ax.plot([coords[i, 0], coords[j, 0]], 
                        [coords[i, 1], coords[j, 1]], 
                        [coords[i, 2], coords[j, 2]], c='black', linewidth=2)

    # Set equal aspect ratio and better viewing angle
    ax.set_xlabel('X (Angstroms)', fontsize=10)
    ax.set_ylabel('Y (Angstroms)', fontsize=10)
    ax.set_zlabel('Z (Angstroms)', fontsize=10)
    ax.set_title(f"Generated Molecule Structure ({num_atoms} atoms)", fontsize=12, fontweight='bold')
    
    # Set equal aspect ratio to prevent distortion
    max_range = np.array([coords[:, 0].max() - coords[:, 0].min(),
                          coords[:, 1].max() - coords[:, 1].min(),
                          coords[:, 2].max() - coords[:, 2].min()]).max() / 2.0
    mid_x = (coords[:, 0].max() + coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max() + coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max() + coords[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set a good viewing angle to see all atoms
    ax.view_init(elev=20, azim=45)
    
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return full_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample molecules using trained diffusion model')
    parser.add_argument(
        '--quantum', action='store_true',
        help='Use quantum model (default: auto-detect from available checkpoints)'
    )
    parser.add_argument(
        '--classical', action='store_true',
        help='Use classical model (default: auto-detect from available checkpoints)'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to specific checkpoint file (overrides auto-detection)'
    )
    parser.add_argument(
        '--n-nodes', type=int, default=5,
        help='Number of atoms in the molecule to generate (default: 5)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device to run on (default: cpu, quantum models require cpu)'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate and save a 3D plot of the molecule'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for the plot (only used with --plot, default: auto-generated based on model type)'
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.quantum:
        DEVICE = torch.device('cpu')  # Quantum models require CPU
        print("Quantum model specified: forcing CPU device")
    else:
        DEVICE = torch.device(args.device)
    
    # Determine checkpoint path
    checkpoint_path = None
    use_quantum = None
    
    if args.checkpoint:
        # User specified checkpoint path
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(project_root, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        print(f"Using specified checkpoint: {checkpoint_path}")
    elif args.quantum:
        # User explicitly requested quantum
        checkpoint_path = os.path.join(project_root, "best_model_quantum.pt")
        use_quantum = True
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Quantum checkpoint not found: {checkpoint_path}")
        print(f"Loading quantum model from {checkpoint_path}")
    elif args.classical:
        # User explicitly requested classical
        checkpoint_path = os.path.join(project_root, "best_model_classical.pt")
        use_quantum = False
        DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Classical checkpoint not found: {checkpoint_path}")
        print(f"Loading classical model from {checkpoint_path}")
    else:
        # Auto-detect: try quantum first, then classical, then legacy
        CHECKPOINT_PATH_QUANTUM = os.path.join(project_root, "best_model_quantum.pt")
        CHECKPOINT_PATH_CLASSICAL = os.path.join(project_root, "best_model_classical.pt")
        
        if os.path.exists(CHECKPOINT_PATH_QUANTUM):
            checkpoint_path = CHECKPOINT_PATH_QUANTUM
            use_quantum = True
            DEVICE = torch.device('cpu')
            print(f"Auto-detected: Loading quantum model from {checkpoint_path}")
        elif os.path.exists(CHECKPOINT_PATH_CLASSICAL):
            checkpoint_path = CHECKPOINT_PATH_CLASSICAL
            use_quantum = False
            DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"Auto-detected: Loading classical model from {checkpoint_path}")
        else:
            # Fallback to old checkpoint name
            old_checkpoint = os.path.join(project_root, "best_model.pt")
            if os.path.exists(old_checkpoint):
                checkpoint_path = old_checkpoint
                print(f"Auto-detected: Loading from legacy checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(
                    f"No checkpoint found. Expected one of:\n"
                    f"  - {CHECKPOINT_PATH_QUANTUM}\n"
                    f"  - {CHECKPOINT_PATH_CLASSICAL}\n"
                    f"  - {old_checkpoint}\n"
                    f"Or use --checkpoint to specify a path."
                )
    
    # Load checkpoint to determine model type if not already determined
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if use_quantum is None:
        use_quantum = checkpoint.get('use_quantum', True)  # Default to quantum if not specified
        if use_quantum:
            DEVICE = torch.device('cpu')  # Force CPU for quantum
    
    # Set default output filename based on model type if not specified
    if args.plot and args.output is None:
        model_type = 'quantum' if use_quantum else 'classical'
        args.output = f'generated_molecule_{model_type}.png'
    
    # Load model (simplified: always 4 layers, either all classical or 3 classical + 1 quantum)
    if use_quantum:
        model = DenoisingEGNN(
            num_atom_types=10, 
            hidden_dim=128, 
            use_quantum=True, 
            n_qubits=checkpoint.get('n_qubits', 4)
        ).to(DEVICE)
    else:
        model = DenoisingEGNN(
            num_atom_types=10, 
            hidden_dim=128, 
            use_quantum=False, 
            timesteps=2500
        ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create scheduler (must match training timesteps)
    scheduler = DiffusionSchedule(timesteps=2500, device=DEVICE)
    
    # Sample a molecule
    print(f"\nGenerating molecule with {args.n_nodes} atoms...")
    generated_coords = sample_molecule(model, scheduler, n_nodes=args.n_nodes, device=DEVICE)
    print("\nGenerated coordinates:")
    print(generated_coords)
    
    # Plot if requested
    if args.plot:
        plot_path = plot_molecule(generated_coords, path=args.output)
        print(f"\nPlot saved to: {plot_path}")
