import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import os # type: ignore
import sys # type: ignore
import torch # type: ignore

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.diffusion_scheduler import DiffusionSchedule
from utils.sampling import sample_molecule
from models import ClassicalEDM, QuantumEDM

def plot_molecule(coords, path: str='generated_molecule.png'):
    """
    Plot a 3D molecule structure.
    
    Args:
        coords: numpy array of shape [n_atoms, 3] with atomic coordinates
        path: Output path for the plot (default: 'generated_molecule.png')
    
    Returns:
        Full path to the saved plot file
    """
    # Get project root (one directory up from utils/)
    # project_root is already defined at module level
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
    
    print(f"Plotted {num_atoms} atoms")
    return full_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample and plot molecules using trained diffusion model')
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
        '--output', type=str, default=None,
        help='Output path for the plot (default: auto-generated based on model type)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device to run on (default: cpu, quantum models require cpu)'
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
    if args.output is None:
        model_type = 'quantum' if use_quantum else 'classical'
        args.output = f'generated_molecule_{model_type}.png'
    
    # Load model
    if use_quantum:
        model = QuantumEDM(
            num_atom_types=10, 
            hidden_dim=128, 
            num_layers=4,
            n_qubits=checkpoint.get('n_qubits', 4),
            timesteps=2500
        ).to(DEVICE)
    else:
        model = ClassicalEDM(
            num_atom_types=10, 
            hidden_dim=128, 
            num_layers=4,
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
    
    # Plot the molecule
    plot_path = plot_molecule(generated_coords, path=args.output)
    print(f"\nPlot saved to: {plot_path}")
