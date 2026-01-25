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
from models.denoising_model import DenoisingEGNN

def plot_molecule(coords, path: str='generated_molecule.png'):
    # Get project root (one directory up from utils/)
    # project_root is already defined at module level
    full_path = os.path.join(project_root, path)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=100, c='red', alpha=0.6)
    
    # Draw "bonds" (simple distance heuristic) to see structure
    # If two atoms are closer than 1.5 units, draw a line
    num_atoms = coords.shape[0]
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 1.7: # Typical bond length threshold ~1.5 - 1.7 Angstroms
                ax.plot([coords[i, 0], coords[j, 0]], 
                        [coords[i, 1], coords[j, 1]], 
                        [coords[i, 2], coords[j, 2]], c='black')

    ax.set_title("Generated Molecule Structure")
    plt.savefig(full_path)
    plt.close()

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
    plot_molecule(generated_coords)
