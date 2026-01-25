import torch # type: ignore
import torch.nn.functional as F # type: ignore
from torch.optim import Adam # type: ignore
from data.data_loader import get_data
from models.denoising_model import DenoisingEGNN
from utils.diffusion_scheduler import DiffusionSchedule
from tqdm import tqdm # type: ignore
import argparse
import os
import matplotlib.pyplot as plt # type: ignore

# --- Configuration ---
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 25
TIMESTEPS = 2500  # Number of diffusion steps
EARLY_STOPPING_PATIENCE = 5  # Stop training after 5 epochs without improvement

def get_checkpoint_path(use_quantum):
    """Get checkpoint path based on model type."""
    return "best_model_quantum.pt" if use_quantum else "best_model_classical.pt"

def get_plot_path(use_quantum):
    """Get plot path based on model type."""
    return "training_loss_curve_quantum.png" if use_quantum else "training_loss_curve_classical.png"

def train(epochs=EPOCHS, resume_from=None, dataset_percent=0.05, 
          use_quantum=True, n_qubits=4, num_layers=3, validation_split=0.2):
    """
    Train DenoisingEGNN model.
    
    Args:
        epochs: Number of epochs to train
        resume_from: Path to checkpoint to resume from
        dataset_percent: Percentage of full dataset to sample (default: 0.05 = 5%)
        use_quantum: If True, use quantum layers; if False, use classical layers
        n_qubits: Number of qubits for quantum layers (only used if use_quantum=True)
        num_layers: Number of EGNN layers
        validation_split: Validation split ratio (default: 0.2 = 20% validation, 80% training). Set to None to disable.
    """
    # Force CPU device for quantum models (quantum circuits don't work well on MPS)
    if use_quantum:
        DEVICE = torch.device('cpu')
        print("Quantum model detected: forcing CPU device (quantum circuits require CPU)")
    else:
        DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Initialize checkpoint variable
    checkpoint = None
    
    # Handle resume logic: if no specific path provided, try to auto-detect
    if resume_from is not None:
        # If resume_from is the default const value, try to auto-detect based on model type
        if resume_from == "best_model.pt":  # Old default path
            # Try model-specific checkpoint first
            model_specific_path = get_checkpoint_path(use_quantum)
            if os.path.exists(model_specific_path):
                resume_from = model_specific_path
                print(f"Auto-detected checkpoint: {resume_from}")
            elif os.path.exists(resume_from):
                # Fall back to old checkpoint if it exists
                print(f"Using legacy checkpoint: {resume_from}")
            else:
                # Try the other model type's checkpoint
                other_path = get_checkpoint_path(not use_quantum)
                if os.path.exists(other_path):
                    print(f"Warning: Found {other_path} but model type is {'quantum' if use_quantum else 'classical'}")
                    print(f"Attempting to load anyway - will adjust model type if needed")
                    resume_from = other_path
                else:
                    resume_from = None
                    print(f"Warning: No checkpoint found. Starting from scratch.")
    
    # Check checkpoint first to determine if it was quantum (for resume)
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
        checkpoint_use_quantum = checkpoint.get('use_quantum', use_quantum)
        # Override use_quantum if checkpoint indicates different model type
        if checkpoint_use_quantum != use_quantum:
            print(f"Warning: Checkpoint was trained with {'quantum' if checkpoint_use_quantum else 'classical'} model, but --classical flag suggests otherwise.")
            print(f"Using checkpoint's model type: {'quantum' if checkpoint_use_quantum else 'classical'}")
            use_quantum = checkpoint_use_quantum
            # Re-set device based on checkpoint model type
            if use_quantum:
                DEVICE = torch.device('cpu')
                print("Quantum model detected: forcing CPU device (quantum circuits require CPU)")
            else:
                DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Determine checkpoint and plot paths based on final model type
    CHECKPOINT_PATH = get_checkpoint_path(use_quantum)
    PLOT_PATH = get_plot_path(use_quantum)
    
    print(f"Training on {DEVICE}...")
    print(f"Model type: {'Quantum' if use_quantum else 'Classical'}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Plot path: {PLOT_PATH}")
    
    # Get data loaders (default: 5% dataset sample with seed 42, 80/20 train/val split)
    # validation_split=None means disable validation, otherwise use the provided value (default: 0.2)
    if validation_split is None:
        result = get_data(batch_size=BATCH_SIZE, dataset_percent=dataset_percent, validation_split=0)
        train_loader = result
        val_loader = None
    else:
        train_loader, val_loader = get_data(
            batch_size=BATCH_SIZE, 
            dataset_percent=dataset_percent,
            validation_split=validation_split
        )
    
    # Create model
    # num_atom_types=10 covers QM9 (H=1, C=6, N=7, O=8, F=9)
    if use_quantum:
        model = DenoisingEGNN(
            num_atom_types=10, 
            hidden_dim=128, 
            num_layers=num_layers, 
            use_quantum=True,
            n_qubits=n_qubits
        ).to(DEVICE)
    else:
        model = DenoisingEGNN(
            num_atom_types=10, 
            hidden_dim=128, 
            num_layers=num_layers, 
            use_quantum=False,
            timesteps=TIMESTEPS
        ).to(DEVICE)
    
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = DiffusionSchedule(timesteps=TIMESTEPS, device=DEVICE)
    
    # Track best loss for saving best model
    best_loss = float('inf')
    best_epoch = 0
    start_epoch = 0
    patience_counter = 0  # Track epochs without improvement for early stopping
    
    # Track loss history for plotting
    train_loss_history = []
    val_loss_history = [] if val_loader else []
    epoch_history = []
    
    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}...")
        # Checkpoint already loaded above to check model type
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        # When resuming, epochs means "train for this many more epochs"
        total_epochs = start_epoch + epochs
        print(f"Resumed from epoch {start_epoch-1}, will train until epoch {total_epochs-1}, best loss: {best_loss:.4f}")
    elif resume_from:
        print(f"Warning: Checkpoint {resume_from} not found. Starting from scratch.")
        total_epochs = epochs
    else:
        total_epochs = epochs
    
    model.train()
    
    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, total_epochs), desc="Training", unit="epoch", initial=start_epoch, total=total_epochs)
    
    for epoch in epoch_pbar:
        total_loss = 0
        num_batches = len(train_loader)
        
        # Inner progress bar for batches
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False, unit="batch")
        
        for batch in batch_pbar:
            batch = batch.to(DEVICE)
            
            # 1. Prepare Inputs
            # QM9 gives us batch.z (atom types) and batch.pos (coordinates)
            x_0 = batch.pos
            
            # Sample random timestep for each molecule in the batch
            # batch.batch is a vector [0, 0, 0, 1, 1, 2, ...] indicating which graph a node belongs to
            num_graphs = batch.batch.max().item() + 1
            t_per_graph = torch.randint(0, TIMESTEPS, (num_graphs,), device=DEVICE).long()
            
            # Map graph-level t to node-level t
            t_per_node = t_per_graph[batch.batch]
            
            # 3. Add Noise (Forward Process)
            # We need to reshape sched vars to align with nodes
            sqrt_alpha = scheduler.alphas_cumprod[t_per_node].sqrt().unsqueeze(-1)
            sqrt_one_minus = (1 - scheduler.alphas_cumprod[t_per_node]).sqrt().unsqueeze(-1)
            
            epsilon = torch.randn_like(x_0)
            x_t = sqrt_alpha * x_0 + sqrt_one_minus * epsilon
            
            # 4. Predict Noise (Reverse Process)
            optimizer.zero_grad()
            if use_quantum:
                epsilon_pred = model(batch.z, x_t, batch.edge_index)
            else:
                epsilon_pred = model(batch.z, x_t, batch.edge_index, t_per_node)
            
            # 5. Loss & Backprop
            loss = F.mse_loss(epsilon_pred, epsilon)
            loss.backward()
            
            # --- NEW: Clip Gradients to prevent explosions ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / num_batches
        
        # Validation loop (if validation split is provided)
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(DEVICE)
                    x_0 = val_batch.pos
                    
                    # Sample random timestep for each molecule in the batch
                    num_graphs = val_batch.batch.max().item() + 1
                    t_per_graph = torch.randint(0, TIMESTEPS, (num_graphs,), device=DEVICE).long()
                    t_per_node = t_per_graph[val_batch.batch]
                    
                    # Add noise
                    sqrt_alpha = scheduler.alphas_cumprod[t_per_node].sqrt().unsqueeze(-1)
                    sqrt_one_minus = (1 - scheduler.alphas_cumprod[t_per_node]).sqrt().unsqueeze(-1)
                    epsilon = torch.randn_like(x_0)
                    x_t = sqrt_alpha * x_0 + sqrt_one_minus * epsilon
                    
                    # Predict noise
                    if use_quantum:
                        epsilon_pred = model(val_batch.z, x_t, val_batch.edge_index)
                    else:
                        epsilon_pred = model(val_batch.z, x_t, val_batch.edge_index, t_per_node)
                    loss = F.mse_loss(epsilon_pred, epsilon)
                    val_loss += loss.item()
                    val_batches += 1
            
            model.train()
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            val_loss_history.append(avg_val_loss)
        
        # Track loss history
        train_loss_history.append(avg_train_loss)
        epoch_history.append(epoch + 1)  # 1-indexed for display
        
        # Determine which loss to use for early stopping and best model
        loss_for_early_stopping = avg_val_loss if avg_val_loss is not None else avg_train_loss
        
        # Update epoch progress bar with average loss and best loss
        if avg_val_loss is not None:
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'best_val': f'{best_loss:.4f}',
                'patience': f'{patience_counter}/{EARLY_STOPPING_PATIENCE}'
            })
        else:
            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_train_loss:.4f}',
                'best_loss': f'{best_loss:.4f}',
                'patience': f'{patience_counter}/{EARLY_STOPPING_PATIENCE}'
            })
        
        # Save best model if loss improved
        if loss_for_early_stopping < best_loss:
            best_loss = loss_for_early_stopping
            best_epoch = epoch + 1  # 1-indexed for display
            patience_counter = 0  # Reset patience counter on improvement
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'use_quantum': use_quantum,
                'n_qubits': n_qubits if use_quantum else None,
                'num_layers': num_layers,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            if avg_val_loss is not None:
                tqdm.write(f"Epoch {epoch+1}/{total_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | ✓ New best model saved!")
            else:
                tqdm.write(f"Epoch {epoch+1}/{total_epochs} complete. Avg Loss: {avg_train_loss:.4f} | ✓ New best model saved!")
        else:
            patience_counter += 1  # Increment patience counter when no improvement
            if avg_val_loss is not None:
                tqdm.write(f"Epoch {epoch+1}/{total_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Best Val: {best_loss:.4f} | Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            else:
                tqdm.write(f"Epoch {epoch+1}/{total_epochs} complete. Avg Loss: {avg_train_loss:.4f} | Best: {best_loss:.4f} | Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        # Early stopping: break if patience exhausted
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            tqdm.write(f"\nEarly stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            tqdm.write(f"Best loss achieved: {best_loss:.4f} at epoch {best_epoch}")
            break
    
    # Plot loss curve at the end of training
    if len(train_loss_history) > 0:
        # Filter out loss values > 1 to keep scale consistent
        filtered_epochs = []
        filtered_train_losses = []
        filtered_val_losses = []
        for epoch, train_loss in zip(epoch_history, train_loss_history):
            if train_loss <= 1.0:
                filtered_epochs.append(epoch)
                filtered_train_losses.append(train_loss)
                if val_loader:
                    val_loss = val_loss_history[epoch - epoch_history[0]]
                    if val_loss <= 1.0:
                        filtered_val_losses.append(val_loss)
                    else:
                        filtered_val_losses.append(None)
        
        if len(filtered_train_losses) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(filtered_epochs, filtered_train_losses, 'b-', linewidth=2, label='Training Loss')
            
            if val_loader and len(filtered_val_losses) > 0:
                valid_val_losses = [v for v in filtered_val_losses if v is not None]
                valid_epochs = [e for e, v in zip(filtered_epochs, filtered_val_losses) if v is not None]
                if len(valid_val_losses) > 0:
                    plt.plot(valid_epochs, valid_val_losses, 'r-', linewidth=2, label='Validation Loss')
            
            # Mark best loss with a star
            if best_epoch > 0 and best_epoch in filtered_epochs:
                best_idx = filtered_epochs.index(best_epoch)
                best_loss_value = filtered_train_losses[best_idx]
                plt.plot(filtered_epochs[best_idx], best_loss_value, 'r*', 
                        markersize=15, label=f'Best Loss: {best_loss_value:.4f} (Epoch {best_epoch})')
            
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            title = 'Training Loss Curve' if not val_loader else 'Training and Validation Loss Curves'
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            # Save plot with model-specific name
            plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
            tqdm.write(f"\nLoss curve saved to {PLOT_PATH}")
            plt.close()
        else:
            tqdm.write("\nNo valid loss values (<= 1.0) to plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EGNN Diffusion Model (Quantum or Classical)')
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Number of epochs to train (default: {EPOCHS})'
    )
    parser.add_argument(
        '--resume', type=str, nargs='?', const="best_model.pt", default=None,
        help='Resume training from checkpoint. If no path provided, auto-detects based on model type (quantum: best_model_quantum.pt, classical: best_model_classical.pt). Can also specify explicit path. (default: None)'
    )
    parser.add_argument(
        '--dataset-percent', type=float, default=0.05,
        help='Percentage of full dataset to sample BEFORE train/val split (default: 0.05 = 5%%, with seed 42)'
    )
    parser.add_argument(
        '--classical', action='store_true',
        help='Use classical EGNN layers instead of quantum (default: False, uses quantum)'
    )
    parser.add_argument(
        '--n-qubits', type=int, default=4,
        help='Number of qubits for quantum layers (default: 4, only used if --classical is not set)'
    )
    parser.add_argument(
        '--num-layers', type=int, default=3,
        help='Number of EGNN layers (default: 3)'
    )
    parser.add_argument(
        '--validation-split', type=float, default=0.2,
        help='Validation split ratio (default: 0.2 = 20%% validation, 80%% training). Set to 0 to disable validation. Uses seed 42 for reproducibility.'
    )
    parser.add_argument(
        '--no-validation', action='store_true',
        help='Disable validation split (equivalent to --validation-split 0)'
    )
    args = parser.parse_args()
    
    # Handle --no-validation flag or validation_split=0
    if args.no_validation or args.validation_split == 0:
        validation_split = None
    else:
        validation_split = args.validation_split
    
    train(
        epochs=args.epochs, 
        resume_from=args.resume, 
        dataset_percent=args.dataset_percent,
        use_quantum=not args.classical,
        n_qubits=args.n_qubits,
        num_layers=args.num_layers,
        validation_split=validation_split
    )
