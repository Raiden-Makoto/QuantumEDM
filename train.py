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
DEVICE = 'cpu' #torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 25
TIMESTEPS = 2500  # Number of diffusion steps
CHECKPOINT_PATH = "best_model.pt"
EARLY_STOPPING_PATIENCE = 5  # Stop training after 5 epochs without improvement

def train(epochs=EPOCHS, resume_from=None, dataset_percent=None):
    print(f"Training on {DEVICE}...")
    
    loader = get_data(batch_size=BATCH_SIZE, dataset_percent=dataset_percent)
    
    # num_atom_types=10 covers QM9 (H=1, C=6, N=7, O=8, F=9)
    # Using quantum version with n_qubits=4
    model = DenoisingEGNN(num_atom_types=10, hidden_dim=128, num_layers=3, n_qubits=4).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = DiffusionSchedule(timesteps=TIMESTEPS, device=DEVICE)
    
    # Track best loss for saving best model
    best_loss = float('inf')
    best_epoch = 0
    start_epoch = 0
    patience_counter = 0  # Track epochs without improvement for early stopping
    
    # Track loss history for plotting
    loss_history = []
    epoch_history = []
    
    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=DEVICE, weights_only=False)
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
        num_batches = len(loader)
        
        # Inner progress bar for batches
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False, unit="batch")
        
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
            epsilon_pred = model(batch.z, x_t, batch.edge_index)
            
            # 5. Loss & Backprop
            loss = F.mse_loss(epsilon_pred, epsilon)
            loss.backward()
            
            # --- NEW: Clip Gradients to prevent explosions ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        
        # Track loss history
        loss_history.append(avg_loss)
        epoch_history.append(epoch + 1)  # 1-indexed for display
        
        # Update epoch progress bar with average loss and best loss
        epoch_pbar.set_postfix({
            'avg_loss': f'{avg_loss:.4f}',
            'best_loss': f'{best_loss:.4f}',
            'patience': f'{patience_counter}/{EARLY_STOPPING_PATIENCE}'
        })
        
        # Save best model if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1  # 1-indexed for display
            patience_counter = 0  # Reset patience counter on improvement
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            tqdm.write(f"Epoch {epoch+1}/{total_epochs} complete. Avg Loss: {avg_loss:.4f} | ✓ New best model saved!")
        else:
            patience_counter += 1  # Increment patience counter when no improvement
            tqdm.write(f"Epoch {epoch+1}/{total_epochs} complete. Avg Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        # Early stopping: break if patience exhausted
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            tqdm.write(f"\nEarly stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            tqdm.write(f"Best loss achieved: {best_loss:.4f} at epoch {best_epoch}")
            break
    
    # Plot loss curve at the end of training
    if len(loss_history) > 0:
        # Filter out loss values > 1 to keep scale consistent
        filtered_epochs = []
        filtered_losses = []
        for epoch, loss in zip(epoch_history, loss_history):
            if loss <= 1.0:
                filtered_epochs.append(epoch)
                filtered_losses.append(loss)
        
        if len(filtered_losses) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(filtered_epochs, filtered_losses, 'b-', linewidth=2, label='Training Loss')
            
            # Mark best loss with a star (only if it's <= 1.0)
            if best_epoch > 0 and best_epoch in filtered_epochs:
                best_idx = filtered_epochs.index(best_epoch)
                best_loss_value = filtered_losses[best_idx]
                plt.plot(filtered_epochs[best_idx], best_loss_value, 'r*', 
                        markersize=15, label=f'Best Loss: {best_loss_value:.4f} (Epoch {best_epoch})')
            elif len(filtered_losses) > 0:
                # If best_epoch not in filtered history, mark the minimum in filtered session
                best_idx = filtered_losses.index(min(filtered_losses))
                best_loss_value = filtered_losses[best_idx]
                plt.plot(filtered_epochs[best_idx], best_loss_value, 'r*', 
                        markersize=15, label=f'Best Loss (this session): {best_loss_value:.4f} (Epoch {filtered_epochs[best_idx]})')
            
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            # Save plot
            plot_path = 'training_loss_curve.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            tqdm.write(f"\nLoss curve saved to {plot_path}")
            plt.close()
        else:
            tqdm.write("\nNo valid loss values (<= 1.0) to plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Quantum EGNN Diffusion Model')
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Number of epochs to train (default: {EPOCHS})'
    )
    parser.add_argument(
        '--resume', type=str, nargs='?', const=CHECKPOINT_PATH, default=None,
        help=f'Resume training from checkpoint. If no path provided, uses {CHECKPOINT_PATH} (default: None)'
    )
    parser.add_argument(
        '--dataset-percent', type=float, default=None,
        help='Percentage of training dataset to use (0.0-1.0). If not specified, uses full dataset. Example: 0.01 for 1%%'
    )
    args = parser.parse_args()
    
    train(epochs=args.epochs, resume_from=args.resume, dataset_percent=args.dataset_percent)
