import torch # type: ignore
import os

weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)

# Load your PyTorch checkpoint
checkpoint_path = "best_model_classical.pt"  # Change this to your filename
state_dict = torch.load(checkpoint_path, map_location='cpu') # type: ignore

# 1. Inspect the keys to see what your model calls the weights
print("--- Checkpoint Keys Found ---")
for key, tensor in state_dict['model_state_dict'].items():
    print(f"ExtractingKey: {key:25} | Shape: {tensor.shape}")
    filename = os.path.join(weights_dir, f"{key}.bin")
    tensor.detach().cpu().float().numpy().flatten().tofile(filename)

