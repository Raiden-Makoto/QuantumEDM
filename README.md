# Quantum Equivariant Diffusion Model (QuantumEDM)

A unified implementation of Equivariant Graph Neural Networks (EGNN) for molecular diffusion models, supporting both **quantum** and **classical** architectures.

## Project Structure

```
QuantumEDM/
├── data/
│   ├── __init__.py
│   └── data_loader.py          # Unified data loader (supports train/val split)
├── layers/
│   ├── __init__.py
│   ├── egnn.py                 # Classical EGNN layer
│   ├── egg.py                  # Quantum EGNN layer (uses quantum circuits)
│   └── qegnn.py                # Quantum edge update module
├── models/
│   ├── __init__.py
│   └── denoising_model.py      # Unified DenoisingEGNN (supports both quantum/classical)
├── utils/
│   ├── __init__.py
│   ├── diffusion_scheduler.py  # Diffusion noise schedule
│   ├── sampling.py             # Molecule sampling/generation
│   └── plot.py                 # Visualization utilities
├── train.py                    # Unified training script
├── requirements.txt
└── README.md
```

## Features

- **Dual Architecture Support**: Train with either quantum or classical EGNN layers
- **Flexible Data Loading**: Support for train/validation splits and dataset sampling
- **Validation & Early Stopping**: Optional validation set with early stopping
- **Molecule Generation**: Sampling utilities for generating new molecules
- **Visualization**: Plotting tools for generated molecules

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

**Quantum Model (default):**
```bash
python train.py --epochs 25 --n-qubits 4 --num-layers 3
```

**Classical Model:**
```bash
python train.py --classical --epochs 25 --num-layers 4
```

**With Validation:**
```bash
python train.py --validation-split 0.2 --epochs 25
```

**Resume Training:**
```bash
python train.py --resume best_model.pt --epochs 10
```

**Use Subset of Data:**
```bash
python train.py --dataset-percent 0.01  # Use 1% of training data
```

### Sampling Molecules

```bash
python utils/sampling.py
```

### Visualizing Generated Molecules

```bash
python utils/plot.py
```

## Model Architecture

### Quantum EGNN Layer (`layers/egg.py`)
- Uses quantum circuits (PennyLane) for coordinate updates
- Projects features to qubits, applies quantum edge update, then rescales
- Configurable number of qubits (default: 4)

### Classical EGNN Layer (`layers/egnn.py`)
- Standard MLP-based coordinate update
- Includes timestep embedding for conditioning on noise level
- Simpler and faster than quantum version

## Configuration

Key parameters in `train.py`:
- `BATCH_SIZE`: Batch size (default: 64)
- `LR`: Learning rate (default: 1e-4)
- `TIMESTEPS`: Number of diffusion steps (default: 2500)
- `EPOCHS`: Number of training epochs (default: 25)
- `EARLY_STOPPING_PATIENCE`: Early stopping patience (default: 5)

## Notes

- The quantum model does not use timestep embeddings (simpler architecture)
- The classical model requires timestep embeddings and takes `t` as input
- Both models output predicted noise (displacement) for denoising
