import torch # type: ignore
import torch.nn as nn # type: ignore
import pennylane as qml # type: ignore

class ReuploadingQuantumAttention(nn.Module):
    def __init__(self, hidden_dim, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Project 128 features -> 4 qubits
        self.projector = nn.Linear(hidden_dim, n_qubits)
        
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        # ADJOINT DIFFERENTIATION (Fast Gradients)
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            # inputs: [n_qubits]
            # weights: [n_layers, n_qubits, 3]
            
            # --- THE RE-UPLOADING LOOP ---
            for i in range(n_layers):
                # 1. Re-encode Data (Angle Embedding)
                # We use RY rotations to map features to the Bloch sphere latitude
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
                
                # 2. Trainable Entanglement
                # We slice the weights manually for this layer
                qml.templates.StronglyEntanglingLayers(weights[i:i+1], wires=range(n_qubits))
            # -----------------------------
            
            # Measurement: PauliZ on Wire 0 -> Range [-1, 1]
            return qml.expval(qml.PauliZ(0))
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = qml.qnn.TorchLayer(circuit, weight_shapes)
        
    def forward(self, edge_attr):
        # 1. Project & Squash to [-pi, pi]
        q_in = torch.tanh(self.projector(edge_attr)) * torch.pi
        
        # 2. Run Deep Circuit
        q_out = self.qnode(q_in)
        
        # 3. Sigmoid -> [0, 1] Attention Score
        return torch.sigmoid(q_out).view(-1, 1)
