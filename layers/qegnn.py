import torch # type: ignore
import torch.nn as nn # type: ignore
import pennylane as qml # type: ignore

class QuantumEdgeUpdate(nn.Module):
    def __init__(self, n_qubits=5, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = qml.qnn.TorchLayer(circuit, weight_shapes)
        
    def forward(self, x):
        return self.qnode(x)

if __name__ == "__main__":
    layer = QuantumEdgeUpdate()
    noise = torch.randn(5)
    print(layer(noise))