import time
from pennylane import numpy as np
import pennylane as qml
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. QUANTUM SUPPORT VECTOR MACHINE (QSVM)
# ==========================================

class QSVMClassifier:
    """
    Quantum Support Vector Machine using a Quantum Kernel.
    Constructs a quantum kernel matrix using PennyLane and trains a classical SVM.
    """
    def __init__(self, num_qubits=4, C=1.0, random_state=42):
        self.num_qubits = num_qubits
        self.C = C
        self.random_state = random_state
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.svm = SVC(kernel="precomputed", C=self.C, probability=True, random_state=self.random_state)
        
        self.train_features = None
        self.train_time = 0.0
        self.infer_time = 0.0
        
        # Build the kernel QNode
        @qml.qnode(self.device)
        def _kernel(x1, x2):
            # Feature Map: Angle embedding + simple entangling layer
            for i in range(self.num_qubits):
                qml.RY(x1[i], wires=i)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                
            # Adjoint of Feature Map for x2
            for i in reversed(range(self.num_qubits - 1)):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.num_qubits):
                qml.RY(-x2[i], wires=i)
                
            return qml.probs(wires=range(self.num_qubits))
            
        self._kernel_qnode = _kernel

    def _compute_kernel_matrix(self, A, B):
        """Computes the pairwise quantum kernel matrix."""
        matrix = np.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                # Probability of measuring the |0...0> state
                probs = self._kernel_qnode(A[i], B[j])
                matrix[i, j] = probs[0]
        return matrix

    def fit(self, X, y):
        start_time = time.perf_counter()
        self.train_features = X
        kernel_train = self._compute_kernel_matrix(X, X)
        self.svm.fit(kernel_train, y)
        self.train_time = time.perf_counter() - start_time
        return self

    def predict(self, X):
        start_time = time.perf_counter()
        kernel_test = self._compute_kernel_matrix(X, self.train_features)
        preds = self.svm.predict(kernel_test)
        self.infer_time = time.perf_counter() - start_time
        return preds

    def predict_proba(self, X):
        kernel_test = self._compute_kernel_matrix(X, self.train_features)
        return self.svm.predict_proba(kernel_test)

    def get_resource_metrics(self):
        # QSVM has N qubits. For kernel circuit, we execute U(x1) and U^\dagger(x2).
        # U(x1) has N RY gates and N-1 CNOTs. U^\dagger(x2) has N RY gates and N-1 CNOTs.
        # Total: 2N single-qubit gates, 2(N-1) two-qubit gates.
        return {
            "qubit_count": self.num_qubits,
            "circuit_depth": 2 + 2 * (self.num_qubits - 1),
            "gate_count": 2 * self.num_qubits + 2 * (self.num_qubits - 1),
            "trainable_parameters": 0  # QSVM is non-parametric (kernel based)
        }


# ==========================================
# 2. VARIATIONAL QUANTUM CLASSIFIER (VQC)
# ==========================================

class VQCClassifier:
    """
    Variational Quantum Classifier.
    Uses Angle Embedding and Strongly Entangling Layers to output class predictions.
    """
    def __init__(self, num_qubits=4, n_layers=3, epochs=20, lr=0.05, callback=None):
        self.num_qubits = num_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.callback = callback
        
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.weights = None
        self.bias = None
        self.train_time = 0.0
        self.infer_time = 0.0
        
        # QNode outputting expectation value of Z on qubit 0
        @qml.qnode(self.device, interface="autograd")
        def _circuit(weights, features):
            # Encoding
            qml.AngleEmbedding(features, wires=range(self.num_qubits), rotation='Y')
            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.expval(qml.PauliZ(0))
            
        self._circuit_qnode = _circuit

    def _vqc_predict(self, weights, bias, features):
        """Computes logits: expectation value + bias."""
        return self._circuit_qnode(weights, features) + bias

    def _cost(self, weights, bias, X, y):
        """MSE loss function (labels converted to {-1, 1})."""
        predictions = np.array([self._vqc_predict(weights, bias, x) for x in X])
        # Convert labels from {0, 1} to {-1, 1}
        y_transformed = y * 2.0 - 1.0
        return np.mean((predictions - y_transformed) ** 2)

    def fit(self, X, y):
        start_time = time.perf_counter()
        
        # Initialize weights for StronglyEntanglingLayers: (n_layers, num_qubits, 3)
        np.random.seed(42)
        raw_weights = 0.01 * np.random.randn(self.n_layers, self.num_qubits, 3)
        self.weights = np.tensor(raw_weights, requires_grad=True)
        self.bias = np.tensor([0.0], requires_grad=True)
        
        opt = qml.AdamOptimizer(stepsize=self.lr)
        
        weights = self.weights
        bias = self.bias
        
        for epoch in range(self.epochs):
            weights, bias, _, _ = opt.step(self._cost, weights, bias, X, y)
            loss = self._cost(weights, bias, X, y)
            
            if self.callback:
                self.callback(epoch + 1, self.epochs, float(loss))
                
        self.weights = weights
        self.bias = bias
        self.train_time = time.perf_counter() - start_time
        return self

    def predict_proba(self, X):
        raw_preds = np.array([self._vqc_predict(self.weights, self.bias, x) for x in X])
        # Map expectation value [-1, 1] to probability [0, 1] using sigmoid
        prob_1 = 1.0 / (1.0 + np.exp(-2.0 * raw_preds))
        prob_0 = 1.0 - prob_1
        return np.column_stack((prob_0, prob_1))

    def predict(self, X):
        start_time = time.perf_counter()
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        self.infer_time = time.perf_counter() - start_time
        return preds

    def get_resource_metrics(self):
        # VQC has N qubits.
        # Feature map: N RY gates (depth 1)
        # StronglyEntanglingLayers: each layer has N rotations (3 angles per qubit) and N CNOTs.
        # Depth is roughly 1 + L * (1 + N)
        return {
            "qubit_count": self.num_qubits,
            "circuit_depth": 1 + self.n_layers * (1 + self.num_qubits),
            "gate_count": self.num_qubits + self.n_layers * (self.num_qubits * 4), # 3 rotation gates + 1 CNOT per qubit per layer
            "trainable_parameters": self.n_layers * self.num_qubits * 3 + 1  # Weights + 1 bias
        }


# ==========================================
# 3. QUANTUM NEURAL NETWORK (QNN)
# ==========================================

class TorchQNN(nn.Module):
    """PyTorch-integrated Hybrid Classical-Quantum Neural Network."""
    def __init__(self, input_dim, num_qubits, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.n_layers = n_layers
        
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        
        # QNode returning expectation of PauliZ for all qubits
        @qml.qnode(self.device, interface="torch")
        def _qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            
        weight_shapes = {"weights": (self.n_layers, self.num_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(_qnode, weight_shapes)
        
        # Classical pre-processing and post-processing layers
        self.fc_in = nn.Linear(input_dim, self.num_qubits)
        self.fc_out = nn.Linear(self.num_qubits, 2) # Binary classes
        
    def forward(self, x):
        # Down-project inputs classically
        x = torch.tanh(self.fc_in(x))
        # Rescale features to range [0, pi] for angle embedding
        x = (x + 1.0) * (np.pi / 2.0)
        # Execute quantum layer
        x = self.qlayer(x)
        # Up-project quantum outputs classically
        x = self.fc_out(x)
        return x


class QNNClassifier:
    """
    Scikit-learn compatible wrapper for the Torch Hybrid Classical-Quantum Neural Network.
    """
    def __init__(self, num_qubits=4, n_layers=2, epochs=20, lr=0.01, callback=None):
        self.num_qubits = num_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.callback = callback
        
        self.model = None
        self.train_time = 0.0
        self.infer_time = 0.0
        
    def fit(self, X, y):
        start_time = time.perf_counter()
        
        input_dim = X.shape[1]
        self.model = TorchQNN(input_dim=input_dim, num_qubits=self.num_qubits, n_layers=self.n_layers)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Convert numpy arrays to PyTorch Tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if self.callback:
                self.callback(epoch + 1, self.epochs, float(loss.item()))
                
        self.train_time = time.perf_counter() - start_time
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()
        return probs

    def predict(self, X):
        start_time = time.perf_counter()
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        self.infer_time = time.perf_counter() - start_time
        return preds

    def get_resource_metrics(self, n_features):
        # Calculate classical and quantum parameters
        classical_in = n_features * self.num_qubits + self.num_qubits
        quantum_params = self.n_layers * self.num_qubits * 3
        classical_out = self.num_qubits * 2 + 2
        total_params = classical_in + quantum_params + classical_out
        
        return {
            "qubit_count": self.num_qubits,
            "circuit_depth": 1 + self.n_layers * (1 + self.num_qubits),
            "gate_count": self.num_qubits + self.n_layers * (self.num_qubits * 4),
            "trainable_parameters": total_params
        }
