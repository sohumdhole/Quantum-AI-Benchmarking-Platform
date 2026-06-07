# Research Methodology: Quantum AI Benchmarking for Digital Transformation

This document details the mathematical framework, data encoding strategies, model architectures, and evaluation criteria employed in this comparative benchmarking study.

---

## 1. Mathematical Framework for Data Encoding

In order to run classification on classical data using Quantum Machine Learning (QML) models, the classical feature vectors $x \in \mathbb{R}^d$ must be encoded into a quantum state space (Hilbert space $\mathcal{H}$). We employ **Angle Encoding**, which maps each feature to the rotation angle of a specific qubit.

Given a normalized feature vector $x = (x_1, x_2, \dots, x_n)^T$ where $x_i \in [0, \pi]$, the encoding unitary $U(x)$ acts on the initial ground state $|0\rangle^{\otimes n}$ as:

$$| \psi(x) \rangle = U(x) |0\rangle^{\otimes n} = \bigotimes_{k=1}^n R_y(x_k) |0\rangle$$

where $R_y(\theta)$ is the single-qubit rotation about the y-axis:

$$R_y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

---

## 2. Quantum Model Formulations

### A. Quantum Support Vector Machine (QSVM)
Instead of finding a hyper-plane in the classical input space, QSVM maps inputs to a high-dimensional quantum Hilbert space using $U(x)$ and computes a quantum kernel matrix.

The quantum kernel $K(x_i, x_j)$ represents the inner product (overlap) between the quantum states representing features $x_i$ and $x_j$:

$$K(x_i, x_j) = \left| \langle \psi(x_i) | \psi(x_j) \rangle \right|^2 = \left| \langle 0^{\otimes n} | U^\dagger(x_j) U(x_i) | 0^{\otimes n} \rangle \right|^2$$

This kernel matrix is precomputed using a quantum simulator and fed into a classical soft-margin Support Vector Classifier which solves the dual optimization problem:

$$\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

$$\text{subject to } \quad 0 \le \alpha_i \le C, \quad \sum_{i=1}^m \alpha_i y_i = 0$$

### B. Variational Quantum Classifier (VQC)
The VQC represents a parameterized quantum circuit (PQC) which is trained using classical optimization loops. The architecture consists of:
1. **State Preparation**: $U(x)$ to encode classical features.
2. **Parameterized Ansatz**: $W(\theta)$ representing strongly entangling layers composed of single-qubit rotations $R(\alpha, \beta, \gamma)$ and nearest-neighbor CNOT entangling gates.
3. **Measurement**: Expectation value of the Pauli-Z operator on the readout qubit:

$$\hat{y}(x, \theta) = \langle \psi(x) | W^\dagger(\theta) Z_0 W(\theta) | \psi(x) \rangle + b$$

The parameters $\theta$ and bias $b$ are updated using gradient descent (Adam optimizer) by minimizing the Mean Squared Error (MSE) loss function:

$$\mathcal{L}(\theta, b) = \frac{1}{m} \sum_{i=1}^m (\hat{y}(x_i, \theta) - y_i)^2$$

### C. Hybrid Quantum Neural Network (QNN)
The QNN represents a multi-layer hybrid network containing classical feed-forward layers and a quantum layer:
1. **Classical Input Layer**: Linear layer $f_{in}(x) = W_1 x + b_1$ reducing dimensions from input features to $n_{qubits}$.
2. **Quantum Layer**: An $n$-qubit circuit executing $U(f_{in}(x))$ followed by Strongly Entangling ansatz layers, returning expectation values $z_k = \langle Z_k \rangle$ for all qubits.
3. **Classical Output Layer**: A linear layer mapping the expectation outputs $z$ to classification logits: $f_{out}(z) = W_2 z + b_2$.

Gradient propagation through the quantum circuit is handled via the **Parameter-Shift Rule**:

$$\frac{\partial \langle \hat{B} \rangle}{\partial \theta_i} = \frac{\langle \hat{B} \rangle_{\theta_i + s} - \langle \hat{B} \rangle_{\theta_i - s}}{2 \sin(s)}$$

---

## 3. Evaluation Criteria

The benchmarking framework tracks performance on:
1. **Statistical Metrics**: Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
2. **Computational Footprint**: Training time (seconds) and Inference latency (seconds).
3. **Hardware Resources**: Qubit count, circuit depth, total gate counts, and number of trainable parameters.

---

## 4. References

1. Havlíček, V., et al. "Supervised learning with quantum-enhanced feature spaces." *Nature* 567.7747 (2019): 209-212.
2. Schuld, M., & Killoran, N. "Quantum machine learning in feature Hilbert spaces." *Physical Review Letters* 122.4 (2019): 040504.
3. Cerezo, M., et al. "Variational quantum algorithms." *Nature Reviews Physics* 3.9 (2021): 625-644.
