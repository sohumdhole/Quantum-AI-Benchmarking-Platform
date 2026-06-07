# Quantum AI Benchmarking Platform for Digital Transformation

⚛️ **A rigorous, publication-grade benchmark suite comparing classical and quantum machine learning algorithms on industrial digital transformation datasets.**

This platform compares standard classical machine learning classifiers against variational quantum classifiers and quantum support vector machine kernels. It features an interactive, high-end Streamlit dashboard with custom dark glassmorphic styling, live training optimization tracking, and download-ready IEEE-formatted scientific manuscript assets.

---

## 🔬 Scientific & Mathematical Overview

The platform evaluates performance across three digital transformation scenarios:
1. **Predictive Maintenance**: Wear and tear classification based on telemetry sensor features.
2. **Financial Fraud Detection**: Anomaly identification over transaction velocities and risk scores.
3. **Customer Churn Prediction**: Customer behavior classification based on service usage logs.

### Feature Mapping & State Preparation
Classical feature vectors $x \in \mathbb{R}^d$ are mapped to Hilbert space $\mathcal{H}$ using **Angle Embedding**:
$$| \psi(x) \rangle = \bigotimes_{k=1}^d R_y(x_k) |0\rangle$$

### Evaluated Model Architectures

#### Classical Models:
* **Random Forest**: Ensemble decision tree classifier.
* **XGBoost**: Gradient-boosted decision trees.
* **Support Vector Machine (SVM)**: Classical kernel-based classifier (Radial Basis Function).
* **Neural Network**: Classical Multi-Layer Perceptron (MLP).

#### Quantum Models:
* **Quantum Support Vector Machine (QSVM)**: precomputes a pairwise kernel using transition probability:
  $$K(x_i, x_j) = \left| \langle \psi(x_i) | \psi(x_j) \rangle \right|^2$$
* **Variational Quantum Classifier (VQC)**: Encodes features, applies a parameterized strongly entangling ansatz $W(\theta)$, and reads the expectation value:
  $$\hat{y}(x, \theta) = \langle \psi(x) | W^\dagger(\theta) Z_0 W(\theta) | \psi(x) \rangle + b$$
* **Hybrid Quantum Neural Network (QNN)**: Standard Feed-Forward Neural Network integrated with a `PennyLane` variational quantum layer.

---

## 📂 Project Structure

```
quantum-ai-benchmarking-platform/
├── app.py                      # Main Streamlit Dashboard (Premium UI)
├── requirements.txt            # Python dependencies
├── README.md                   # Repository Documentation (this file)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data generation & PCA preprocessing pipeline
│   ├── classical_models.py     # Classical model wrapper class
│   ├── quantum_models.py       # QSVM, VQC, and Hybrid QNN models (PennyLane/PyTorch)
│   ├── benchmarker.py          # Metrics (F1, Accuracy, AUC) and cost orchestrator
│   └── visualizer.py           # IEEE-compliant plot generation
└── paper/
    ├── paper.tex               # IEEE Transactions LaTeX template
    ├── references.bib          # Bibliography references database
    └── methodology.md          # Theoretical methodology reference
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or 3.11 (highly recommended for package compatibility)
- C++ compiler tools (required for installing some simulator backends)

### 1. Installation
Clone the repository and set up a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Benchmarking Dashboard
Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to the local URL (typically `http://localhost:8501`).

---

## 📈 Performance & Resource Metrics

The platform evaluates algorithms across three major dimensions:
1. **Classification Power**: Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
2. **Computational Footprint**: Training execution time and inference latency.
3. **Quantum Resource Constraints**: Qubit usage, circuit depth, total gates, and CNOT counts.

*Note: Since simulating quantum models is computationally expensive, the dashboard dynamically performs PCA to reduce feature dimensionality (typically 2-8 features/qubits) to fit simulator limits.*

---

## 📄 IEEE Manuscript Templates
The `paper/` directory contains standard manuscript assets suitable for IEEE journal submission. The dashboard allows you to:
1. View the abstract and mathematical methodology.
2. View and modify the LaTeX source code (`paper.tex`).
3. Download the compiled-ready bibliography database (`references.bib`).
4. Download vector-graphic PDF plots (ROC curves, metrics) styled to match IEEE publication margins.

---

## 📜 Citation
If you use this platform in your academic research, please cite:
```bibtex
@article{quantum_ai_benchmarking_2026,
  title={Quantum AI Benchmarking Platform for Digital Transformation},
  author={Sohum Vivek Dhole},
  journal={Dublin Business School},
  year={2026}
}
```

---

Developed by:

**Sohum Vivek Dhole**  
MSc. in Business Analytics,  
Dublin Business School.  

📧 **Email** - dholesohum@gmail.com
