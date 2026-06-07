import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from src.data_loader import DigitalTransformationDataset
from src.benchmarker import BenchmarkOrchestrator
from src.visualizer import PublicationVisualizer

# Set page layout and title
st.set_page_config(
    page_title="Quantum AI Benchmarking Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Academic Aesthetics (Dark Mode & Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Fira+Code:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: 'Outfit', -apple-system, sans-serif;
    }
    
    /* Header/Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.9) 0%, rgba(13, 15, 20, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 35px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Cards and Glassmorphism Container */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.2);
    }
    
    /* Code & Equations */
    code, pre {
        font-family: 'Fira Code', monospace !important;
    }
    
    /* Streamlit Components custom styling */
    div[data-testid="stSidebar"] {
        background-color: #090b0f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    div[data-testid="stMetricValue"] {
        font-weight: 800;
        background: linear-gradient(135deg, #a855f7, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* IEEE Alert badge */
    .ieee-badge {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(14, 165, 233, 0.3);
        color: #38bdf8;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "benchmark_orchestrator" not in st.session_state:
    st.session_state.benchmark_orchestrator = None
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "benchmarked_models" not in st.session_state:
    st.session_state.benchmarked_models = {}
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "dataset_meta" not in st.session_state:
    st.session_state.dataset_meta = {}

# Visualizer instance
visualizer = PublicationVisualizer()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/IEEE_logo.svg", width=120)
    st.markdown("### Benchmarking Suite")
    
    dataset_choice = st.selectbox(
        "Digital Transformation Scenario",
        ["Predictive Maintenance", "Financial Fraud", "Customer Churn"]
    )
    
    st.markdown("---")
    st.markdown("### Dataset Dimensions")
    num_samples = st.slider("Number of Samples", min_value=50, max_value=500, value=150, step=10)
    num_qubits = st.slider("Qubits / PCA Features", min_value=2, max_value=8, value=4, step=1)
    noise_level = st.slider("Dataset Noise Level", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
    
    st.markdown("---")
    st.markdown("### Model Hyperparameters")
    quantum_epochs = st.slider("Quantum Training Epochs", min_value=5, max_value=50, value=15, step=5)
    quantum_lr = st.selectbox("Quantum Learning Rate", [0.01, 0.05, 0.1])
    
    st.markdown("---")
    st.info("💡 Pro-Tip: Simulating QML scales exponentially. We limit qubits to 2-8 and samples to 50-500 to guarantee fast simulation times in standard CPUs.")

# --- HERO SECTION ---
st.markdown("""
<div class="hero-container">
    <div class="ieee-badge">IEEE Academic Format Compatible</div>
    <h1 class="hero-title">Quantum AI Benchmarking Platform</h1>
    <p class="hero-subtitle">Rigorous Empirical Comparison of Classical vs. Quantum Machine Learning Algorithms for Digital Transformation Scenarios</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_hub, tab_data, tab_circuits, tab_benchmark, tab_analytics, tab_paper = st.tabs([
    "🏠 Dashboard Hub",
    "📊 Dataset Explorer",
    "🔬 Quantum Circuits",
    "⚡ Train & Benchmark",
    "📈 Comparative Analytics",
    "📄 IEEE Manuscript"
])

# Define mapping for dataset types
db_type_map = {
    "Predictive Maintenance": "predictive_maintenance",
    "Financial Fraud": "financial_fraud",
    "Customer Churn": "customer_churn"
}

# Load data based on sidebar configuration
dataset_manager = DigitalTransformationDataset(
    dataset_type=db_type_map[dataset_choice],
    n_samples=num_samples,
    n_features=num_qubits if num_qubits > 6 else 6, # ensure enough features before PCA
    noise=noise_level,
    random_state=42
)
X_train, X_test, y_train, y_test = dataset_manager.preprocess(
    test_size=0.25,
    pca_components=num_qubits,
    scale_quantum=True
)

# --- TAB 1: DASHBOARD HUB ---
with tab_hub:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Abstract & Platform Overview
        
        The digitalization of legacy operations requires intelligent classifiers for tabular datasets. This platform evaluates whether **Quantum Machine Learning (QML)** models provide an advantage in accuracy, robustness, or parameter efficiency over **Classical Machine Learning** systems.
        
        #### Core Framework Stack
        - **Classical Architectures**: Random Forest, XGBoost, Support Vector Classifier, Classical Neural Network (MLP).
        - **Quantum Architectures**: Quantum SVM (QSVM), Variational Quantum Classifier (VQC), Hybrid Classical-Quantum Neural Network (QNN).
        - **Simulation Backend**: `PennyLane` using the `default.qubit` simulator, mapped to Qiskit-style unitary feature mappings.
        
        #### Research Framework & Mathematical Embeddings
        - **Quantum Kernels (QSVM)**: Maps data $x \in \mathbb{R}^d$ to state $|\phi(x)\rangle$, computing pairwise matrix $K_{ij} = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$.
        - **Variational Classifier (VQC)**: Encodes input features via Pauli $Y$ rotation: $R_Y(x_i) |0\rangle$, followed by parameterized entangling layers: $U(\theta)$.
        - **Hybrid QNN**: Integrates classical Dense Layers with parameterized quantum layers. Backward pass utilizes PennyLane's Parameter Shift rule.
        """)
        
        st.markdown(r"""
        $$\text{VQC Output Formulation: } f(x, \theta) = \langle 0 | U^\dagger(x) W^\dagger(\theta) Z_0 W(\theta) U(x) | 0 \rangle + b$$
        """)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Configured Experiment")
        st.metric(label="Target Scenario", value=dataset_choice)
        st.metric(label="Total Dataset Size", value=f"{num_samples} records")
        st.metric(label="Dimensionality ($d$)", value=f"{num_qubits} Qubits")
        st.metric(label="Noise Level ($flip\_y$)", value=f"{noise_level * 100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: DATASET EXPLORER ---
with tab_data:
    st.markdown("### Dataset Overview")
    df_raw = dataset_manager.get_data()
    
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.markdown("#### Tabular Schema Sample")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.markdown("#### Class Distributions")
        st.dataframe(df_raw["Target"].value_counts(normalize=True).to_frame(name="Proportion"), use_container_width=True)
        
    with col_d2:
        st.markdown("#### PCA 2D Feature Space")
        fig_pca = visualizer.plot_pca_space(df_raw.drop("Target", axis=1).values, df_raw["Target"].values)
        st.plotly_chart(fig_pca, use_container_width=True)

# --- TAB 3: QUANTUM CIRCUITS ---
with tab_circuits:
    st.markdown("### Quantum Model Mathematical Architectures")
    
    col_c1, col_c2 = st.columns([1, 1])
    
    with col_c1:
        st.markdown("#### 1. Quantum SVM Kernel Mapping")
        st.markdown(r"""
        The QSVM utilizes a dual optimization mapping. Classical features $x_i$ are encoded via angle embedding $\phi(x)$, and the kernel $K(x_i, x_j)$ represents transition probabilities:
        $$K(x_i, x_j) = \left| \langle 0^{\otimes n} | U^\dagger(x_j) U(x_i) | 0^{\otimes n} \rangle \right|^2$$
        """)
        
        st.markdown("##### PennyLane Implementation Code Snippet:")
        st.code("""
@qml.qnode(dev)
def kernel_circuit(x1, x2):
    # Encoding x1
    for i in range(num_qubits):
        qml.RY(x1[i], wires=i)
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
        
    # Adjoint Encoding of x2
    for i in reversed(range(num_qubits - 1)):
        qml.CNOT(wires=[i, i + 1])
    for i in range(num_qubits):
        qml.RY(-x2[i], wires=i)
        
    return qml.probs(wires=range(num_qubits))
        """, language="python")
        
    with col_c2:
        st.markdown("#### 2. Variational Quantum Classifier (VQC)")
        st.markdown(r"""
        Features are loaded via $R_Y(\theta)$ angle embeddings. Parameterized state rotation is completed via Strongly Entangling Layers $U(\theta)$, comprising arbitrary single-qubit rotations followed by entangling rings.
        """)
        
        st.markdown("##### PennyLane Implementation Code Snippet:")
        st.code("""
@qml.qnode(dev)
def vqc_circuit(weights, features):
    # Encoding Features
    qml.AngleEmbedding(features, wires=range(num_qubits), rotation='Y')
    
    # Strongly Entangling Ansatz
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    
    # Measuring Expectation Value
    return qml.expval(qml.PauliZ(0))
        """, language="python")

    st.markdown("---")
    st.markdown("#### 3. Hybrid Quantum-Classical Neural Network (QNN)")
    st.markdown(r"""
    A unified multi-layer network where classical feed-forward layers down-project input space to $N_{qubits}$, a quantum variational layer implements rotations, and a final classical dense layer maps the state expectation outputs to target classification logits.
    """)
    st.code("""
class TorchQNN(nn.Module):
    def __init__(self, input_dim, num_qubits, n_layers=2):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, num_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
            
        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, num_qubits, 3)})
        self.fc_out = nn.Linear(num_qubits, 2)
        
    def forward(self, x):
        x = torch.tanh(self.fc_in(x))
        x = (x + 1.0) * (np.pi / 2.0) # Scale to [0, pi]
        x = self.qlayer(x)
        return self.fc_out(x)
    """, language="python")

# --- TAB 4: TRAIN & BENCHMARK ---
with tab_benchmark:
    st.markdown("### Benchmarking Execution Console")
    
    # Selection of models
    col_sel1, col_sel2 = st.columns([1, 1])
    with col_sel1:
        classical_models_to_run = st.multiselect(
            "Classical ML Models to Evaluate",
            ["Random Forest", "XGBoost", "SVM", "Neural Network"],
            default=["Random Forest", "XGBoost", "SVM", "Neural Network"]
        )
    with col_sel2:
        quantum_models_to_run = st.multiselect(
            "Quantum ML Models to Evaluate",
            ["QSVM", "VQC", "QNN"],
            default=["QSVM", "VQC", "QNN"]
        )
        
    run_btn = st.button("🚀 Run Rigorous Benchmarking Suite", type="primary")
    
    if run_btn:
        st.session_state.benchmarked_models = {}
        orchestrator = BenchmarkOrchestrator(
            X_train, X_test, y_train, y_test,
            num_qubits=num_qubits,
            random_state=42
        )
        st.session_state.benchmark_orchestrator = orchestrator
        st.session_state.y_test = y_test
        
        # 1. Classical Model Execution
        for model in classical_models_to_run:
            with st.spinner(f"Fitting Classical: {model}..."):
                metrics = orchestrator.run_classical_model(model)
                st.session_state.benchmarked_models[model] = metrics
                st.toast(f"✅ {model} Training Completed!")
                
        # 2. QSVM Execution
        if "QSVM" in quantum_models_to_run:
            with st.spinner("Computing Quantum Kernel and fitting QSVM..."):
                metrics = orchestrator.run_qsvm(C=1.0)
                st.session_state.benchmarked_models["QSVM"] = metrics
                st.toast("✅ Quantum SVM Training Completed!")
                
        # 3. VQC Execution
        if "VQC" in quantum_models_to_run:
            st.markdown("#### VQC Real-Time Training Log")
            vqc_progress = st.progress(0)
            vqc_chart = st.empty()
            losses = []
            
            def vqc_callback(epoch, max_epochs, loss):
                vqc_progress.progress(epoch / max_epochs)
                losses.append(loss)
                df_loss = pd.DataFrame({"Epoch": range(1, len(losses)+1), "Loss": losses})
                vqc_chart.line_chart(df_loss.set_index("Epoch"), height=180)
                
            metrics = orchestrator.run_vqc(
                n_layers=3, 
                epochs=quantum_epochs, 
                lr=quantum_lr, 
                callback=vqc_callback
            )
            st.session_state.benchmarked_models["VQC"] = metrics
            st.toast("✅ VQC Training Completed!")
            
        # 4. QNN Execution
        if "QNN" in quantum_models_to_run:
            st.markdown("#### Hybrid QNN Real-Time Training Log")
            qnn_progress = st.progress(0)
            qnn_chart = st.empty()
            qnn_losses = []
            
            def qnn_callback(epoch, max_epochs, loss):
                qnn_progress.progress(epoch / max_epochs)
                qnn_losses.append(loss)
                df_loss = pd.DataFrame({"Epoch": range(1, len(qnn_losses)+1), "Loss": qnn_losses})
                qnn_chart.line_chart(df_loss.set_index("Epoch"), height=180)
                
            metrics = orchestrator.run_qnn(
                n_layers=2, 
                epochs=quantum_epochs, 
                lr=0.01, 
                callback=qnn_callback
            )
            st.session_state.benchmarked_models["QNN"] = metrics
            st.toast("✅ Quantum Neural Network Training Completed!")
            
        # Build comparative table
        df_comparison = orchestrator.get_comparison_table()
        st.session_state.df_results = df_comparison
        
    # Display comparison if exists
    if st.session_state.df_results is not None:
        st.success("🎉 Benchmarking completed successfully! Results are detailed below:")
        
        # Display Table
        st.dataframe(
            st.session_state.df_results.style.background_gradient(
                cmap="Purples", 
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
            ).background_gradient(
                cmap="Oranges", 
                subset=["Training Time (s)", "Inference Time (s)"]
            ),
            use_container_width=True
        )
    else:
        st.info("📊 Configure settings and click the 'Run Rigorous Benchmarking Suite' button above to generate comparison data.")

# --- TAB 5: COMPARATIVE ANALYTICS ---
with tab_analytics:
    if st.session_state.df_results is not None:
        st.markdown("### IEEE Academic Standard Visualizations")
        
        col_v1, col_v2 = st.columns([1, 1])
        
        with col_v1:
            st.markdown("#### 1. Overlaid Receiver Operating Characteristic (ROC)")
            fig_roc = visualizer.plot_roc_curves(st.session_state.benchmarked_models, st.session_state.y_test)
            st.pyplot(fig_roc)
            
            # Export button
            buf = io.BytesIO()
            fig_roc.savefig(buf, format="pdf", bbox_inches="tight")
            st.download_button(
                label="📥 Download ROC Curve (IEEE PDF Format)",
                data=buf.getvalue(),
                file_name="fig_roc_curves.pdf",
                mime="application/pdf"
            )
            
        with col_v2:
            st.markdown("#### 2. Overall Performance Metrics")
            fig_met = visualizer.plot_metrics_comparison(st.session_state.df_results)
            st.pyplot(fig_met)
            
            buf = io.BytesIO()
            fig_met.savefig(buf, format="pdf", bbox_inches="tight")
            st.download_button(
                label="📥 Download Performance Bar Plot (IEEE PDF Format)",
                data=buf.getvalue(),
                file_name="fig_performance_metrics.pdf",
                mime="application/pdf"
            )
            
        st.markdown("---")
        
        col_v3, col_v4 = st.columns([1, 1])
        
        with col_v3:
            st.markdown("#### 3. Execution Time Benchmarking (Log Scale)")
            fig_time = visualizer.plot_execution_times(st.session_state.df_results)
            st.pyplot(fig_time)
            
            buf = io.BytesIO()
            fig_time.savefig(buf, format="pdf", bbox_inches="tight")
            st.download_button(
                label="📥 Download Execution Cost Plot (IEEE PDF Format)",
                data=buf.getvalue(),
                file_name="fig_execution_cost.pdf",
                mime="application/pdf"
            )
            
        with col_v4:
            st.markdown("#### 4. Quantum Hardware Resource Analysis")
            fig_res = visualizer.plot_quantum_resources(st.session_state.df_results)
            st.pyplot(fig_res)
            
            buf = io.BytesIO()
            fig_res.savefig(buf, format="pdf", bbox_inches="tight")
            st.download_button(
                label="📥 Download Quantum Resource Plot (IEEE PDF Format)",
                data=buf.getvalue(),
                file_name="fig_quantum_resources.pdf",
                mime="application/pdf"
            )
            
        st.markdown("---")
        st.markdown("#### 5. Interactive Radar Metric Profile")
        fig_radar = visualizer.plot_interactive_radar(st.session_state.df_results)
        st.plotly_chart(fig_radar, use_container_width=True)
        
    else:
        st.info("📉 Training has not been executed yet. Run the benchmarking suite on Tab 4 to generate visualizations.")

# --- TAB 6: IEEE MANUSCRIPT ---
with tab_paper:
    st.markdown("### IEEE Research Methodology & Compiled Abstract")
    
    st.markdown("""
    Below is the structured draft of the research paper describing this study, ready for submission to the **IEEE Transactions on Neural Networks and Learning Systems** or **IEEE Transactions on Quantum Engineering**.
    """)
    
    col_p1, col_p2 = st.columns([2, 1])
    
    with col_p1:
        st.markdown('<div class="glass-card" style="background: rgba(15, 23, 42, 0.8);">', unsafe_allow_html=True)
        st.markdown("""
        # Quantum AI Benchmarking Platform for Digital Transformation
        
        **Author:** Sohum Vivek Dhole  
        **Affiliation:** MSc. in Business Analytics, Dublin Business School  
        **Email:** dholesohum@gmail.com  
        
        ---
        
        ### Abstract
        Modern digital transformation strategies in industrial environments rely on accurate classification algorithms. In this study, we provide a formal, comparative evaluation of four classical machine learning classifiers (Random Forest, XGBoost, SVM, and Classical Multi-Layer Perceptrons) against three state-of-the-art quantum counterparts (Quantum SVM with Pauli-Z kernel, Variational Quantum Classifiers (VQC), and Hybrid Quantum-Classical Neural Networks). Evaluating these on digital transformation scenarios, we systematically map the boundaries where Classical ML models outperform, and where Quantum models demonstrate competitive performance despite high simulation latencies. Our results outline critical scaling limits, parameter efficiencies, and hardware-level quantum resource constraints (circuit depth, qubit requirements).
        
        ---
        
        ### I. Introduction
        Digital transformation drives optimization across predictive maintenance, fraud mitigation, and customer retention. However, standard classifiers are prone to parameter inflation and overfitting when boundaries are complex. Quantum Machine Learning (QML) models utilize Hilbert spaces via feature mappings to represent high-dimensional features directly on qubits...
        
        ### II. Methodology
        - **Data Encoding**: Features are mapped into quantum states using angle embeddings. Let $x = (x_1, \dots, x_d) \in [0, \pi]^d$, then the encoding unitary is:
          $$U(x) = \bigotimes_{k=1}^d R_Y(x_k)$$
        - **Variational Classifier (VQC)**: Entanglement is introduced using CNOT gates on nearest-neighbor qubits, followed by trainable rotation gates parametrized by $\theta$.
        - **QSVM**: Evaluated using a precomputed quantum kernel generated via:
          $$K(x_i, x_j) = |\langle 0 | U^\dagger(x_j) U(x_i) | 0 \rangle|^2$$
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_p2:
        st.markdown("#### Manuscript Assets")
        
        latex_template = r"""\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{cite}

\begin{document}

\title{Quantum AI Benchmarking Platform for Digital Transformation}
\author{Sohum Vivek Dhole \\ MSc. in Business Analytics, Dublin Business School \\ Email: dholesohum@gmail.com}

\maketitle

\begin{abstract}
Digital transformation requires predictive analytics over high-dimensional operations. This work presents a rigorous benchmarking platform comparing classical and quantum classifiers (QSVM, VQC, Hybrid QNN) across predictive maintenance, financial fraud, and customer churn.
\end{abstract}

\begin{IEEEkeywords}
Quantum Machine Learning, Digital Transformation, Benchmarking, QSVM, Variational Quantum Classifier.
\end{IEEEkeywords}

\section{Introduction}
\IEEEPARstart{A}{s} legacy systems digitize, the requirement for robust classification of tabular data becomes paramount. Quantum computer capabilities offer mathematical Hilbert space processing.

\section{Quantum Classifiers}
\subsection{Quantum Support Vector Machine}
The quantum kernel matrix is computed by measuring transition probability:
\begin{equation}
K(x_i, x_j) = |\langle 0^{\otimes n} | U^\dagger(x_j) U(x_i) | 0^{\otimes n} \rangle|^2
\end{equation}

\subsection{Variational Quantum Classifier}
The parameterized model outputs state:
\begin{equation}
f(x, \theta) = \langle 0 | U^\dagger(x) W^\dagger(\theta) Z_0 W(\theta) U(x) | 0 \rangle
\end{equation}

\section{Experimental Benchmarks}
Evaluation parameters include:
\begin{itemize}
    \item Classification Metrics (Accuracy, F1-Score, AUC-ROC)
    \item Computational Overhead (Training and Inference Times)
    \item Hardware Resources (Qubits, Gate Count, Circuit Depth)
\end{itemize}

\section{Conclusion}
Our platform charts the threshold where Quantum ML begins to display parameter efficiency at the expense of computational overhead on simulators.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""
        st.text_area("LaTeX Source Code (`paper.tex`)", value=latex_template, height=350)
        
        st.download_button(
            label="📥 Download LaTeX Source (.tex)",
            data=latex_template,
            file_name="paper.tex",
            mime="text/plain"
        )
        
        bib_template = """@article{havlicek2019supervised,
  title={Supervised learning with quantum-enhanced feature spaces},
  author={Havl{\'\i}{\v{c}}ek, Vojt{\v{e}}ch and C{\'o}rcoles, Antonio D and Temme, Kristan and Harrow, Aram W and Kandala, Abhinav and Chow, Jerry M and Gambetta, Jay M},
  journal={Nature},
  volume={567},
  number={7747},
  pages={209--212},
  year={2019},
  publisher={Nature Publishing Group}
}

@article{schuld2019quantum,
  title={Quantum machine learning in feature Hilbert spaces},
  author={Schuld, Maria and Killoran, Nathan},
  journal={Physical review letters},
  volume={122},
  number={4},
  pages={040504},
  year={2019},
  publisher={APS}
}
"""
        st.download_button(
            label="📥 Download Bibliography (.bib)",
            data=bib_template,
            file_name="references.bib",
            mime="text/plain"
        )
