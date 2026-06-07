import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.classical_models import ClassicalMLWrapper
from src.quantum_models import QSVMClassifier, VQCClassifier, QNNClassifier

class BenchmarkOrchestrator:
    """
    Orchestrates the training, prediction, evaluation, and resource tracking
    for all classical and quantum classifiers.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, num_qubits=4, random_state=42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_qubits = num_qubits
        self.random_state = random_state
        self.results = {}
        
    def run_classical_model(self, model_name, **kwargs):
        """Fits and evaluates a classical ML model."""
        wrapper = ClassicalMLWrapper(model_name, random_state=self.random_state, **kwargs)
        wrapper.fit(self.X_train, self.y_train)
        
        preds = wrapper.predict(self.X_test)
        probs = wrapper.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(preds, probs)
        metrics["train_time_sec"] = wrapper.train_time
        metrics["infer_time_sec"] = wrapper.infer_time
        
        # Resources
        res = wrapper.get_resource_metrics(self.X_train.shape[1])
        metrics.update(res)
        
        self.results[model_name] = metrics
        return metrics

    def run_qsvm(self, C=1.0):
        """Fits and evaluates Quantum SVM."""
        qsvm = QSVMClassifier(num_qubits=self.num_qubits, C=C, random_state=self.random_state)
        qsvm.fit(self.X_train, self.y_train)
        
        preds = qsvm.predict(self.X_test)
        probs = qsvm.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(preds, probs)
        metrics["train_time_sec"] = qsvm.train_time
        metrics["infer_time_sec"] = qsvm.infer_time
        
        # Resources
        res = qsvm.get_resource_metrics()
        metrics.update(res)
        
        self.results["QSVM"] = metrics
        return metrics

    def run_vqc(self, n_layers=3, epochs=20, lr=0.05, callback=None):
        """Fits and evaluates Variational Quantum Classifier."""
        vqc = VQCClassifier(
            num_qubits=self.num_qubits, 
            n_layers=n_layers, 
            epochs=epochs, 
            lr=lr, 
            callback=callback
        )
        vqc.fit(self.X_train, self.y_train)
        
        preds = vqc.predict(self.X_test)
        probs = vqc.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(preds, probs)
        metrics["train_time_sec"] = vqc.train_time
        metrics["infer_time_sec"] = vqc.infer_time
        
        # Resources
        res = vqc.get_resource_metrics()
        metrics.update(res)
        
        self.results["VQC"] = metrics
        return metrics

    def run_qnn(self, n_layers=2, epochs=20, lr=0.01, callback=None):
        """Fits and evaluates Hybrid Quantum Neural Network."""
        qnn = QNNClassifier(
            num_qubits=self.num_qubits, 
            n_layers=n_layers, 
            epochs=epochs, 
            lr=lr, 
            callback=callback
        )
        qnn.fit(self.X_train, self.y_train)
        
        preds = qnn.predict(self.X_test)
        probs = qnn.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(preds, probs)
        metrics["train_time_sec"] = qnn.train_time
        metrics["infer_time_sec"] = qnn.infer_time
        
        # Resources
        res = qnn.get_resource_metrics(self.X_train.shape[1])
        metrics.update(res)
        
        self.results["QNN"] = metrics
        return metrics

    def _calculate_metrics(self, preds, probs):
        """Computes statistical classifiers evaluation metrics."""
        acc = accuracy_score(self.y_test, preds)
        
        # Use zero_division parameter for edge cases
        precision = precision_score(self.y_test, preds, zero_division=0)
        recall = recall_score(self.y_test, preds, zero_division=0)
        f1 = f1_score(self.y_test, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(self.y_test, probs)
        except Exception:
            auc = 0.5  # default if ROC cannot be calculated (e.g. only 1 class in sample)
            
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc,
            "predictions": preds.tolist(),
            "probabilities": probs.tolist()
        }
        
    def get_comparison_table(self):
        """Aggregates results into a structured Pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
            
        records = []
        for name, metrics in self.results.items():
            record = {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "AUC-ROC": metrics["auc_roc"],
                "Training Time (s)": metrics["train_time_sec"],
                "Inference Time (s)": metrics["infer_time_sec"],
                "Qubits Used": metrics["qubit_count"],
                "Circuit Depth": metrics["circuit_depth"],
                "Gate Count": metrics["gate_count"],
                "Trainable Params": metrics["trainable_parameters"]
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        df.set_index("Model", inplace=True)
        return df
