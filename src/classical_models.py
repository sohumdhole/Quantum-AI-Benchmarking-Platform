import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

class ClassicalMLWrapper:
    """Wrapper class to provide a unified API for classical classifiers."""
    
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.model = self._init_model()
        self.train_time = 0.0
        self.infer_time = 0.0
        
    def _init_model(self):
        if self.model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                max_depth=self.kwargs.get("max_depth", 5),
                random_state=self.kwargs.get("random_state", 42),
                n_jobs=-1
            )
        elif self.model_name == "XGBoost":
            return XGBClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                max_depth=self.kwargs.get("max_depth", 4),
                learning_rate=self.kwargs.get("learning_rate", 0.1),
                random_state=self.kwargs.get("random_state", 42),
                n_jobs=-1,
                eval_metric="logloss"
            )
        elif self.model_name == "SVM":
            return SVC(
                C=self.kwargs.get("C", 1.0),
                kernel=self.kwargs.get("kernel", "rbf"),
                probability=True,
                random_state=self.kwargs.get("random_state", 42)
            )
        elif self.model_name == "Neural Network":
            hidden_layer_sizes = self.kwargs.get("hidden_layer_sizes", (64, 32))
            return MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=self.kwargs.get("activation", "relu"),
                max_iter=self.kwargs.get("max_iter", 500),
                learning_rate_init=self.kwargs.get("learning_rate_init", 0.01),
                random_state=self.kwargs.get("random_state", 42)
            )
        else:
            raise ValueError(f"Unknown classical model: {self.model_name}")
            
    def fit(self, X, y):
        """Fits the model and tracks execution time."""
        start_time = time.perf_counter()
        self.model.fit(X, y)
        self.train_time = time.perf_counter() - start_time
        return self
        
    def predict(self, X):
        """Predicts classes and tracks execution time."""
        start_time = time.perf_counter()
        preds = self.model.predict(X)
        self.infer_time = time.perf_counter() - start_time
        return preds
        
    def predict_proba(self, X):
        """Predicts class probabilities."""
        return self.model.predict_proba(X)

    def get_resource_metrics(self, n_features):
        """Computes classical model parameters and sizing."""
        metrics = {
            "qubit_count": 0,
            "circuit_depth": 0,
            "gate_count": 0,
            "trainable_parameters": 0
        }
        
        if self.model_name == "Neural Network":
            # Count weights + biases in MLP
            hidden_layers = self.kwargs.get("hidden_layer_sizes", (64, 32))
            layers = [n_features] + list(hidden_layers) + [2]  # Binary classification
            params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
            metrics["trainable_parameters"] = params
            
        elif self.model_name == "SVM":
            # Number of support vectors * features + dual coefficients + intercept
            n_sv = getattr(self.model, "support_", np.array([])).shape[0]
            metrics["trainable_parameters"] = n_sv * (n_features + 1) + 1
            
        elif self.model_name == "Random Forest":
            # Estimation of decision nodes
            total_nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
            metrics["trainable_parameters"] = total_nodes  # surrogate for model complexity
            
        elif self.model_name == "XGBoost":
            # XGBoost booster trees
            booster = self.model.get_booster()
            dump = booster.get_dump()
            total_nodes = sum(len(tree.split('\n')) for tree in dump)
            metrics["trainable_parameters"] = total_nodes
            
        return metrics
