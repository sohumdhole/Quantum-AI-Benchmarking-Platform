import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc

class PublicationVisualizer:
    """
    Generates publication-quality figures matching IEEE formatting standards
    alongside interactive Plotly charts.
    """
    
    def __init__(self, style_name="seaborn-v0_8-whitegrid"):
        self.style_name = style_name
        self._apply_ieee_style()
        
    def _apply_ieee_style(self):
        """Applies styling parameters matching IEEE journal design conventions."""
        try:
            plt.style.use(self.style_name)
        except Exception:
            plt.style.use("default")
            
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8
        })
        # Classic academic color palette (Colorblind safe)
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    def plot_roc_curves(self, benchmark_results, y_test):
        """Generates publication-grade overlaid ROC curves."""
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        
        for idx, (name, metrics) in enumerate(benchmark_results.items()):
            probs = metrics["probabilities"]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(
                fpr, tpr, 
                color=self.colors[idx % len(self.colors)],
                lw=1.5, 
                label=f"{name} (AUC = {roc_auc:.3f})"
            )
            
        ax.plot([0, 1], [0, 1], color="#888888", lw=1.0, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title("Receiver Operating Characteristic (ROC) Comparison")
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")
        ax.grid(True)
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    def plot_metrics_comparison(self, df_comparison):
        """Generates a grouped bar chart comparing performance metrics across models."""
        df_melt = df_comparison.reset_index().melt(
            id_vars="Model", 
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
            var_name="Metric", 
            value_name="Value"
        )
        
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        sns.barplot(
            data=df_melt, 
            x="Metric", 
            y="Value", 
            hue="Model", 
            ax=ax, 
            palette="Set2"
        )
        
        ax.set_ylim([0.0, 1.1])
        ax.set_ylabel("Metric Value")
        ax.set_xlabel("")
        ax.set_title("Multi-Model Statistical Classification Comparison")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=True)
        ax.grid(True, axis="y")
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    def plot_execution_times(self, df_comparison):
        """Generates a bar plot comparing Training and Inference times on log scale."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
        
        models = df_comparison.index
        train_times = df_comparison["Training Time (s)"]
        infer_times = df_comparison["Inference Time (s)"]
        
        # Training time plot (log scale due to huge scale difference)
        ax1.bar(models, train_times, color="#1f77b4", alpha=0.85, edgecolor="#115588")
        ax1.set_yscale("log")
        ax1.set_ylabel("Training Time (seconds) [Log Scale]")
        ax1.set_title("Model Training Execution Cost")
        ax1.set_xticklabels(models, rotation=45, ha="right")
        ax1.grid(True, which="both", axis="y")
        
        # Inference time plot
        ax2.bar(models, infer_times, color="#ff7f0e", alpha=0.85, edgecolor="#cc5500")
        ax2.set_yscale("log")
        ax2.set_ylabel("Inference Time (seconds) [Log Scale]")
        ax2.set_title("Model Inference Execution Cost")
        ax2.set_xticklabels(models, rotation=45, ha="right")
        ax2.grid(True, which="both", axis="y")
        
        sns.despine(ax=ax1)
        sns.despine(ax=ax2)
        fig.tight_layout()
        return fig

    def plot_quantum_resources(self, df_comparison):
        """Plots qubit usage, gate counts, and circuit depths for quantum models."""
        # Filter for models containing 'Q' or 'VQC'
        q_df = df_comparison[df_comparison.index.str.contains("QSVM|VQC|QNN")].copy()
        
        if q_df.empty:
            # Fallback if no quantum models ran
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No Quantum Data Available", ha="center")
            return fig
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
        
        models = q_df.index
        
        # Depth & Gate Count
        ax1.bar(models, q_df["Circuit Depth"], color="#2ca02c", width=0.4, label="Circuit Depth", alpha=0.8)
        ax1.set_ylabel("Circuit Depth (Gate Layers)")
        ax1.set_title("Quantum Circuit Depth Comparison")
        ax1.grid(True, axis="y")
        
        ax2.bar(models, q_df["Gate Count"], color="#9467bd", width=0.4, label="Gate Count", alpha=0.8)
        ax2.set_ylabel("Total Quantum Gates")
        ax2.set_title("Quantum Gate Count Comparison")
        ax2.grid(True, axis="y")
        
        sns.despine(ax=ax1)
        sns.despine(ax=ax2)
        fig.tight_layout()
        return fig

    def plot_pca_space(self, X, y, title="PCA Dimension Reduction Space"):
        """Plots 2D feature projection using Plotly for dashboard interactivity."""
        # If dataset has >2 dimensions, project to 2D
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_projected = pca.fit_transform(X)
        else:
            X_projected = X
            
        df = pd.DataFrame(X_projected, columns=["PC1", "PC2"])
        df["Class"] = y.astype(str)
        
        fig = px.scatter(
            df, x="PC1", y="PC2", color="Class",
            title=title,
            color_discrete_sequence=["#1f77b4", "#d62728"],
            labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
            template="plotly_white"
        )
        fig.update_layout(
            font_family="Times New Roman",
            title_font_size=16,
            xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
            yaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
            legend=dict(bordercolor="#cccccc", borderwidth=1)
        )
        return fig

    def plot_interactive_radar(self, df_comparison):
        """Generates interactive Plotly radar chart comparing all performance metrics."""
        fig = go.Figure()
        categories = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
        
        for model in df_comparison.index:
            values = df_comparison.loc[model, categories].values.tolist()
            # Close the radar circle
            values += [values[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=model,
                fill='toself',
                opacity=0.25
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Algorithm Comparison Profile (Radar View)",
            template="plotly_white",
            font_family="Times New Roman"
        )
        return fig
