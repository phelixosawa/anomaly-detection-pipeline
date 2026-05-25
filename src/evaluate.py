from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def compute_metrics(
    y_true,
    y_pred,
    y_scores,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for anomaly detection.
    Notes
    -----
    Accuracy is intentionally excluded because the dataset
    is highly imbalanced (~0.17% fraud cases).
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Binary predictions.
    y_scores : array-like
        Raw anomaly scores from the model.
    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics.
    """
    # Invert scores because lower Isolation Forest scores
    # indicate more anomalous observations.
    anomaly_scores = -np.asarray(y_scores)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_pr = average_precision_score(
        y_true,
        anomaly_scores,
    )
    auc_roc = roc_auc_score(
        y_true,
        anomaly_scores,
    )
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC-PR": auc_pr,
        "AUC-ROC": auc_roc,
    }
    return metrics

def plot_pr_curve(
    y_true,
    y_scores,
    save_path: str,
) -> None:
    """
    Plot and save the Precision-Recall curve.
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_scores : array-like
        Raw anomaly scores.
    save_path : str
        Path to save the PR curve image.
    Returns
    -------
    None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    anomaly_scores = -np.asarray(y_scores)

    precision, recall, _ = precision_recall_curve(
        y_true,
        anomaly_scores,
    )
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"PR AUC = {pr_auc:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] PR curve saved to: {save_path}")

def plot_roc_curve(
    y_true,
    y_scores,
    save_path: str,
) -> None:
    """
    Plot and save the ROC curve.
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_scores : array-like
        Raw anomaly scores.
    save_path : str
        Path to save the ROC curve image.
    Returns
    -------
    None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    anomaly_scores = -np.asarray(y_scores)
    fpr, tpr, _ = roc_curve(
        y_true,
        anomaly_scores,
    )
    roc_auc = roc_auc_score(
        y_true,
        anomaly_scores,
    )
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"ROC AUC = {roc_auc:.4f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curve saved to: {save_path}")

def print_metrics_table(
    metrics: Dict[str, float]
) -> None:
    """
    Print evaluation metrics in a formatted table.
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary containing evaluation metrics.
    Returns
    -------
    None
    """
    print("\n" + "=" * 40)
    print(" ANOMALY DETECTION METRICS ")
    print("=" * 40)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:<15}: {metric_value:.6f}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    """
    Quick standalone test for evaluation module.
    """
    np.random.seed(42)
    y_true = np.random.choice(
        [0, 1],
        size=1000,
        p=[0.98, 0.02],
    )
    y_pred = np.random.choice(
        [0, 1],
        size=1000,
        p=[0.97, 0.03],
    )
    y_scores = np.random.randn(1000)
    metrics = compute_metrics(
        y_true,
        y_pred,
        y_scores,
    )
    print_metrics_table(metrics)
    plot_pr_curve(
        y_true,
        y_scores,
        "outputs/eval_plots/pr_curve.png",
    )
    plot_roc_curve(
        y_true,
        y_scores,
        "outputs/eval_plots/roc_curve.png",
    )
    print("[INFO] Evaluation module executed successfully.")