from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

def sample_anomalies(
    X: pd.DataFrame,
    predictions: np.ndarray,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Sample flagged anomalies for SHAP analysis.
    Notes
    -----
    SHAP is computationally expensive on large datasets.
    We therefore:
        1. Filter to predicted anomalies only
        2. Sample up to `sample_size` rows
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataset.
    predictions : np.ndarray
        Binary anomaly predictions.
    sample_size : int
        Maximum number of anomaly rows to sample.
    random_state : int
        Random seed for reproducibility.
    Returns
    -------
    pd.DataFrame
        Sampled anomaly records.
    """
    anomaly_df = X[predictions == 1]
    if anomaly_df.empty:
        raise ValueError(
            "No anomalies detected. SHAP analysis cannot proceed."
        )
    sample_size = min(sample_size, len(anomaly_df))
    sampled_df = anomaly_df.sample(
        n=sample_size,
        random_state=random_state,
    )
    print(
        f"[INFO] Sampled {len(sampled_df)} anomalies "
        f"for SHAP analysis."
    )
    return sampled_df

def get_shap_explainer(
    model,
    X_sample: pd.DataFrame,
):
    """
    Create a SHAP TreeExplainer for Isolation Forest.
    Parameters
    ----------
    model : IsolationForest
        Trained Isolation Forest model.
    X_sample : pd.DataFrame
        Sample feature set.
    Returns
    -------
    shap.TreeExplainer
        SHAP explainer object.
    """
    print("[INFO] Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(
        model,
        X_sample,
    )
    print("[INFO] SHAP explainer created successfully.")
    return explainer

def compute_shap_values(
    explainer,
    X_sample: pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for sampled anomalies.
    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer object.
    X_sample : pd.DataFrame
        Sample feature set.
    Returns
    -------
    np.ndarray
        Computed SHAP values.
    """
    print("[INFO] Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    print("[INFO] SHAP value computation completed.")
    return shap_values

def plot_force(
    explainer,
    shap_values,
    X_sample: pd.DataFrame,
    idx: int,
    save_path: str,
) -> None:
    """
    Save a SHAP force plot for a single anomaly.
    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer object.
    shap_values : np.ndarray
        SHAP values array.
    X_sample : pd.DataFrame
        Sample feature set.
    idx : int
        Row index to visualize.
    save_path : str
        Output image path.
    Returns
    -------
    None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.figure(figsize=(16, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X_sample.iloc[idx],
        matplotlib=True,
        show=False,
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Force plot saved to: {save_path}")

def plot_summary(
    shap_values,
    X_sample: pd.DataFrame,
    save_path: str,
) -> None:
    """
    Save a SHAP summary bar plot.
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array.
    X_sample : pd.DataFrame
        Sample feature set.
    save_path : str
        Output image path.
    Returns
    -------
    None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] SHAP summary plot saved to: {save_path}")

if __name__ == "__main__":
    """
    Quick standalone test for explainability module.
    """
    from sklearn.ensemble import IsolationForest
    np.random.seed(42)
    sample_df = pd.DataFrame(
        np.random.randn(500, 5),
        columns=[f"feature_{i}" for i in range(5)],
    )
    model = IsolationForest(
        contamination=0.05,
        random_state=42,
    )
    model.fit(sample_df)
    raw_predictions = model.predict(sample_df)
    predictions = np.where(raw_predictions == -1, 1, 0)
    sampled_anomalies = sample_anomalies(
        sample_df,
        predictions,
        sample_size=50,
        random_state=42,
    )
    explainer = get_shap_explainer(
        model,
        sampled_anomalies,
    )
    shap_values = compute_shap_values(
        explainer,
        sampled_anomalies,
    )
    for i in range(min(5, len(sampled_anomalies))):
        plot_force(
            explainer,
            shap_values,
            sampled_anomalies,
            idx=i,
            save_path=f"outputs/shap_plots/force_plot_{i}.png",
        )
    plot_summary(
        shap_values,
        sampled_anomalies,
        save_path="outputs/shap_plots/shap_summary.png",
    )
    print("[INFO] Explainability module executed successfully.")