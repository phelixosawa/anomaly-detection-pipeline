from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

def save_flagged_records(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
) -> None:
    """
    Save flagged anomaly records to CSV.
    Notes
    -----
    Adds:
        - anomaly_score
        - is_anomaly
    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    anomaly_scores : np.ndarray
        Raw anomaly scores from the model.
    predictions : np.ndarray
        Binary anomaly predictions.
    save_path : str
        Output CSV path.
    Returns
    -------
    None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_df = df.copy()
    output_df["anomaly_score"] = anomaly_scores
    output_df["is_anomaly"] = predictions
    output_df.to_csv(
        save_path,
        index=False,
    )
    print(
        f"[INFO] Flagged records saved to: {save_path}"
    )

def save_metrics_report(
    metrics: Dict[str, float],
    save_path: str,
) -> None:
    """
    Save evaluation metrics to a text report.
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary containing evaluation metrics.
    save_path : str
        Output report path.
    Returns
    -------
    None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    with open(save_path, "w", encoding="utf-8") as report_file:
        report_file.write(
            "ANOMALY DETECTION METRICS REPORT\n"
        )
        report_file.write("=" * 40 + "\n")
        report_file.write(
            f"Generated: {timestamp}\n\n"
        )
        for metric_name, metric_value in metrics.items():
            report_file.write(
                f"{metric_name}: "
                f"{metric_value:.6f}\n"
            )
    print(
        f"[INFO] Metrics report saved to: {save_path}"
    )

if __name__ == "__main__":
    """
    Quick standalone test for reporting module.
    """
    np.random.seed(42)
    sample_df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
    })
    sample_scores = np.random.randn(100)
    sample_predictions = np.random.choice(
        [0, 1],
        size=100,
        p=[0.95, 0.05],
    )
    sample_metrics = {
        "Precision": 0.812345,
        "Recall": 0.723456,
        "F1 Score": 0.765432,
        "AUC-PR": 0.845678,
        "AUC-ROC": 0.912345,
    }
    save_flagged_records(
        sample_df,
        sample_scores,
        sample_predictions,
        save_path="outputs/flagged_records.csv",
    )
    save_metrics_report(
        sample_metrics,
        save_path="outputs/metrics_report.txt",
    )
    print("[INFO] Reporting module executed successfully.")