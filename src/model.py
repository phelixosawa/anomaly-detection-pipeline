from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    Wrapper class for Isolation Forest anomaly detection.
    """

    def __init__(
        self,
        contamination: float,
        n_estimators: int,
        random_state: int,
    ) -> None:
        """
        Initialize the anomaly detector.
        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies in the dataset.
        n_estimators : int
            Number of trees in the Isolation Forest.
        random_state : int
            Random seed for reproducibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model: Optional[IsolationForest] = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X_train) -> None:
        """
        Train the Isolation Forest model.
        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training feature set.
        Returns
        -------
        None
        """
        print("[INFO] Training Isolation Forest model...")

        self.model.fit(X_train)

        print("[INFO] Model training completed.")

    def predict(self, X) -> np.ndarray:
        """
        Predict anomalies.
        Notes
        -----
        IsolationForest returns:
            1  -> normal
           -1 -> anomaly
        This method converts predictions to:
            0 -> normal
            1 -> anomaly
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature set for prediction.
        Returns
        -------
        np.ndarray
            Binary anomaly predictions.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been initialized."
            )

        raw_predictions = self.model.predict(X)
        predictions = np.where(raw_predictions == -1, 1, 0)
        print("[INFO] Anomaly prediction completed.")
        return predictions

    def score(self, X) -> np.ndarray:
        """
        Compute anomaly scores using the decision function.
        Notes
        -----
        Lower scores indicate more anomalous observations.
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature set for scoring.
        Returns
        -------
        np.ndarray
            Raw anomaly scores.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been initialized."
            )
        scores = self.model.decision_function(X)
        print("[INFO] Anomaly scoring completed.")
        return scores

    def save_model(self, save_path: str) -> None:
        """
        Save the trained model to disk.
        Parameters
        ----------
        save_path : str
            Path to save the model file.
        Returns
        -------
        None
        """
        if self.model is None:
            raise ValueError(
                "Cannot save an uninitialized model."
            )

        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        joblib.dump(self.model, save_path)
        print(f"[INFO] Model saved to: {save_path}")

    def load_model(self, load_path: str) -> None:
        """
        Load a trained model from disk.
        Parameters
        ----------
        load_path : str
            Path to the saved model file.
        Returns
        -------
        None
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {load_path}"
            )
        self.model = joblib.load(load_path)
        print(f"[INFO] Model loaded from: {load_path}")


if __name__ == "__main__":
    """
    Quick standalone test for model module.
    """
    import pandas as pd
    np.random.seed(42)
    sample_data = pd.DataFrame(
        np.random.randn(1000, 5),
        columns=[f"feature_{i}" for i in range(5)],
    )
    detector = AnomalyDetector(
        contamination=0.01,
        n_estimators=100,
        random_state=42,
    )
    detector.fit(sample_data)
    predictions = detector.predict(sample_data)
    scores = detector.score(sample_data)
    print("\n[INFO] Sample Predictions:")
    print(predictions[:10])
    print("\n[INFO] Sample Scores:")
    print(scores[:10])
    detector.save_model("outputs/models/isolation_forest.joblib")
    detector.load_model("outputs/models/isolation_forest.joblib")
    print("\n[INFO] Model module executed successfully.")