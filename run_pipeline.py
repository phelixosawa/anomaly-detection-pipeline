from pathlib import Path
import yaml
from src.evaluate import (
    compute_metrics,
    plot_pr_curve,
    plot_roc_curve,
    print_metrics_table,
)
from src.explain import (
    compute_shap_values,
    get_shap_explainer,
    plot_force,
    plot_summary,
    sample_anomalies,
)
from src.model import AnomalyDetector
from src.preprocess import (
    load_path,
    scale_features,
    split_data,
)
from src.report import (
    save_flagged_records,
    save_metrics_report,
)

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load YAML configuration file.
    Parameters
    ----------
    config_path : str
        Path to configuration YAML.
    Returns
    -------
    dict
        Loaded configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}"
        )
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    print(f"[INFO] Configuration loaded from: {config_file}")
    return config

def main() -> None:
    """
    Run the end-to-end anomaly detection pipeline.
    """
    try:
        print("\n" + "=" * 60)
        print(" END-TO-END ANOMALY DETECTION PIPELINE ")
        print("=" * 60)

        # [1/7] Load Configuration
        print("\n[1/7] Loading configuration...")
        config = load_config()

        # [2/7] Load and Preprocess Data
        print("\n[2/7] Loading and preprocessing dataset...")
        df = load_path(config["data_path"])
        scaled_df = scale_features(df)
        X_train, X_test, y_train, y_test = split_data(
            scaled_df,
            test_size=config["test_size"],
            random_state=config["random_state"],
        )
    
        # [3/7] Train Model
        print("\n[3/7] Training anomaly detection model...")
        detector = AnomalyDetector(
            contamination=config["contamination"],
            n_estimators=config["n_estimators"],
            random_state=config["random_state"],
        )
        detector.fit(X_train)
        model_output_path = (
            Path(config["output_dir"])
            / "models"
            / "isolation_forest.joblib"
        )
        detector.save_model(model_output_path)

        # [4/7] Generate Predictions and Scores
        print("\n[4/7] Generating predictions and anomaly scores...")
        predictions = detector.predict(X_test)
        anomaly_scores = detector.score(X_test)

        # [5/7] Evaluate Model
        print("\n[5/7] Evaluating model performance...")
        metrics = compute_metrics(
            y_test,
            predictions,
            anomaly_scores,
        )
        print_metrics_table(metrics)
        eval_output_dir = (
            Path(config["output_dir"])
            / "eval_plots"
        )
        plot_pr_curve(
            y_test,
            anomaly_scores,
            eval_output_dir / "pr_curve.png",
        )
        plot_roc_curve(
            y_test,
            anomaly_scores,
            eval_output_dir / "roc_curve.png",
        )
    
        # [6/7] Generate SHAP Explainability
        print("\n[6/7] Generating SHAP explainability outputs...")
        sampled_anomalies = sample_anomalies(
            X_test,
            predictions,
            sample_size=config["shap_sample_size"],
            random_state=config["random_state"],
        )
        explainer = get_shap_explainer(
            detector.model,
            sampled_anomalies,
        )
        shap_values = compute_shap_values(
            explainer,
            sampled_anomalies,
        )
        shap_output_dir = (
            Path(config["output_dir"])
            / "shap_plots"
        )
        for idx in range(
            min(5, len(sampled_anomalies))
        ):
            plot_force(
                explainer,
                shap_values,
                sampled_anomalies,
                idx=idx,
                save_path=(
                    shap_output_dir
                    / f"force_plot_{idx}.png"
                ),
            )
        plot_summary(
            shap_values,
            sampled_anomalies,
            shap_output_dir / "shap_summary.png",
        )
        
        # [7/7] Save Reports and Outputs
        print("\n[7/7] Saving reports and flagged records...")
        report_output_dir = Path(
            config["output_dir"]
        )
        save_flagged_records(
            X_test.assign(Class=y_test.values),
            anomaly_scores,
            predictions,
            report_output_dir / "flagged_records.csv",
        )
        save_metrics_report(
            metrics,
            report_output_dir / "metrics_report.txt",
        )
        print("\n" + "=" * 60)
        print(" PIPELINE EXECUTED SUCCESSFULLY ")
        print("=" * 60)
    except FileNotFoundError as error:
        print(f"\n[ERROR] {error}")
        print(
            "\nPlease download the dataset from:\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place 'creditcard.csv' inside:\n"
            "data/raw/"
        )
    except Exception as error:
        print(f"\n[ERROR] Pipeline execution failed: {error}")

if __name__ == "__main__":
    main()