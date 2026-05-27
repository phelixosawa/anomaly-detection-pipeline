# End-to-End Anomaly Detection Pipeline with Explainability

An offline-first machine learning pipeline for detecting fraudulent credit card transactions using Isolation Forest and SHAP explainability.
---

## Project Overview

This project is a modular anomaly detection system built in Python using the public Kaggle Credit Card Fraud Detection dataset.

The pipeline:

- Loads and validates transaction data
- Applies preprocessing and feature scaling
- Trains an Isolation Forest anomaly detector
- Scores and flags suspicious transactions
- Evaluates detection performance using imbalance-aware metrics
- Explains anomalies using SHAP
- Saves all outputs as reproducible artifacts

The entire workflow runs with a single command:

```bash
python run_pipeline.py
```
---

## Features

- Modular `src/` architecture
- Config-driven pipeline using `config.yaml`
- Isolation Forest anomaly detection
- Precision-Recall and ROC evaluation plots
- SHAP explainability for detected anomalies
- Model persistence with Joblib
- Automated artifact generation
- Unit testing with Pytest
- Offline-first reproducibility

---

## Dataset

Source: Kaggle — Credit Card Fraud Detection Dataset
:contentReference[oaicite:0]{index=0}

### Dataset Characteristics

- ~285,000 transactions
- 31 columns
- Highly imbalanced:
  - Normal transactions: ~99.83%
  - Fraudulent transactions: ~0.17%
- No missing values
- Features `V1-V28` are PCA-transformed

### Expected Dataset Location

```text
data/raw/creditcard.csv
```
---

## Results

| Model | Precision | Recall | F1 Score | AUC-PR |
|---|---|---|---|---|
| Isolation Forest | TBD | TBD | TBD | TBD |

> Replace placeholder values after running the full pipeline.

---

## Example SHAP Explainability Output

### SHAP Summary Plot

![SHAP Summary Plot](outputs/shap_plots/shap_summary.png)

The SHAP summary plot highlights which features contribute most strongly to anomaly predictions across flagged transactions.

---

## Evaluation Outputs

The pipeline automatically generates:

- Precision-Recall Curve
- ROC Curve
- SHAP Force Plots
- SHAP Summary Plot
- Metrics Report
- Flagged Transaction CSV
- Serialized Isolation Forest Model

---

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:phelixosawa/anomaly-detection-pipeline.git

cd anomaly-detection-pipeline
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
Download the dataset manually from Kaggle:
:contentReference[oaicite:1]{index=1}
Place the file here:
```text
data/raw/creditcard.csv
```

---

## How to Run

Execute the full pipeline:

```bash
python run_pipeline.py
```

---

## Running Tests

Run preprocessing unit tests:

```bash
pytest tests/
```

---

## Project Structure

```text
anomaly-detection-pipeline/
│
├── data/
│   └── raw/
│       └── creditcard.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling_sandbox.ipynb
│
├── outputs/
│   ├── eval_plots/
│   ├── models/
│   ├── shap_plots/
│   ├── flagged_records.csv
│   └── metrics_report.txt
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── evaluate.py
│   ├── explain.py
│   └── report.py
│
├── tests/
│   └── test_preprocess.py
│
├── config.yaml
├── requirements.txt
├── run_pipeline.py
└── README.md
```

---

## Design Decisions

### Why Isolation Forest?

Isolation Forest is:
- lightweight
- efficient on large datasets
- well-suited for unsupervised anomaly detection
- effective for highly imbalanced fraud scenarios

It also integrates well with SHAP explainability.

---

### Why AUC-PR Instead of Accuracy?
Accuracy is misleading for highly imbalanced datasets.
A model predicting every transaction as normal would still achieve ~99.83% accuracy.
This project therefore prioritizes:
- Precision
- Recall
- F1 Score
- AUC-PR
- AUC-ROC

AUC-PR is especially important because it better reflects minority-class detection quality.

---

### Why SHAP Instead of LIME?

SHAP provides:
- consistent feature attributions
- local explanations
- global explanations
- strong support for tree-based models

It is also widely adopted in production ML explainability workflows.

---

### Why Offline-First?

This project intentionally avoids:
- cloud dependencies
- APIs
- web serving
- orchestration tooling

to maximize:
- reproducibility
- simplicity
- portability
- recruiter accessibility

---

## Limitations and Future Work

Potential future improvements include:

- Local Outlier Factor (LOF) comparison model
- Autoencoder-based anomaly detection
- Streamlit visualization dashboard
- Hyperparameter tuning experiments
- Additional anomaly explainability techniques
- CI/CD integration for automated testing

---

## Example Outputs

After execution, the pipeline generates:

```text
outputs/
├── eval_plots/
│   ├── pr_curve.png
│   └── roc_curve.png
├── models/
│   └── isolation_forest.joblib
├── shap_plots/
│   ├── force_plot_0.png
│   ├── force_plot_1.png
│   ├── force_plot_2.png
│   ├── force_plot_3.png
│   ├── force_plot_4.png
│   └── shap_summary.png
├── flagged_records.csv
└── metrics_report.txt
```

---

## License

This project is intended for educational and portfolio purposes.

---

## Author
Built by Phelix Osawa using Python, Scikit-learn, SHAP, and modular ML engineering principles.