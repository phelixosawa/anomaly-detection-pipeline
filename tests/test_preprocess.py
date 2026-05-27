from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.preprocess import (
    load_path,
    scale_features,
    split_data,
)

@pytest.fixture
def synthetic_dataframe():
    """
    Create a small synthetic dataframe for testing.
    Returns
    -------
    pd.DataFrame
        Synthetic fraud-like dataset.
    """
    np.random.seed(42)
    data = {
        "Time": np.random.randint(0, 100000, 100),
        "Amount": np.random.uniform(1, 500, 100),
        "Class": np.random.choice(
            [0, 1],
            size=100,
            p=[0.95, 0.05],
        ),
    }
    # Add PCA-like features V1-V28
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(100)
    return pd.DataFrame(data)

def test_load_path_returns_dataframe(
    synthetic_dataframe,
    tmp_path,
):
    """
    Test that load_path returns a dataframe
    with the correct number of columns.
    """
    test_csv_path = tmp_path / "test_creditcard.csv"
    synthetic_dataframe.to_csv(
        test_csv_path,
        index=False,
    )
    loaded_df = load_path(test_csv_path)
    assert isinstance(
        loaded_df,
        pd.DataFrame,
    )
    assert loaded_df.shape[1] == 31

def test_scale_features_zero_mean(
    synthetic_dataframe,
):
    """
    Test that scaled Amount and Time
    columns have near-zero mean.
    """
    scaled_df = scale_features(
        synthetic_dataframe
    )
    time_mean = scaled_df["Time"].mean()
    amount_mean = scaled_df["Amount"].mean()
    tolerance = 1e-6
    assert abs(time_mean) < tolerance
    assert abs(amount_mean) < tolerance

def test_split_data_returns_correct_outputs(
    synthetic_dataframe,
):
    """
    Test that split_data returns four outputs
    and training data is larger than test data.
    """
    X_train, X_test, y_train, y_test = split_data(
        synthetic_dataframe,
        test_size=0.2,
        random_state=42,
    )
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert len(X_train) > len(X_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)