from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)

def validate_schema(df : pd.DataFrame) -> None:
    """
    Validate that the input DataFrame contains the required schema.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.
    Returns
    -------
    None
    Raises
    ------
    ValueError
        If required columns are missing.
    """

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset schema validation failed. "
            f"Missing columns: {missing_columns}"
        )
    
def load_path(path: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset from CSV.
    Parameters
    ----------
    path : str
        Path to the CSV dataset.
    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the schema is invalid.
    """
    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            f"Download dataset from Kaggle and place "
            f"'careditcard.csv' inside data/raw/.\n"
            f"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    
    print(f"[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    validate_schema(df)
    print(f"[INFO] Dataset loaded successfully.")
    print(f"[INFO] Dataset shape: {df.shape}")
    return df

def scale_features(df : pd.DataFrame) -> pd.DataFrame:
    """
    Scale the 'Time' and 'Amount' columns using StandardScaler.
    Notes
    -----
    V1-V28 are already PCA-transformed and scaled in original dataset,
    only Time and Amount need scaling.
    
    No rows are dropped because the dataset contains no missing values.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    Returns
    -------
    pd.DataFrame
        Scaled dataframe copy.
    """
    scaled_df = df.copy()
    scaler = StandardScaler()
    columns_to_scale = ["Time", "Amount"]
    scaled_df[columns_to_scale] = scaler.fit_transform(
        scaled_df[columns_to_scale]
    )
    print("[INFO] Scaling complete for Time and Amount columns.")
    return scaled_df

def split_data(
        df : pd.DataFrame,
        test_size: float,
        random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing sets.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Random seed for reproducibility.
    Returns
    -------
    Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    ]
        X_train, X_test, y_train, y_test
    """

    x = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print("[INFO] Train/test split completed.")
    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    """
    Quick standalone test for preprocessing module.
    """
    SAMPLE_PATH = "data/raw/creditcard.csv"
    try:
        dataframe = load_path(SAMPLE_PATH)
        scaled_dataframe = scale_features(dataframe)
        X_train, X_test, y_train, y_test = split_data(
            scaled_dataframe, test_size=0.2, random_state=42
        )
        print("[INFO] Preprocessing module executed successfully.")
    except Exception as error:
        print(f"[ERROR] {error}")