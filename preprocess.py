"""
Stratified train / test split.
Optionally drop the raw Time column to avoid temporal leakage.
No scaling or SMOTE here – that lives in the pipeline.
"""
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(df: pd.DataFrame,
                    target: str      = "Class",
                    test_size: float = 0.20,
                    random_state: int = 42,
                    drop_time: bool   = True):
    work_df = df.copy()
    if drop_time and "Time" in work_df.columns:
        work_df = work_df.drop(columns=["Time"])

    X = work_df.drop(columns=[target])
    y = work_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        stratify=y, random_state=random_state
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test