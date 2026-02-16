import pandas as pd
from sklearn.datasets import fetch_openml

def load_dataset():
    data = fetch_openml(name="heart-disease", version=1, as_frame=True)
    df = data.frame.copy()

    target_col = "class" if "class" in df.columns else "target"

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Convert target safely
    y = y.astype(float).astype(int)

    feature_names = list(X.columns)
    return X, y, feature_names

