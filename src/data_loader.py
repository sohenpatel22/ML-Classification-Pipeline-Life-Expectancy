import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_data(
    url,
    local_path="data/LifeExpectancy.csv",
    test_size=0.3,
    random_state=42
):
    # Download if not exists
    if not os.path.exists(local_path):
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_csv(local_path, index=False)
        print("Dataset cached locally.")
    else:
        print("Loading dataset from local cache")
        df = pd.read_csv(local_path)

    # Fix column names
    df.columns = df.columns.str.strip()

    # Target creation
    median_val = df["Life expectancy"].median()
    df["target_binary"] = (df["Life expectancy"] >= median_val).astype(int)

    # Features and labels
    X = df.drop(columns=["Life expectancy", "target_binary"])
    y = df["target_binary"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test