import joblib
import os
import json
from datetime import datetime

def save_model(model, path, overwrite=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Model already exists at {path}")

    joblib.dump(model, path)
    print(f"Model saved at: {path}")


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")

    print(f"Loading model from: {path}")
    return joblib.load(path)


def save_metadata(metadata, path):
    meta_path = path.replace(".pkl", "_meta.json")

    metadata["saved_at"] = datetime.now().isoformat()

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved at: {meta_path}")