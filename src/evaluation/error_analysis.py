import pandas as pd
import numpy as np

def analyze_errors(y_true, y_pred):
    df = pd.DataFrame({
        "true": y_true,
        "pred": y_pred
    })

    df["error"] = df["true"] != df["pred"]

    fp = df[(df["true"] == 0) & (df["pred"] == 1)]
    fn = df[(df["true"] == 1) & (df["pred"] == 0)]

    return {
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "error_rate": df["error"].mean()
    }