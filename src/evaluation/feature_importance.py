import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names, save_path=None, top_n=15, title="Feature Importance"):
    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    df = df.sort_values("importance", ascending=False).head(top_n)

    plt.figure()
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance saved to: {save_path}")

    plt.close()  # important (avoid memory leaks)

    return df