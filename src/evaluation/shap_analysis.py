import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_shap_analysis(pipeline_model, X_train, X_test, feature_names, results_dir):

    # Extract final trained model from pipeline
    model = pipeline_model.named_steps["model"]

    # Apply preprocessing (important!)
    X_train_transformed = pipeline_model[:-1].transform(X_train)
    X_test_transformed = pipeline_model[:-1].transform(X_test)

    # Use TreeExplainer for tree-based models
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_train_transformed)

    shap_values = explainer(X_test_transformed)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)

    save_path = os.path.join(results_dir, "shap_summary.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"SHAP summary saved to: {save_path}")

    # Bar plot (global importance)
    shap.plots.bar(shap_values)

    return shap_values