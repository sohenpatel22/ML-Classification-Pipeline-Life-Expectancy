import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    classification_report
)

def evaluate_model(model, X_test, y_test, results_dir):

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities (important!)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        print("Model does not support predict_proba")
        return

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure()
    disp.plot()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Confusion Matrix saved: {cm_path}")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_path = os.path.join(results_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    print(f"ROC Curve saved: {roc_path}")

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    pr_path = os.path.join(results_dir, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()

    print(f"PR Curve saved: {pr_path}")

    # 4. Classification Report
    report = classification_report(y_test, y_pred)

    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Classification report saved: {report_path}")

    return {
        "roc_auc": auc_score
    }