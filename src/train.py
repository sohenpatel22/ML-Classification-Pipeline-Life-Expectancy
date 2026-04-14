import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_split_data
from preprocess import preprocess_data

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score


def find_best_k(X_train, y_train):
    """
    Tune K using cross-validation (ONLY on training data).
    """
    k_range = range(1, 101)
    best_k = None
    best_score = 0

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    return best_k, best_score


def main():
    # Load + split (already includes target creation)
    X_train, X_test, y_train, y_test = load_and_split_data(
        url="https://raw.githubusercontent.com/Sabaae/Dataset/main/LifeExpectancy.csv",
        random_state=0 
    )

    # Scale
    X_train_scaled, X_test_scaled, imputer, scaler = preprocess_data(X_train, X_test)

    # Tune K (ONLY on training data)
    best_k, best_cv_score = find_best_k(X_train_scaled, y_train)

    print(f"Best K: {best_k}")
    print(f"Best CV Accuracy: {best_cv_score:.4f}")

    # Train final model
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set (NO CV here)
    y_pred = model.predict(X_test_scaled)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\nFinal Test Performance:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")


if __name__ == "__main__":
    main()