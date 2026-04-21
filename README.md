# Machine Learning Pipeline for Classification (Life Expectancy Dataset)

## Overview

This project implements a complete end-to-end machine learning pipeline to classify countries based on life expectancy levels.

The goal was not just to train a model, but to build a **structured, modular, and reproducible system** that covers:

* Data loading and preprocessing
* Model training and comparison
* Hyperparameter tuning
* Evaluation with multiple metrics
* Explainability using SHAP and feature importance
* Saving all experiment outputs for analysis

---

## Motivation

I built this project to move beyond isolated ML models and understand how real-world pipelines are structured.

Instead of focusing on a single algorithm, I wanted to:

* Compare multiple models under the same pipeline
* Prevent data leakage using proper design
* Evaluate models beyond accuracy
* Understand model behavior using interpretability tools

---

## Project Structure

```id="proj-structure"
ML-Classification-Pipeline-Life-Expectancy/
│
├── data/
│   └── LifeExpectancy.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── results/
│   ├── best_model.pkl
│   ├── model_results.csv
│   ├── evaluation_summary.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   └── shap_summary.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── main.py
│   │
│   ├── evaluation/
│   │   ├── error_analysis.py
│   │   ├── feature_importance.py
│   │   ├── model_metrics.py
│   │   └── shap_analysis.py
│   │
│   └── utils/
│       └── io.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Pipeline Design

Each model is trained using a structured pipeline to ensure consistency and avoid data leakage:

* Missing value imputation (median strategy)
* Feature scaling (only for relevant models)
* Model training within a pipeline
* Hyperparameter tuning using RandomizedSearchCV
* Stratified cross-validation (5 folds, shuffled, seeded)

---

## Models Implemented

* K-Nearest Neighbors (KNN)
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Extra Trees Classifier

---

## Evaluation Strategy

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score (weighted)
* ROC-AUC score
* Optimal classification threshold (F1-based)

Artifacts generated:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Classification Report
* Evaluation summary (JSON)

---

## Explainability

To understand model decisions:

* Feature importance for tree-based models
* SHAP analysis for global and local explanations
* Error analysis to inspect misclassifications

---

## Key Insights

* Tree-based models (Random Forest, XGBoost, Extra Trees) performed significantly better than linear and distance-based models
* Hyperparameter tuning had a strong impact, especially for ensemble models
* Scaling was essential for KNN and Logistic Regression, but unnecessary for tree-based models
* Cross-validation was critical for stable model selection
* Using an optimized threshold improved F1-score over the default 0.5

---

## How to Run

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the pipeline:

   ```
   python src/main.py
   ```

3. Outputs will be saved in the `results/` directory.

---

## Future Improvements

* Add feature engineering (domain-specific features)
* Try LightGBM and CatBoost
* Integrate experiment tracking (MLflow / W&B)
* Deploy model using FastAPI or Streamlit
* Add automated validation and monitoring

---

## Final Thoughts

This project demonstrates how to design machine learning systems that are not only accurate, but also reproducible, interpretable, and production-aware.

The focus on pipelines, cross-validation, and explainability reflects practices used in real-world machine learning workflows.
