# Customer Churn Prediction - Machine Learning Project

## Project Overview

This project builds a machine learning system to predict customer churn in a telecom dataset. The objective is to identify customers likely to leave so that retention strategies can be applied early.

The pipeline includes:
- Data cleaning and feature engineering
- Handling categorical and numerical variables
- Imbalanced dataset handling
- Multiple machine learning model comparison
- Threshold tuning for better recall and precision trade-off
- Evaluation using multiple performance metrics

---

## Dataset

- Telecom churn dataset (~3333 samples)
- Target variable: Churn (Yes/No converted to 1/0)

Class distribution:
- Non-Churn: ~2850
- Churn: ~483

---

## Project Pipeline

### Data Loading
- CSV ingestion using pandas

### Data Cleaning
- Removed redundant usage columns
- Engineered features:
  - Total_calls
  - Total_charges

### Feature Engineering

Numerical features:
- Imputation using mean
- Scaling using StandardScaler

Categorical features:
- Imputation using most frequent value
- Encoding using OneHotEncoder

Implemented using ColumnTransformer and Pipeline.

---

## Train-Test Split

- Stratified split to preserve class distribution

---

## Models Trained

- Logistic Regression
- Decision Tree Classifier
- Bagging Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

Accuracy is not used as the primary metric due to class imbalance.

---

## Imbalanced Learning Strategy

- Class weighting where applicable
- Threshold tuning using probability outputs
- Focus on recall for churn detection

---

## Threshold Tuning

Predictions are based on probability thresholds:

y_pred = (y_prob >= threshold)

Threshold is adjusted to balance precision and recall.

---

## Model Performance Summary

Logistic Regression:
- Accuracy: 0.85
- Precision: 0.54
- Recall: 0.19
- F1 Score: 0.28
- ROC-AUC: 0.82

AdaBoost:
- Accuracy: 0.29
- Precision: 0.17
- Recall: 0.96
- F1 Score: 0.29
- ROC-AUC: 0.89

Bagging Classifier:
- Accuracy: 0.91
- Precision: 0.67
- Recall: 0.80
- F1 Score: 0.73
- ROC-AUC: 0.90

Decision Tree:
- Accuracy: 0.91
- Precision: 0.71
- Recall: 0.75
- F1 Score: 0.73
- ROC-AUC: 0.85

Gradient Boosting:
- Accuracy: 0.91
- Precision: 0.71
- Recall: 0.74
- F1 Score: 0.72
- ROC-AUC: 0.93

---

## Best Model

Gradient Boosting Classifier

Reason:
- Highest ROC-AUC
- Balanced precision and recall
- Strong generalization performance
- Stable probability outputs

---

## Key Insights

- Accuracy is misleading for imbalanced datasets
- ROC-AUC is more reliable for model comparison
- Threshold tuning is essential
- Ensemble methods outperform single models
- Recall is critical for churn detection

---

## Future Improvements

- Hyperparameter tuning (GridSearch / Optuna)
- LightGBM / XGBoost implementation
- Cost-sensitive learning
- Model deployment using API
- Monitoring pipeline

---

## Conclusion

This project demonstrates that model evaluation should focus on trade-offs between precision and recall rather than accuracy alone.
