from sklearn.metrics import (accuracy_score,f1_score,precision_score, 
                             recall_score,roc_auc_score, classification_report,
                             confusion_matrix,precision_recall_curve) 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score 
import numpy as np



def eval_model(model, x_test, y_test, model_name='model'):

    y_prob = model.predict_proba(x_test)[:, 1]

    threshold = 0.3
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'F1 Score': f1,
        'Precision': prec,
        'Recall': rec,
        'ROC-AUC': roc
    }

    print(f"\n======== Model = {model_name} ========")
    for k, v in metrics.items():
        if k != 'Model':
            print(f"{k}: {v:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    print("\n====== Classification Report ======")
    print(classification_report(y_test, y_pred))

    print("\n====== Precision-Recall Curve ======")
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

def cross_validation(model, x, y, cv=5):
    scores = cross_val_score(model, x, y, cv=cv, scoring='f1')
    return scores.mean()