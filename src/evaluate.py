# src/evaluate.py
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from .visualize import plot_confusion_matrix, plot_roc_curves

def get_probabilities(pipe, X_test):
    y_proba = None
    try:
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)
        elif hasattr(pipe.named_steps["clf"], "decision_function"):
            dec = pipe.decision_function(X_test)
            from scipy.special import expit
            if dec.ndim == 1:
                y_proba = expit(dec)
            else:
                ex = np.exp(dec - dec.max(axis=1, keepdims=True))
                y_proba = ex / ex.sum(axis=1, keepdims=True)
    except Exception:
        pass
    return y_proba

def evaluate_and_save(name: str, pipe, X_test, y_test, class_names: List[str], out_dir: str) -> Tuple[Dict, Dict]:
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    report_txt = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    cm_path = f"{out_dir}/{name}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path, title=f"{name} – Confusion Matrix")

    with open(f"{out_dir}/{name}_classification_report.txt", "w") as f:
        f.write(report_txt)

    y_proba = get_probabilities(pipe, X_test)
    return (
        {"Model": name, "Accuracy": acc, "Precision_w": pr, "Recall_w": rc, "F1_w": f1},
        {name: y_proba} if y_proba is not None else {}
    )

def save_metrics_table(metrics_rows: List[Dict], out_dir: str) -> str:
    df = pd.DataFrame(metrics_rows)
    path = f"{out_dir}/model_comparison_metrics.csv"
    df.to_csv(path, index=False)
    return path
