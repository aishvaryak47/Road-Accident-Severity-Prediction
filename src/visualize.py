# src/visualize.py
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(cm, class_names: List[str], out_path: str, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted label", ylabel="True label", title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_roc_curves(y_test, y_proba_dict: Dict[str, np.ndarray], classes: List[str], out_path: str):
    fig, ax = plt.subplots(figsize=(7,6))
    n_classes = len(classes)
    for model_name, y_proba in y_proba_dict.items():
        try:
            if y_proba.ndim == 1 or (hasattr(y_proba, "shape") and y_proba.shape[1] == 1):
                RocCurveDisplay.from_predictions(y_test, y_proba, name=model_name, ax=ax)
            else:
                y_bin = label_binarize(y_test, classes=list(range(n_classes)))
                RocCurveDisplay.from_predictions(y_true=y_bin.ravel(), y_pred=y_proba.ravel(),
                                                 name=f"{model_name} (OvR macro)", ax=ax)
        except Exception:
            pass
    ax.set_title("ROC Curves (One-vs-Rest)")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_model_comparison_bar(metrics_df, out_path: str):
    import numpy as np
    import matplotlib.pyplot as plt
    models = metrics_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - 1.5*width, metrics_df["Accuracy"], width, label="Accuracy")
    ax.bar(x - 0.5*width, metrics_df["Precision_w"], width, label="Precision_w")
    ax.bar(x + 0.5*width, metrics_df["Recall_w"], width, label="Recall_w")
    ax.bar(x + 1.5*width, metrics_df["F1_w"], width, label="F1_w")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
