# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. Load Dataset
# =========================
dataset_path = r"C:\Users\selco\OneDrive\Desktop\road-accident-severity-prediction\data\rta_dataset.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

print("Loading dataset...")
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully!\n")
print(df.head())

# =========================
# 2. Preprocessing
# =========================
print("\nPreprocessing data...")
df = df.dropna()
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
print("Preprocessing completed!\n")

# =========================
# 3. Features & Target
# =========================
if "Accident_severity" not in df.columns:
    raise KeyError("Target column 'Accident_severity' not found in dataset!")

X = df.drop("Accident_severity", axis=1)
y = df["Accident_severity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 4. Models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
print("Training models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\n{name} Results:")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =========================
# 5. Create results folder
# =========================
results_folder = r"C:\Users\selco\OneDrive\Desktop\road-accident-severity-prediction\results"
os.makedirs(results_folder, exist_ok=True)

# =========================
# 6. Model Comparison (Bar Chart)
# =========================
plt.figure(figsize=(6, 4))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
bar_chart_path = os.path.join(results_folder, "model_comparison.png")
plt.savefig(bar_chart_path, dpi=300)
print(f"\nModel comparison chart saved at: {bar_chart_path}")
plt.show()

# =========================
# 7. ROC Curve (Binary or Multi-class)
# =========================
y_classes = np.unique(y)
n_classes = len(y_classes)
y_test_bin = label_binarize(y_test, classes=y_classes)

plt.figure(figsize=(6, 4))
colors = cycle(['blue', 'green', 'orange', 'red', 'purple', 'cyan'])

if n_classes == 2:
    # Binary ROC
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
else:
    # Multi-class ROC
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            for i, color in zip(range(n_classes), colors):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
                plt.plot(fpr, tpr, color=color, linestyle='--',
                         label=f"{name} Class {y_classes[i]} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC-AUC Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

roc_chart_path = os.path.join(results_folder, "roc_curve.png")
plt.savefig(roc_chart_path, dpi=300)
print(f"ROC curve chart saved at: {roc_chart_path}")
plt.show()
