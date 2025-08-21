# =========================
# MODEL SUMMARY TABLE
# =========================

# Example: Replace these with your actual results from main.py
results = {
    "Logistic Regression": 0.78,
    "Decision Tree": 0.82,
    "Random Forest": 0.87
}

# Example: Replace with your actual ROC-AUC values
roc_auc_scores = {
    "Logistic Regression": 0.82,
    "Decision Tree": 0.85,
    "Random Forest": 0.91
}

# Find best model
best_model = max(results, key=results.get)

# Print Summary Table
print("\n========== MODEL SUMMARY ==========")
print(f"Best model based on accuracy: {best_model} ({results[best_model]:.2f})\n")

# Table Header
print(f"{'MODEL':<25}{'ACCURACY':<10}{'ROC-AUC':<10}")
print("-" * 45)

# Table Rows
for model in results:
    print(f"{model:<25}{results[model]:<10.2f}{roc_auc_scores[model]:<10.2f}")
