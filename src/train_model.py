# src/train_model.py
from typing import Dict
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def get_models(random_state: int) -> Dict[str, object]:
    models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "SVC": SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=random_state),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, objective="multi:softprob", random_state=random_state, tree_method="hist"
        )
    return models

def build_pipeline(preprocessor, clf, random_state: int):
    pipe = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=random_state)),
        ("clf", clf),
    ])
    return pipe

def train_models(preprocessor, models: Dict[str, object], X_train, y_train, random_state: int):
    pipelines = {}
    for name, clf in models.items():
        pipe = build_pipeline(preprocessor, clf, random_state)
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
    return pipelines
