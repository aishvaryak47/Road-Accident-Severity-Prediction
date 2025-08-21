# src/utils.py
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_ohe_feature_names(pre: ColumnTransformer, cat_cols: List[str], num_cols: List[str]) -> List[str]:
    names: List[str] = []
    for name, transformer, cols in pre.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            ohe: OneHotEncoder = transformer.named_steps["encoder"]
            try:
                ohe_names = ohe.get_feature_names_out(cols)
            except Exception:
                ohe_names = [f"{c}_encoded" for c in cols]
            names.extend(ohe_names.tolist() if hasattr(ohe_names, "tolist") else list(ohe_names))
    return names

def extract_feature_importance(model, feat_names: List[str]) -> Optional[pd.DataFrame]:
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            return pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
        if hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim == 2:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            return pd.DataFrame({"feature": feat_names, "importance": coef}).sort_values("importance", ascending=False)
    except Exception:
        pass
    return None
