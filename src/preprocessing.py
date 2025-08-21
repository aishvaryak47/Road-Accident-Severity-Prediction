# src/preprocessing.py
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def load_data(csv_path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV. Found: {list(df.columns)[:12]}...")
    df = df.dropna(subset=[target_col]).copy()
    return df

def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in df.columns if c != target_col and df[c].dtype == 'object']
    num_cols = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]
    others = [c for c in df.columns if c not in cat_cols + num_cols + [target_col]]
    cat_cols += others
    return cat_cols, num_cols

def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder",  OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])
    pre = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])
    return pre

def split_data(df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
    y = df[target_col].astype("category")
    class_names = list(y.cat.categories.astype(str))
    y_codes = y.cat.codes  # integers 0..n-1
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_codes, test_size=test_size, random_state=random_state, stratify=y_codes
    )
    return X_train, X_test, y_train, y_test, class_names
