# model_building/train.py
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import joblib

import mlflow
import mlflow.sklearn


PREP_DIR = Path("artifacts/prepared")
MODEL_DIR = Path("artifacts/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Prefer prepared.csv; fall back to tourism_clean.csv
CANDIDATE_DATASETS = [
    PREP_DIR / "prepared.csv",
    PREP_DIR / "tourism_clean.csv",
]

MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def find_dataset_path() -> Path:
    for p in CANDIDATE_DATASETS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Prepared dataset not found. Expected one of: {[str(p) for p in CANDIDATE_DATASETS]}. "
        f"Run data/data_prep.py first."
    )


def infer_target_column(df: pd.DataFrame) -> str:
    """
    Tries to infer target column. Prefer environment variable TARGET_COL if set.
    Otherwise chooses the last numeric column (common in many templates).
    """
    env_target = os.getenv("TARGET_COL")
    if env_target and env_target in df.columns:
        return env_target

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found to infer a target. Set TARGET_COL env var to a valid column name.")
    return numeric_cols[-1]


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def run_experiments(X_train, X_test, y_train, y_test, preprocessor):
    """
    Runs multiple candidate models, logs to MLflow, returns the best (by RMSE).
    """
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "VisitWithUs-Tourism-Experiments"))

    candidates = [
        ("LinearRegression", LinearRegression(), {}),
        ("RandomForest", RandomForestRegressor(random_state=42), {"model__n_estimators": 200, "model__max_depth": None}),
        ("RandomForest_depth10", RandomForestRegressor(random_state=42), {"model__n_estimators": 300, "model__max_depth": 10}),
    ]

    best = {"name": None, "rmse": float("inf"), "model": None, "metrics": None}

    for name, estimator, params in candidates:
        with mlflow.start_run(run_name=name):
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            if params:
                pipe.set_params(**params)

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))

            mlflow.log_param("candidate", name)
            for k, v in params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # Log model
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            metrics = {"rmse": rmse, "r2": r2}
            if rmse < best["rmse"]:
                best.update({"name": name, "rmse": rmse, "model": pipe, "metrics": metrics})

            print(f"[INFO] {name}: rmse={rmse:.4f}, r2={r2:.4f}")

    return best


def main():
    dataset_path = find_dataset_path()
    df = pd.read_csv(dataset_path)
    print(f"[INFO] Training using dataset: {dataset_path} shape={df.shape}")

    target_col = infer_target_column(df)
    print(f"[INFO] Target column: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Basic guardrails: ensure target numeric
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"Target column '{target_col}' must be numeric. Got dtype={y.dtype}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(os.getenv("TEST_SIZE", "0.2")), random_state=42
    )

    preprocessor = build_preprocessor(X)

    # Experimentation BEFORE deployment happens here:
    best = run_experiments(X_train, X_test, y_train, y_test, preprocessor)

    if best["model"] is None:
        raise RuntimeError("No model trained successfully.")

    # Save best model locally for deployment step
    joblib.dump(best["model"], MODEL_PATH)

    out_metrics = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "best_run": best["name"],
        "metrics": best["metrics"],
        "dataset_used": str(dataset_path),
        "target_col": target_col,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    print(f"[INFO] Saved best model: {MODEL_PATH}")
    print(f"[INFO] Saved metrics: {METRICS_PATH}")
    print(f"[INFO] Best: {best['name']} rmse={best['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    main()

