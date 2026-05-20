from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODELS_DIR, SEVERITY_ORDER, TABLES_DIR


FEATURES = [
    "state",
    "city",
    "hour_of_day",
    "rush_hour",
    "weekend",
    "night_time",
    "weather_conditions",
    "weather_category",
    "road_condition",
    "road_type",
    "vehicle_type",
    "traffic_control_presence",
    "junction_type",
    "lighting_conditions",
    "speed_limit_kmph",
    "driver_age",
    "driver_gender",
    "driver_license_status",
    "alcohol_involvement",
    "number_of_vehicles_involved",
    "number_of_casualties",
    "visibility_km",
    "visibility_risk",
    "temperature_c",
]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "int64", "float64"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def train_models(df: pd.DataFrame) -> dict[str, object]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    X = df[FEATURES].copy()
    y = df["severity"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = _build_preprocessor(X)
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=260,
                    max_depth=14,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1200)),
        ]
    )

    rf_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    rf_pred = rf_pipeline.predict(X_test)
    lr_pred = lr_pipeline.predict(X_test)

    results = {
        "random_forest": _metrics(y_test, rf_pred),
        "logistic_regression": _metrics(y_test, lr_pred),
        "rf_confusion_matrix": confusion_matrix(y_test, rf_pred, labels=SEVERITY_ORDER).tolist(),
        "lr_confusion_matrix": confusion_matrix(y_test, lr_pred, labels=SEVERITY_ORDER).tolist(),
        "classification_report_rf": classification_report(y_test, rf_pred, zero_division=0, output_dict=True),
    }

    feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = rf_pipeline.named_steps["model"].feature_importances_
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    feature_importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)

    with open(TABLES_DIR / "model_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest_severity_model.joblib")
    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_regression_baseline.joblib")
    X_test.assign(actual=y_test, predicted=rf_pred).to_csv(TABLES_DIR / "test_predictions.csv", index=False)

    return {
        "rf_pipeline": rf_pipeline,
        "lr_pipeline": lr_pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "rf_pred": rf_pred,
        "metrics": results,
        "feature_importance": feature_importance,
    }
