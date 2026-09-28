"""
TRAFFIQ ML Training Pipeline
Trains leakage-free supervised classification models on pre-trip traffic features.
Compares Random Forest and Gradient Boosting, reporting metrics and feature importances.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from ml.preprocess import (
    TARGET_COL,
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    get_location_lag_defaults,
    get_location_infrastructure_defaults
)

from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _ML_DIR.parent
DATA_PATH = str(_BACKEND_DIR / "data" / "traffic_processed.csv")
ARTIFACTS_DIR = str(_ML_DIR / "artifacts")

def train_and_evaluate():
    print(f"Loading processed data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Save location infrastructure and lag defaults for pre-trip UI assistance
    infra_defaults = get_location_infrastructure_defaults(df)
    lag_defaults = get_location_lag_defaults(df)
    
    with open(os.path.join(ARTIFACTS_DIR, "location_infrastructure.json"), "w") as f:
        json.dump(infra_defaults, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "location_lag_defaults.json"), "w") as f:
        json.dump(lag_defaults, f, indent=2)
    print("Saved location infrastructure and lag defaults.")

    X = df[ALL_MODEL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    
    class_names = sorted(y.unique().tolist()) # ['High', 'Low', 'Medium']
    print(f"Target classes: {class_names}")
    print(f"Feature count: {len(ALL_MODEL_FEATURES)} ({len(CATEGORICAL_FEATURES)} categorical, {len(NUMERICAL_FEATURES)} numerical)")

    # 80/20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    preprocessor = build_preprocessor()

    # Model 1: Random Forest
    print("\n--- Training Random Forest Classifier ---")
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average="weighted")
    rf_macro_f1 = f1_score(y_test, rf_pred, average="macro")
    print(f"Random Forest -> Accuracy: {rf_acc:.4f} | Weighted F1: {rf_f1:.4f} | Macro F1: {rf_macro_f1:.4f}")

    # Model 2: HistGradientBoosting
    print("\n--- Training Gradient Boosting Classifier ---")
    # Note: HistGradientBoosting handles dense array from OneHotEncoder directly
    gb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    gb_pipeline.fit(X_train, y_train)
    gb_pred = gb_pipeline.predict(X_test)
    
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_f1 = f1_score(y_test, gb_pred, average="weighted")
    gb_macro_f1 = f1_score(y_test, gb_pred, average="macro")
    print(f"Gradient Boosting -> Accuracy: {gb_acc:.4f} | Weighted F1: {gb_f1:.4f} | Macro F1: {gb_macro_f1:.4f}")

    # Extract One-Hot Encoded feature names for explainability
    ohe_step = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_features = list(ohe_step.get_feature_names_out(CATEGORICAL_FEATURES))
    full_transformed_features = encoded_cat_features + NUMERICAL_FEATURES

    # Compute Feature Importances from Random Forest
    rf_model = rf_pipeline.named_steps["classifier"]
    importances = rf_model.feature_importances_
    feat_imp = sorted(
        zip(full_transformed_features, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Also aggregate importances back to raw feature groups
    raw_feature_importance = {}
    for feat_name, imp in feat_imp:
        # Match back to original feature name
        matched_raw = None
        for raw_col in ALL_MODEL_FEATURES:
            if feat_name.startswith(f"{raw_col}_") or feat_name == raw_col:
                matched_raw = raw_col
                break
        if not matched_raw:
            matched_raw = feat_name
        raw_feature_importance[matched_raw] = raw_feature_importance.get(matched_raw, 0.0) + float(imp)
        
    sorted_raw_importances = sorted(raw_feature_importance.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Top 10 Feature Importances (Raw Aggregated) ---")
    for name, imp in sorted_raw_importances[:10]:
        print(f"  {name:<28}: {imp*100:.2f}%")

    # Select Best Model based on Weighted F1
    best_name = "RandomForest" if rf_f1 >= gb_f1 else "GradientBoosting"
    best_pipeline = rf_pipeline if best_name == "RandomForest" else gb_pipeline
    print(f"\nBest Model Selected: {best_name}")

    # Save artifacts
    joblib.dump(best_pipeline, os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
    joblib.dump(rf_pipeline, os.path.join(ARTIFACTS_DIR, "rf_model.joblib"))
    joblib.dump(gb_pipeline, os.path.join(ARTIFACTS_DIR, "gb_model.joblib"))
    
    # Save metadata & feature list
    metadata = {
        "best_model": best_name,
        "classes": class_names,
        "features": {
            "all": ALL_MODEL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
            "transformed": full_transformed_features
        },
        "performance": {
            "RandomForest": {
                "accuracy": float(rf_acc),
                "weighted_f1": float(rf_f1),
                "macro_f1": float(rf_macro_f1)
            },
            "GradientBoosting": {
                "accuracy": float(gb_acc),
                "weighted_f1": float(gb_f1),
                "macro_f1": float(gb_macro_f1)
            }
        },
        "feature_importances": [
            {"feature": k, "importance": round(v, 4), "percentage": round(v * 100, 2)}
            for k, v in sorted_raw_importances
        ]
    }
    
    with open(os.path.join(ARTIFACTS_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save test dataset partition for standalone evaluation script
    test_data = X_test.copy()
    test_data[TARGET_COL] = y_test
    test_data.to_csv(os.path.join(ARTIFACTS_DIR, "test_dataset.csv"), index=False)
    
    print("\nSaved all training artifacts to ml/artifacts/ successfully!")

if __name__ == "__main__":
    train_and_evaluate()
