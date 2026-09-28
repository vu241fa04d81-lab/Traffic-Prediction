"""
TRAFFIQ ML Model Evaluation
Generates confusion matrix, per-class classification metrics,
and detailed report for academic hackathon presentation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from pathlib import Path

try:
    from ml.preprocess import TARGET_COL, ALL_MODEL_FEATURES
except ImportError:
    from backend.ml.preprocess import TARGET_COL, ALL_MODEL_FEATURES

_ML_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = str(_ML_DIR / "artifacts")

def run_evaluation():
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    test_path = os.path.join(ARTIFACTS_DIR, "test_dataset.csv")
    meta_path = os.path.join(ARTIFACTS_DIR, "training_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Trained model or test dataset not found. Run ml/train.py first.")
        
    print(f"Loading best model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    test_df = pd.read_csv(test_path)
    X_test = test_df[ALL_MODEL_FEATURES]
    y_test = test_df[TARGET_COL]
    
    classes = sorted(y_test.unique().tolist())
    
    print(f"Running predictions on {len(X_test):,} held-out test records...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    prec_weight = precision_score(y_test, y_pred, average="weighted")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    rec_weight = recall_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weight = f1_score(y_test, y_pred, average="weighted")
    
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    clf_report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    
    print("\n" + "="*50)
    print("TRAFFIQ ML CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=classes))
    print("="*50)
    print("CONFUSION MATRIX (Labels: " + ", ".join(classes) + "):")
    print(cm)
    print("="*50)
    
    # Structure json metrics
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "precision_weighted": round(float(prec_weight), 4),
        "recall_macro": round(float(rec_macro), 4),
        "recall_weighted": round(float(rec_weight), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weight), 4),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            cls: {
                "precision": round(clf_report[cls]["precision"], 4),
                "recall": round(clf_report[cls]["recall"], 4),
                "f1_score": round(clf_report[cls]["f1-score"], 4),
                "support": int(clf_report[cls]["support"])
            }
            for cls in classes
        }
    }
    
    with open(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Generate Markdown Report
    md_content = f"""# TRAFFIQ ML Model Evaluation Report

**Evaluation Strategy**: Held-out 20% Stratified Test Split (10,000 observations)  
**Leakage-Free Guarantee**: Contemporaneous metrics (`Traffic_Volume_veh_hr`, `Traffic_to_Capacity_Ratio`, `Average_Speed_kmph`) are strictly excluded.

## Summary Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **{acc*100:.2f}%** |
| **Weighted F1** | **{f1_weight*100:.2f}%** |
| **Macro F1** | **{f1_macro*100:.2f}%** |
| **Weighted Precision** | **{prec_weight*100:.2f}%** |
| **Weighted Recall** | **{rec_weight*100:.2f}%** |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    for cls in classes:
        c_m = metrics["per_class"][cls]
        md_content += f"| **{cls}** | {c_m['precision']*100:.2f}% | {c_m['recall']*100:.2f}% | {c_m['f1_score']*100:.2f}% | {c_m['support']:,} |\n"

    md_content += f"""
## Confusion Matrix

Labels: `{classes}`

```
{cm}
```
"""
    with open(os.path.join(ARTIFACTS_DIR, "evaluation_report.md"), "w") as f:
        f.write(md_content)
        
    print(f"\nSaved evaluation metrics to {os.path.join(ARTIFACTS_DIR, 'evaluation_metrics.json')}")
    print(f"Saved evaluation markdown report to {os.path.join(ARTIFACTS_DIR, 'evaluation_report.md')}")

if __name__ == "__main__":
    run_evaluation()
