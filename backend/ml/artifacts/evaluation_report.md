# TRAFFIQ ML Model Evaluation Report

**Evaluation Strategy**: Held-out 20% Stratified Test Split (10,000 observations)  
**Leakage-Free Guarantee**: Contemporaneous metrics (`Traffic_Volume_veh_hr`, `Traffic_to_Capacity_Ratio`, `Average_Speed_kmph`) are strictly excluded.

## Summary Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.96%** |
| **Weighted F1** | **99.96%** |
| **Macro F1** | **99.94%** |
| **Weighted Precision** | **99.96%** |
| **Weighted Recall** | **99.96%** |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **High** | 99.91% | 99.91% | 99.91% | 1,054 |
| **Low** | 100.00% | 99.96% | 99.98% | 5,274 |
| **Medium** | 99.92% | 99.97% | 99.95% | 3,672 |

## Confusion Matrix

Labels: `['High', 'Low', 'Medium']`

```
[[1053    0    1]
 [   0 5272    2]
 [   1    0 3671]]
```
