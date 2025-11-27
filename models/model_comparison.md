# Model Comparison: XGBoost vs Random Forest

## Accuracy Comparison

| Metric | XGBoost | Random Forest | Difference |
|--------|---------|---------------|------------|
| Test Accuracy | 0.9909 | 0.9932 | -0.0023 |
| F1 Score | 0.9909 | 0.9932 | -0.0023 |
| Precision | 0.9915 | 0.9935 | -0.0020 |
| Recall | 0.9909 | 0.9932 | -0.0023 |

## Speed Comparison

| Metric | XGBoost | Random Forest | Ratio |
|--------|---------|---------------|-------|
| Training Time | 4.33s | 0.82s | 0.19x |
| Tuning Time | 20.74s | 60.56s | 2.92x |
| Prediction Time | 0.0032s | 0.0089s | 2.78x |

## Memory Comparison

| Metric | XGBoost | Random Forest | Ratio |
|--------|---------|---------------|-------|
| Model Size | 5.07 MB | 4.76 MB | 0.94x |

## Cross-Validation Comparison

| Metric | XGBoost | Random Forest |
|--------|---------|---------------|
| CV Mean | 0.9903 | 0.9920 |
| CV Std | 0.0053 | 0.0042 |

## Summary

- **Accuracy Winner**: Random Forest
- **Speed Winner**: XGBoost
- **Memory Efficiency Winner**: Random Forest
- **Stability Winner**: Random Forest
- **Overall Recommendation**: Random Forest
