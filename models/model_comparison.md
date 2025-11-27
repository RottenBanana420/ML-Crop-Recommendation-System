# Model Comparison Report: XGBoost vs Random Forest


## Executive Summary

**Comparison Date**: 2025-11-27 02:44:17

**Models Evaluated**:
- XGBoost Classifier
- Random Forest Classifier

**Key Findings**:
- **Selected Model**: Random Forest
- **Accuracy Difference**: 0.0023 (0.23%)
- **XGBoost Accuracy**: 0.9909
- **Random Forest Accuracy**: 0.9932
- **Statistical Significance**: McNemar's test p-value: 1.0000

**Selection Rationale**: Accuracy difference is small (0.23%). Selected based on balanced consideration of speed, size, and stability.

**Production Readiness**: ✓ PASSED



## Detailed Metrics Comparison

### Classification Metrics

| Metric | XGBoost | Random Forest | Difference | Winner |
|--------|---------|---------------|------------|--------|
| Test Accuracy | 0.9909 | 0.9932 | -0.0023 | Random Forest |
| Train Accuracy | 1.0000 | 1.0000 | +0.0000 | Tie |
| F1 Score (Weighted) | 0.9909 | 0.9932 | -0.0023 | Random Forest |
| Precision (Weighted) | 0.9915 | 0.9935 | -0.0020 | Random Forest |
| Recall (Weighted) | 0.9909 | 0.9932 | -0.0023 | Random Forest |
| Matthews Correlation Coefficient | 0.9905 | 0.9929 | -0.0024 | Random Forest |
| Cohen's Kappa | 0.9905 | 0.9929 | -0.0024 | Random Forest |

### Performance Metrics

| Metric | XGBoost | Random Forest | Ratio | Winner |
|--------|---------|---------------|-------|--------|
| Training Time | 4.33s | 0.82s | 0.19x | Random Forest |
| Prediction Time | 0.0032s | 0.0086s | 2.69x | XGBoost |
| P50 Latency | 0.0031s | 0.0085s | - | XGBoost |
| P95 Latency | 0.0036s | 0.0090s | - | XGBoost |
| P99 Latency | 0.0041s | 0.0092s | - | XGBoost |

### Memory Metrics

| Metric | XGBoost | Random Forest | Ratio | Winner |
|--------|---------|---------------|-------|--------|
| Model Size | 5.07 MB | 4.76 MB | 0.94x | Random Forest |



## Per-Class Performance Analysis

### Top 5 Crops Where XGBoost Outperforms Random Forest

| Crop | XGBoost F1 | Random Forest F1 | Difference |
|------|------------|------------------|------------|
| blackgram | 1.0000 | 0.9744 | +0.0256 |
| rice | 1.0000 | 0.9744 | +0.0256 |
| jute | 1.0000 | 0.9756 | +0.0244 |
| apple | 1.0000 | 1.0000 | +0.0000 |
| banana | 1.0000 | 1.0000 | +0.0000 |

### Top 5 Crops Where Random Forest Outperforms XGBoost

| Crop | Random Forest F1 | XGBoost F1 | Difference |
|------|------------------|------------|------------|
| lentil | 0.9744 | 0.9474 | +0.0270 |
| chickpea | 1.0000 | 0.9744 | +0.0256 |
| cotton | 1.0000 | 0.9756 | +0.0244 |
| kidneybeans | 1.0000 | 0.9756 | +0.0244 |
| mothbeans | 0.9756 | 0.9524 | +0.0232 |

### Crops with Lowest F1 Scores (Potential Improvement Areas)

| Crop | XGBoost F1 | Random Forest F1 | Better Model |
|------|------------|------------------|--------------|
| lentil | 0.9474 | 0.9744 | Random Forest |
| mothbeans | 0.9524 | 0.9756 | Random Forest |
| blackgram | 1.0000 | 0.9744 | XGBoost |
| chickpea | 0.9744 | 1.0000 | Random Forest |
| maize | 0.9744 | 0.9756 | Random Forest |



## Model Selection Rationale

### Selected Model: Random Forest

**Decision Criteria**:
Accuracy difference is small (0.23%). Selected based on balanced consideration of speed, size, and stability.

**Production Readiness Assessment**:
- Status: ✓ PASSED
- Minimum Accuracy Threshold (97.5%): ✓
- Per-Class F1 Threshold (95%): ✓
- Inference Latency (<100ms/1000 samples): ✓
- Model Size (<50MB): ✓

**Statistical Significance**:
The accuracy difference is not statistically significant (p=≥0.05).

**Deployment Recommendation**:
Deploy Random Forest to production. All production readiness criteria met.


## Cross-Validation Analysis

| Model | CV Mean | CV Std | CV Scores |
|-------|---------|--------|-----------|
| XGBoost | 0.9903 | 0.0053 | 0.9915, 0.9830, 0.9943, 0.9972, 0.9858 |
| Random Forest | 0.9920 | 0.0042 | 0.9943, 0.9886, 0.9972, 0.9943, 0.9858 |

## Visualizations

The following visualizations are available in the `comparison_visualizations/` directory:

1. **Radar Chart** (`radar_chart.png`): Multi-metric comparison across key performance indicators
2. **Per-Class Heatmap** (`per_class_heatmap.png`): F1 scores for all 22 crop types
3. **Confusion Matrices** (`confusion_matrices.png`): Side-by-side confusion matrix comparison
4. **Inference Time Distribution** (`inference_time_distribution.png`): Box plot of prediction latencies
5. **Feature Importance Correlation** (`feature_importance_correlation.png`): Correlation between feature rankings
6. **Performance-Efficiency Scatter** (`performance_efficiency_scatter.png`): Accuracy vs speed trade-off

## Conclusion

Based on comprehensive evaluation across accuracy, speed, memory efficiency, and stability metrics, **Random Forest** has been selected as the production model. This model meets all production readiness criteria and provides the best balance of performance characteristics for crop recommendation.

---

*Report generated on 2025-11-27 02:44:17*
