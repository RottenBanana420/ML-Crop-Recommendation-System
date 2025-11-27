"""
Comparison Report Generation Module

This module generates comprehensive HTML and Markdown reports for model comparison,
including executive summaries, detailed metrics, and selection rationale.
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def generate_executive_summary(comparison_data, selection_data):
    """
    Generate executive summary of model comparison.
    
    Args:
        comparison_data: Complete comparison data dictionary
        selection_data: Model selection results
        
    Returns:
        Executive summary as string
    """
    xgb_acc = comparison_data['accuracy_comparison']['xgboost']['test_accuracy']
    rf_acc = comparison_data['accuracy_comparison']['random_forest']['test_accuracy']
    
    winner = selection_data['selected_model']
    acc_diff = abs(xgb_acc - rf_acc)
    
    summary = f"""
## Executive Summary

**Comparison Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Models Evaluated**:
- XGBoost Classifier
- Random Forest Classifier

**Key Findings**:
- **Selected Model**: {winner}
- **Accuracy Difference**: {acc_diff:.4f} ({acc_diff*100:.2f}%)
- **XGBoost Accuracy**: {xgb_acc:.4f}
- **Random Forest Accuracy**: {rf_acc:.4f}
- **Statistical Significance**: {selection_data.get('statistical_significance', 'Not computed')}

**Selection Rationale**: {selection_data['selection_rationale']}

**Production Readiness**: {'✓ PASSED' if selection_data['production_ready'] else '✗ FAILED'}
"""
    return summary


def generate_detailed_metrics_table(comparison_data):
    """
    Generate detailed metrics comparison table.
    
    Args:
        comparison_data: Complete comparison data dictionary
        
    Returns:
        Markdown table as string
    """
    xgb = comparison_data['accuracy_comparison']['xgboost']
    rf = comparison_data['accuracy_comparison']['random_forest']
    
    table = """
## Detailed Metrics Comparison

### Classification Metrics

| Metric | XGBoost | Random Forest | Difference | Winner |
|--------|---------|---------------|------------|--------|
"""
    
    metrics = [
        ('Test Accuracy', xgb['test_accuracy'], rf['test_accuracy']),
        ('Train Accuracy', xgb['train_accuracy'], rf['train_accuracy']),
        ('F1 Score (Weighted)', xgb['f1_weighted'], rf['f1_weighted']),
        ('Precision (Weighted)', xgb['precision_weighted'], rf['precision_weighted']),
        ('Recall (Weighted)', xgb['recall_weighted'], rf['recall_weighted']),
    ]
    
    for metric_name, xgb_val, rf_val in metrics:
        diff = xgb_val - rf_val
        winner = 'XGBoost' if xgb_val > rf_val else 'Random Forest' if rf_val > xgb_val else 'Tie'
        table += f"| {metric_name} | {xgb_val:.4f} | {rf_val:.4f} | {diff:+.4f} | {winner} |\n"
    
    # Add advanced metrics if available
    if 'mcc' in xgb:
        table += f"| Matthews Correlation Coefficient | {xgb['mcc']:.4f} | {rf['mcc']:.4f} | {xgb['mcc']-rf['mcc']:+.4f} | {'XGBoost' if xgb['mcc'] > rf['mcc'] else 'Random Forest'} |\n"
    
    if 'cohens_kappa' in xgb:
        table += f"| Cohen's Kappa | {xgb['cohens_kappa']:.4f} | {rf['cohens_kappa']:.4f} | {xgb['cohens_kappa']-rf['cohens_kappa']:+.4f} | {'XGBoost' if xgb['cohens_kappa'] > rf['cohens_kappa'] else 'Random Forest'} |\n"
    
    # Performance metrics
    xgb_speed = comparison_data['speed_comparison']['xgboost']
    rf_speed = comparison_data['speed_comparison']['random_forest']
    
    table += """
### Performance Metrics

| Metric | XGBoost | Random Forest | Ratio | Winner |
|--------|---------|---------------|-------|--------|
"""
    
    table += f"| Training Time | {xgb_speed['training_time']:.2f}s | {rf_speed['training_time']:.2f}s | {rf_speed['training_time']/xgb_speed['training_time']:.2f}x | {'XGBoost' if xgb_speed['training_time'] < rf_speed['training_time'] else 'Random Forest'} |\n"
    table += f"| Prediction Time | {xgb_speed['prediction_time']:.4f}s | {rf_speed['prediction_time']:.4f}s | {rf_speed['prediction_time']/xgb_speed['prediction_time']:.2f}x | {'XGBoost' if xgb_speed['prediction_time'] < rf_speed['prediction_time'] else 'Random Forest'} |\n"
    
    # Add latency percentiles if available
    if 'latency_percentiles' in xgb_speed:
        table += f"| P50 Latency | {xgb_speed['latency_percentiles']['p50']:.4f}s | {rf_speed['latency_percentiles']['p50']:.4f}s | - | {'XGBoost' if xgb_speed['latency_percentiles']['p50'] < rf_speed['latency_percentiles']['p50'] else 'Random Forest'} |\n"
        table += f"| P95 Latency | {xgb_speed['latency_percentiles']['p95']:.4f}s | {rf_speed['latency_percentiles']['p95']:.4f}s | - | {'XGBoost' if xgb_speed['latency_percentiles']['p95'] < rf_speed['latency_percentiles']['p95'] else 'Random Forest'} |\n"
        table += f"| P99 Latency | {xgb_speed['latency_percentiles']['p99']:.4f}s | {rf_speed['latency_percentiles']['p99']:.4f}s | - | {'XGBoost' if xgb_speed['latency_percentiles']['p99'] < rf_speed['latency_percentiles']['p99'] else 'Random Forest'} |\n"
    
    # Memory metrics
    xgb_mem = comparison_data['memory_comparison']['xgboost']
    rf_mem = comparison_data['memory_comparison']['random_forest']
    
    table += """
### Memory Metrics

| Metric | XGBoost | Random Forest | Ratio | Winner |
|--------|---------|---------------|-------|--------|
"""
    
    table += f"| Model Size | {xgb_mem['model_size_mb']:.2f} MB | {rf_mem['model_size_mb']:.2f} MB | {rf_mem['model_size_mb']/xgb_mem['model_size_mb']:.2f}x | {'XGBoost' if xgb_mem['model_size_mb'] < rf_mem['model_size_mb'] else 'Random Forest'} |\n"
    
    return table


def generate_per_class_analysis(comparison_data, crop_names):
    """
    Generate per-class performance analysis.
    
    Args:
        comparison_data: Complete comparison data dictionary
        crop_names: List of crop names
        
    Returns:
        Per-class analysis as string
    """
    xgb_per_class = comparison_data['per_class_comparison']['xgboost']
    rf_per_class = comparison_data['per_class_comparison']['random_forest']
    
    analysis = """
## Per-Class Performance Analysis

### Top 5 Crops Where XGBoost Outperforms Random Forest

| Crop | XGBoost F1 | Random Forest F1 | Difference |
|------|------------|------------------|------------|
"""
    
    # Calculate differences
    differences = []
    for i, crop in enumerate(crop_names):
        diff = xgb_per_class['f1_scores'][i] - rf_per_class['f1_scores'][i]
        differences.append((crop, xgb_per_class['f1_scores'][i], rf_per_class['f1_scores'][i], diff))
    
    # Sort by difference (XGBoost advantage)
    differences.sort(key=lambda x: x[3], reverse=True)
    
    for crop, xgb_f1, rf_f1, diff in differences[:5]:
        analysis += f"| {crop} | {xgb_f1:.4f} | {rf_f1:.4f} | {diff:+.4f} |\n"
    
    analysis += """
### Top 5 Crops Where Random Forest Outperforms XGBoost

| Crop | Random Forest F1 | XGBoost F1 | Difference |
|------|------------------|------------|------------|
"""
    
    # Sort by difference (Random Forest advantage)
    differences.sort(key=lambda x: x[3])
    
    for crop, xgb_f1, rf_f1, diff in differences[:5]:
        analysis += f"| {crop} | {rf_f1:.4f} | {xgb_f1:.4f} | {-diff:+.4f} |\n"
    
    analysis += """
### Crops with Lowest F1 Scores (Potential Improvement Areas)

| Crop | XGBoost F1 | Random Forest F1 | Better Model |
|------|------------|------------------|--------------|
"""
    
    # Find crops with lowest F1 scores
    all_f1s = [(crop, xgb_per_class['f1_scores'][i], rf_per_class['f1_scores'][i]) 
               for i, crop in enumerate(crop_names)]
    all_f1s.sort(key=lambda x: min(x[1], x[2]))
    
    for crop, xgb_f1, rf_f1 in all_f1s[:5]:
        better = 'XGBoost' if xgb_f1 > rf_f1 else 'Random Forest'
        analysis += f"| {crop} | {xgb_f1:.4f} | {rf_f1:.4f} | {better} |\n"
    
    return analysis


def generate_selection_rationale(selection_data):
    """
    Generate detailed selection rationale.
    
    Args:
        selection_data: Model selection results
        
    Returns:
        Selection rationale as string
    """
    rationale = f"""
## Model Selection Rationale

### Selected Model: {selection_data['selected_model']}

**Decision Criteria**:
{selection_data['selection_rationale']}

**Production Readiness Assessment**:
- Status: {'✓ PASSED' if selection_data['production_ready'] else '✗ FAILED'}
- Minimum Accuracy Threshold (97.5%): {'✓' if selection_data.get('meets_accuracy_threshold', False) else '✗'}
- Per-Class F1 Threshold (95%): {'✓' if selection_data.get('meets_per_class_threshold', False) else '✗'}
- Inference Latency (<100ms/1000 samples): {'✓' if selection_data.get('meets_latency_threshold', False) else '✗'}
- Model Size (<50MB): {'✓' if selection_data.get('meets_size_threshold', False) else '✗'}

**Statistical Significance**:
{selection_data.get('statistical_analysis', 'Not computed')}

**Deployment Recommendation**:
{selection_data.get('deployment_recommendation', 'Deploy the selected model to production.')}
"""
    
    return rationale


def generate_markdown_report(comparison_data, selection_data, crop_names, output_path):
    """
    Generate comprehensive Markdown report.
    
    Args:
        comparison_data: Complete comparison data dictionary
        selection_data: Model selection results
        crop_names: List of crop names
        output_path: Path to save the report
    """
    report = f"""# Model Comparison Report: XGBoost vs Random Forest

{generate_executive_summary(comparison_data, selection_data)}

{generate_detailed_metrics_table(comparison_data)}

{generate_per_class_analysis(comparison_data, crop_names)}

{generate_selection_rationale(selection_data)}

## Cross-Validation Analysis

| Model | CV Mean | CV Std | CV Scores |
|-------|---------|--------|-----------|
| XGBoost | {comparison_data['cross_validation_comparison']['xgboost']['cv_mean']:.4f} | {comparison_data['cross_validation_comparison']['xgboost']['cv_std']:.4f} | {', '.join([f'{s:.4f}' for s in comparison_data['cross_validation_comparison']['xgboost']['cv_scores']])} |
| Random Forest | {comparison_data['cross_validation_comparison']['random_forest']['cv_mean']:.4f} | {comparison_data['cross_validation_comparison']['random_forest']['cv_std']:.4f} | {', '.join([f'{s:.4f}' for s in comparison_data['cross_validation_comparison']['random_forest']['cv_scores']])} |

## Visualizations

The following visualizations are available in the `comparison_visualizations/` directory:

1. **Radar Chart** (`radar_chart.png`): Multi-metric comparison across key performance indicators
2. **Per-Class Heatmap** (`per_class_heatmap.png`): F1 scores for all 22 crop types
3. **Confusion Matrices** (`confusion_matrices.png`): Side-by-side confusion matrix comparison
4. **Inference Time Distribution** (`inference_time_distribution.png`): Box plot of prediction latencies
5. **Feature Importance Correlation** (`feature_importance_correlation.png`): Correlation between feature rankings
6. **Performance-Efficiency Scatter** (`performance_efficiency_scatter.png`): Accuracy vs speed trade-off

## Conclusion

Based on comprehensive evaluation across accuracy, speed, memory efficiency, and stability metrics, **{selection_data['selected_model']}** has been selected as the production model. This model meets all production readiness criteria and provides the best balance of performance characteristics for crop recommendation.

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Markdown report saved to: {output_path}")


def generate_html_report(comparison_data, selection_data, crop_names, output_path, viz_dir):
    """
    Generate interactive HTML report.
    
    Args:
        comparison_data: Complete comparison data dictionary
        selection_data: Model selection results
        crop_names: List of crop names
        output_path: Path to save the report
        viz_dir: Directory containing visualizations
    """
    # Convert viz_dir to relative path from output_path
    viz_dir = Path(viz_dir)
    output_path = Path(output_path)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .winner {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .viz-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .viz-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .status-passed {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-failed {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 Crop Recommendation Model Comparison</h1>
        <p>XGBoost vs Random Forest - Comprehensive Analysis</p>
        <p style="font-size: 0.9em; opacity: 0.9;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="highlight">
            <h3>Selected Model: {selection_data['selected_model']}</h3>
            <p><strong>Status:</strong> <span class="status-badge {'status-passed' if selection_data['production_ready'] else 'status-failed'}">
                {'✓ PRODUCTION READY' if selection_data['production_ready'] else '✗ NOT READY'}
            </span></p>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">XGBoost Accuracy</div>
            <div class="metric-value">{comparison_data['accuracy_comparison']['xgboost']['test_accuracy']:.2%}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Random Forest Accuracy</div>
            <div class="metric-value">{comparison_data['accuracy_comparison']['random_forest']['test_accuracy']:.2%}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Accuracy Difference</div>
            <div class="metric-value">{abs(comparison_data['accuracy_comparison']['xgboost']['test_accuracy'] - comparison_data['accuracy_comparison']['random_forest']['test_accuracy']):.2%}</div>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Selection Rationale</h2>
        <p>{selection_data['selection_rationale']}</p>
    </div>

    <div class="section">
        <h2>📈 Performance Visualizations</h2>
        
        <h3>Multi-Metric Comparison</h3>
        <div class="viz-container">
            <img src="{viz_dir.name}/radar_chart.png" alt="Radar Chart">
        </div>
        
        <h3>Per-Class Performance</h3>
        <div class="viz-container">
            <img src="{viz_dir.name}/per_class_heatmap.png" alt="Per-Class Heatmap">
        </div>
        
        <h3>Confusion Matrices</h3>
        <div class="viz-container">
            <img src="{viz_dir.name}/confusion_matrices.png" alt="Confusion Matrices">
        </div>
        
        <h3>Inference Time Distribution</h3>
        <div class="viz-container">
            <img src="{viz_dir.name}/inference_time_distribution.png" alt="Inference Time">
        </div>
        
        <h3>Feature Importance Correlation</h3>
        <div class="viz-container">
            <img src="{viz_dir.name}/feature_importance_correlation.png" alt="Feature Importance">
        </div>
        
        <h3>Performance-Efficiency Trade-off</h3>
        <div class="viz-container">
            <img src="{viz_dir.name}/performance_efficiency_scatter.png" alt="Performance-Efficiency">
        </div>
    </div>

    <div class="section">
        <h2>📋 Detailed Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>XGBoost</th>
                <th>Random Forest</th>
                <th>Difference</th>
            </tr>
            <tr>
                <td>Test Accuracy</td>
                <td>{comparison_data['accuracy_comparison']['xgboost']['test_accuracy']:.4f}</td>
                <td>{comparison_data['accuracy_comparison']['random_forest']['test_accuracy']:.4f}</td>
                <td>{comparison_data['accuracy_comparison']['xgboost']['test_accuracy'] - comparison_data['accuracy_comparison']['random_forest']['test_accuracy']:+.4f}</td>
            </tr>
            <tr>
                <td>F1 Score</td>
                <td>{comparison_data['accuracy_comparison']['xgboost']['f1_weighted']:.4f}</td>
                <td>{comparison_data['accuracy_comparison']['random_forest']['f1_weighted']:.4f}</td>
                <td>{comparison_data['accuracy_comparison']['xgboost']['f1_weighted'] - comparison_data['accuracy_comparison']['random_forest']['f1_weighted']:+.4f}</td>
            </tr>
            <tr>
                <td>Prediction Time</td>
                <td>{comparison_data['speed_comparison']['xgboost']['prediction_time']:.4f}s</td>
                <td>{comparison_data['speed_comparison']['random_forest']['prediction_time']:.4f}s</td>
                <td>{comparison_data['speed_comparison']['xgboost']['prediction_time'] - comparison_data['speed_comparison']['random_forest']['prediction_time']:+.4f}s</td>
            </tr>
            <tr>
                <td>Model Size</td>
                <td>{comparison_data['memory_comparison']['xgboost']['model_size_mb']:.2f} MB</td>
                <td>{comparison_data['memory_comparison']['random_forest']['model_size_mb']:.2f} MB</td>
                <td>{comparison_data['memory_comparison']['xgboost']['model_size_mb'] - comparison_data['memory_comparison']['random_forest']['model_size_mb']:+.2f} MB</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>✅ Production Readiness Checklist</h2>
        <ul>
            <li>{'✓' if selection_data.get('meets_accuracy_threshold', False) else '✗'} Accuracy ≥ 97.5%</li>
            <li>{'✓' if selection_data.get('meets_per_class_threshold', False) else '✗'} All crops F1 ≥ 95%</li>
            <li>{'✓' if selection_data.get('meets_latency_threshold', False) else '✗'} Inference latency < 100ms/1000 samples</li>
            <li>{'✓' if selection_data.get('meets_size_threshold', False) else '✗'} Model size < 50MB</li>
        </ul>
    </div>

    <div class="section">
        <h2>🎓 Conclusion</h2>
        <p>Based on comprehensive evaluation across accuracy, speed, memory efficiency, and stability metrics, 
        <strong>{selection_data['selected_model']}</strong> has been selected as the production model. 
        This model meets all production readiness criteria and provides the best balance of performance 
        characteristics for crop recommendation.</p>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to: {output_path}")
