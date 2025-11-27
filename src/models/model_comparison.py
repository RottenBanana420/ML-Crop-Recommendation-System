"""
Model Comparison: XGBoost vs Random Forest

This module performs comprehensive comparison of XGBoost and Random Forest models
for crop recommendation, including multi-metric evaluation, per-class analysis,
statistical significance testing, and intelligent model selection.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix
)

# Import visualization and reporting modules
from comparison_visualizations import save_all_visualizations
from comparison_report import generate_markdown_report, generate_html_report


def get_model_size(model_path):
    """
    Get model file size in MB.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Model size in MB
    """
    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def measure_prediction_time(model, X_test, n_runs=100):
    """
    Measure prediction time with detailed statistics.
    
    Args:
        model: Trained model
        X_test: Test features
        n_runs: Number of runs for averaging
        
    Returns:
        Dictionary with mean, percentiles, and all times
    """
    times = []
    for _ in range(n_runs):
        start = time.time()
        model.predict(X_test)
        end = time.time()
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
        'all_times': times
    }


def calculate_per_class_metrics(y_true, y_pred, num_classes=22):
    """
    Calculate per-class precision, recall, and F1 scores.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with per-class metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    
    return {
        'precision_scores': precision.tolist(),
        'recall_scores': recall.tolist(),
        'f1_scores': f1.tolist(),
        'support': support.tolist()
    }


def calculate_advanced_metrics(y_true, y_pred):
    """
    Calculate advanced metrics (MCC, Cohen's Kappa).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with advanced metrics
    """
    return {
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohens_kappa': cohen_kappa_score(y_true, y_pred)
    }


def perform_mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Perform McNemar's test to determine if accuracy difference is statistically significant.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        
    Returns:
        Dictionary with test results
    """
    # Create contingency table
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    # McNemar's table
    n01 = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
    n10 = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    
    # Perform test using chi-square approximation with continuity correction
    try:
        # McNemar's test statistic with continuity correction
        if n01 + n10 == 0:
            # No disagreement, models are identical
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'n01': int(n01),
                'n10': int(n10)
            }
        
        statistic = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
        
        # Calculate p-value using chi-square distribution (df=1)
        # For chi-square with df=1, we can use the normal approximation
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(statistic, df=1)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'n01': int(n01),
            'n10': int(n10)
        }
    except Exception as e:
        return {
            'statistic': None,
            'p_value': None,
            'significant': None,
            'n01': int(n01),
            'n10': int(n10),
            'error': str(e)
        }


def select_production_model(comparison_data, xgb_model, rf_model, 
                           xgb_pred, rf_pred, y_test, models_dir):
    """
    Select the best model for production based on multi-criteria analysis.
    
    Args:
        comparison_data: Complete comparison data
        xgb_model: XGBoost model
        rf_model: Random Forest model
        xgb_pred: XGBoost predictions
        rf_pred: Random Forest predictions
        y_test: True labels
        models_dir: Directory to save production model
        
    Returns:
        Selection data dictionary
    """
    xgb_acc = comparison_data['accuracy_comparison']['xgboost']['test_accuracy']
    rf_acc = comparison_data['accuracy_comparison']['random_forest']['test_accuracy']
    
    xgb_speed = comparison_data['speed_comparison']['xgboost']['prediction_time']
    rf_speed = comparison_data['speed_comparison']['random_forest']['prediction_time']
    
    xgb_size = comparison_data['memory_comparison']['xgboost']['model_size_mb']
    rf_size = comparison_data['memory_comparison']['random_forest']['model_size_mb']
    
    xgb_cv_std = comparison_data['cross_validation_comparison']['xgboost']['cv_std']
    rf_cv_std = comparison_data['cross_validation_comparison']['random_forest']['cv_std']
    
    # Production readiness thresholds
    MIN_ACCURACY = 0.975  # 97.5%
    MIN_PER_CLASS_F1 = 0.95  # 95%
    MAX_LATENCY_1000 = 0.1  # 100ms for 1000 samples
    MAX_SIZE_MB = 50  # 50MB
    
    # Check production readiness
    xgb_per_class = comparison_data['per_class_comparison']['xgboost']
    rf_per_class = comparison_data['per_class_comparison']['random_forest']
    
    xgb_min_f1 = min(xgb_per_class['f1_scores'])
    rf_min_f1 = min(rf_per_class['f1_scores'])
    
    # Scale latency to 1000 samples
    test_size = len(y_test)
    xgb_latency_1000 = xgb_speed * (1000 / test_size)
    rf_latency_1000 = rf_speed * (1000 / test_size)
    
    # Check each model's production readiness
    xgb_ready = (
        xgb_acc >= MIN_ACCURACY and
        xgb_min_f1 >= MIN_PER_CLASS_F1 and
        xgb_latency_1000 < MAX_LATENCY_1000 and
        xgb_size < MAX_SIZE_MB
    )
    
    rf_ready = (
        rf_acc >= MIN_ACCURACY and
        rf_min_f1 >= MIN_PER_CLASS_F1 and
        rf_latency_1000 < MAX_LATENCY_1000 and
        rf_size < MAX_SIZE_MB
    )
    
    # Perform statistical significance test
    mcnemar_result = perform_mcnemar_test(y_test, xgb_pred, rf_pred)
    
    # Decision logic
    acc_diff = abs(xgb_acc - rf_acc)
    
    # Primary criterion: Accuracy (if difference > 0.5%)
    if acc_diff > 0.005:
        selected_model = 'XGBoost' if xgb_acc > rf_acc else 'Random Forest'
        rationale = f"Selected based on superior accuracy ({acc_diff*100:.2f}% difference, statistically {'significant' if mcnemar_result.get('significant') else 'not significant'})."
    # Secondary criterion: Speed (if accuracy tied within 0.1%)
    elif acc_diff <= 0.001:
        selected_model = 'XGBoost' if xgb_speed < rf_speed else 'Random Forest'
        speed_diff = abs(xgb_speed - rf_speed)
        rationale = f"Accuracy is equivalent (difference {acc_diff*100:.3f}%). Selected based on faster inference speed ({speed_diff*1000:.2f}ms difference)."
    # Tertiary: Balanced decision
    else:
        # Calculate composite score
        # Normalize metrics (lower is better for speed, size, cv_std)
        xgb_score = xgb_acc - (xgb_speed * 0.1) - (xgb_size * 0.001) - (xgb_cv_std * 0.5)
        rf_score = rf_acc - (rf_speed * 0.1) - (rf_size * 0.001) - (rf_cv_std * 0.5)
        
        selected_model = 'XGBoost' if xgb_score > rf_score else 'Random Forest'
        rationale = f"Accuracy difference is small ({acc_diff*100:.2f}%). Selected based on balanced consideration of speed, size, and stability."
    
    # Determine production readiness
    production_ready = xgb_ready if selected_model == 'XGBoost' else rf_ready
    
    # Save production model
    production_model = xgb_model if selected_model == 'XGBoost' else rf_model
    production_model_path = models_dir / 'production_model.pkl'
    joblib.dump(production_model, production_model_path)
    
    # Create selection data
    selection_data = {
        'selected_model': selected_model,
        'selection_rationale': rationale,
        'production_ready': production_ready,
        'meets_accuracy_threshold': (xgb_acc if selected_model == 'XGBoost' else rf_acc) >= MIN_ACCURACY,
        'meets_per_class_threshold': (xgb_min_f1 if selected_model == 'XGBoost' else rf_min_f1) >= MIN_PER_CLASS_F1,
        'meets_latency_threshold': (xgb_latency_1000 if selected_model == 'XGBoost' else rf_latency_1000) < MAX_LATENCY_1000,
        'meets_size_threshold': (xgb_size if selected_model == 'XGBoost' else rf_size) < MAX_SIZE_MB,
        'statistical_significance': f"McNemar's test p-value: {mcnemar_result.get('p_value', 'N/A'):.4f}" if mcnemar_result.get('p_value') else 'Test failed',
        'statistical_analysis': f"The accuracy difference is {'statistically significant' if mcnemar_result.get('significant') else 'not statistically significant'} (p={'<0.05' if mcnemar_result.get('significant') else '≥0.05'}).",
        'deployment_recommendation': f"{'Deploy' if production_ready else 'Do NOT deploy'} {selected_model} to production. {'All' if production_ready else 'Some'} production readiness criteria {'met' if production_ready else 'not met'}.",
        'production_model_path': str(production_model_path),
        'mcnemar_test': mcnemar_result
    }
    
    return selection_data


def compare_models():
    """Compare XGBoost and Random Forest models with comprehensive analysis."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'models'
    data_dir = project_root / 'data'
    viz_dir = models_dir / 'comparison_visualizations'
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON: XGBoost vs Random Forest")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    xgb_model = joblib.load(models_dir / 'xgboost_model.pkl')
    rf_model = joblib.load(models_dir / 'random_forest_model.pkl')
    
    # Load metrics
    with open(models_dir / 'xgboost_full_metrics.json', 'r') as f:
        xgb_metrics = json.load(f)
    
    with open(models_dir / 'full_metrics.json', 'r') as f:
        rf_metrics = json.load(f)
    
    # Load test data
    test_data = pd.read_csv(data_dir / 'processed' / 'preprocessed_test.csv')
    X_test = test_data.drop('label_encoded', axis=1)
    y_test = test_data['label_encoded'].values
    
    # Load label encoder to get crop names
    label_encoder = joblib.load(models_dir / 'label_encoder.pkl')
    crop_names = label_encoder.classes_.tolist()
    
    # Get predictions
    print("Generating predictions...")
    xgb_pred = xgb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Initialize comparison results
    comparison = {
        'accuracy_comparison': {},
        'speed_comparison': {},
        'memory_comparison': {},
        'cross_validation_comparison': {},
        'feature_importance_comparison': {},
        'per_class_comparison': {},
        'summary': {}
    }
    
    # ==================== Accuracy Comparison ====================
    print("\n" + "-" * 80)
    print("ACCURACY COMPARISON")
    print("-" * 80)
    
    # Calculate advanced metrics
    xgb_advanced = calculate_advanced_metrics(y_test, xgb_pred)
    rf_advanced = calculate_advanced_metrics(y_test, rf_pred)
    
    comparison['accuracy_comparison'] = {
        'xgboost': {
            'test_accuracy': xgb_metrics['test_accuracy'],
            'train_accuracy': xgb_metrics['train_accuracy'],
            'accuracy_gap': xgb_metrics['accuracy_gap'],
            'f1_weighted': xgb_metrics['test_f1_weighted'],
            'precision_weighted': xgb_metrics['test_precision_weighted'],
            'recall_weighted': xgb_metrics['test_recall_weighted'],
            'mcc': xgb_advanced['mcc'],
            'cohens_kappa': xgb_advanced['cohens_kappa']
        },
        'random_forest': {
            'test_accuracy': rf_metrics['test_accuracy'],
            'train_accuracy': rf_metrics['train_accuracy'],
            'accuracy_gap': rf_metrics['accuracy_gap'],
            'f1_weighted': rf_metrics['test_f1_weighted'],
            'precision_weighted': rf_metrics['test_precision_weighted'],
            'recall_weighted': rf_metrics['test_recall_weighted'],
            'mcc': rf_advanced['mcc'],
            'cohens_kappa': rf_advanced['cohens_kappa']
        }
    }
    
    print(f"XGBoost Test Accuracy:        {xgb_metrics['test_accuracy']:.4f}")
    print(f"Random Forest Test Accuracy:  {rf_metrics['test_accuracy']:.4f}")
    print(f"Difference:                   {xgb_metrics['test_accuracy'] - rf_metrics['test_accuracy']:.4f}")
    print()
    print(f"XGBoost MCC:                  {xgb_advanced['mcc']:.4f}")
    print(f"Random Forest MCC:            {rf_advanced['mcc']:.4f}")
    print(f"XGBoost Cohen's Kappa:        {xgb_advanced['cohens_kappa']:.4f}")
    print(f"Random Forest Cohen's Kappa:  {rf_advanced['cohens_kappa']:.4f}")
    
    # ==================== Per-Class Analysis ====================
    print("\n" + "-" * 80)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("-" * 80)
    
    xgb_per_class = calculate_per_class_metrics(y_test, xgb_pred)
    rf_per_class = calculate_per_class_metrics(y_test, rf_pred)
    
    comparison['per_class_comparison'] = {
        'xgboost': xgb_per_class,
        'random_forest': rf_per_class
    }
    
    print(f"XGBoost Min F1:               {min(xgb_per_class['f1_scores']):.4f}")
    print(f"XGBoost Mean F1:              {np.mean(xgb_per_class['f1_scores']):.4f}")
    print(f"Random Forest Min F1:         {min(rf_per_class['f1_scores']):.4f}")
    print(f"Random Forest Mean F1:        {np.mean(rf_per_class['f1_scores']):.4f}")
    
    # ==================== Speed Comparison ====================
    print("\n" + "-" * 80)
    print("SPEED COMPARISON")
    print("-" * 80)
    
    # Measure prediction times with detailed statistics
    print("Measuring prediction times (100 runs)...")
    xgb_timing = measure_prediction_time(xgb_model, X_test)
    rf_timing = measure_prediction_time(rf_model, X_test)
    
    comparison['speed_comparison'] = {
        'xgboost': {
            'training_time': xgb_metrics['training_time'],
            'tuning_time': xgb_metrics['tuning_time'],
            'prediction_time': xgb_timing['mean'],
            'latency_percentiles': {
                'p50': xgb_timing['p50'],
                'p95': xgb_timing['p95'],
                'p99': xgb_timing['p99']
            },
            'total_time': xgb_metrics['training_time'] + xgb_metrics['tuning_time']
        },
        'random_forest': {
            'training_time': rf_metrics['training_time'],
            'tuning_time': rf_metrics['tuning_time'],
            'prediction_time': rf_timing['mean'],
            'latency_percentiles': {
                'p50': rf_timing['p50'],
                'p95': rf_timing['p95'],
                'p99': rf_timing['p99']
            },
            'total_time': rf_metrics['training_time'] + rf_metrics['tuning_time']
        }
    }
    
    print(f"XGBoost Prediction Time:      {xgb_timing['mean']:.4f}s (p95: {xgb_timing['p95']:.4f}s)")
    print(f"Random Forest Prediction Time:{rf_timing['mean']:.4f}s (p95: {rf_timing['p95']:.4f}s)")
    print(f"Speedup:                      {rf_timing['mean'] / xgb_timing['mean']:.2f}x")
    
    # ==================== Memory Comparison ====================
    print("\n" + "-" * 80)
    print("MEMORY COMPARISON")
    print("-" * 80)
    
    xgb_size = get_model_size(models_dir / 'xgboost_model.pkl')
    rf_size = get_model_size(models_dir / 'random_forest_model.pkl')
    
    comparison['memory_comparison'] = {
        'xgboost': {
            'model_size_mb': xgb_size
        },
        'random_forest': {
            'model_size_mb': rf_size
        }
    }
    
    print(f"XGBoost Model Size:           {xgb_size:.2f} MB")
    print(f"Random Forest Model Size:     {rf_size:.2f} MB")
    print(f"Size Ratio (RF/XGB):          {rf_size / xgb_size:.2f}x")
    
    # ==================== Cross-Validation Comparison ====================
    print("\n" + "-" * 80)
    print("CROSS-VALIDATION COMPARISON")
    print("-" * 80)
    
    comparison['cross_validation_comparison'] = {
        'xgboost': {
            'cv_mean': xgb_metrics['cv_mean'],
            'cv_std': xgb_metrics['cv_std'],
            'cv_scores': xgb_metrics['cv_scores']
        },
        'random_forest': {
            'cv_mean': rf_metrics['cv_mean'],
            'cv_std': rf_metrics['cv_std'],
            'cv_scores': rf_metrics['cv_scores']
        }
    }
    
    print(f"XGBoost CV Mean:              {xgb_metrics['cv_mean']:.4f} ± {xgb_metrics['cv_std']:.4f}")
    print(f"Random Forest CV Mean:        {rf_metrics['cv_mean']:.4f} ± {rf_metrics['cv_std']:.4f}")
    
    # ==================== Feature Importance Comparison ====================
    print("\n" + "-" * 80)
    print("FEATURE IMPORTANCE COMPARISON (Top 5)")
    print("-" * 80)
    
    # Load feature importance
    with open(models_dir / 'xgboost_feature_importance.json', 'r') as f:
        xgb_importance = json.load(f)
    
    with open(models_dir / 'feature_importance.json', 'r') as f:
        rf_importance = json.load(f)
    
    comparison['feature_importance_comparison'] = {
        'xgboost': {
            'top_5_features': list(zip(xgb_importance['features'][:5], 
                                      xgb_importance['importances'][:5]))
        },
        'random_forest': {
            'top_5_features': list(zip(rf_importance['features'][:5], 
                                      rf_importance['importances'][:5]))
        }
    }
    
    print("\nXGBoost Top 5 Features:")
    for i, (feat, imp) in enumerate(zip(xgb_importance['features'][:5], 
                                        xgb_importance['importances'][:5]), 1):
        print(f"  {i}. {feat}: {imp:.4f}")
    
    print("\nRandom Forest Top 5 Features:")
    for i, (feat, imp) in enumerate(zip(rf_importance['features'][:5], 
                                        rf_importance['importances'][:5]), 1):
        print(f"  {i}. {feat}: {imp:.4f}")
    
    # ==================== Model Selection ====================
    print("\n" + "=" * 80)
    print("INTELLIGENT MODEL SELECTION")
    print("=" * 80)
    
    selection_data = select_production_model(
        comparison, xgb_model, rf_model, xgb_pred, rf_pred, y_test, models_dir
    )
    
    print(f"\nSelected Model:               {selection_data['selected_model']}")
    print(f"Production Ready:             {'✓ YES' if selection_data['production_ready'] else '✗ NO'}")
    print(f"\nRationale: {selection_data['selection_rationale']}")
    print(f"\nStatistical Analysis: {selection_data['statistical_analysis']}")
    print(f"\nProduction Model Saved:       {selection_data['production_model_path']}")
    
    # ==================== Generate Visualizations ====================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Get confusion matrices
    xgb_cm = confusion_matrix(y_test, xgb_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)
    
    save_all_visualizations(
        comparison, xgb_metrics, rf_metrics,
        xgb_cm, rf_cm,
        xgb_timing['all_times'], rf_timing['all_times'],
        xgb_importance, rf_importance,
        crop_names, viz_dir
    )
    
    # ==================== Generate Reports ====================
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    
    # Save JSON
    comparison_json_path = models_dir / 'model_comparison.json'
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"JSON comparison saved to: {comparison_json_path}")
    
    # Save selection data
    selection_json_path = models_dir / 'production_model_metadata.json'
    with open(selection_json_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        selection_save = {}
        for k, v in selection_data.items():
            if isinstance(v, (np.bool_, np.integer, np.floating)):
                selection_save[k] = v.item()
            elif isinstance(v, dict):
                # Handle nested dicts
                selection_save[k] = {
                    dk: dv.item() if isinstance(dv, (np.bool_, np.integer, np.floating)) else dv
                    for dk, dv in v.items()
                }
            else:
                selection_save[k] = v
        json.dump(selection_save, f, indent=2)
    print(f"Selection metadata saved to: {selection_json_path}")
    
    # Generate Markdown report
    generate_markdown_report(comparison, selection_data, crop_names, 
                            models_dir / 'model_comparison.md')
    
    # Generate HTML report
    generate_html_report(comparison, selection_data, crop_names,
                        models_dir / 'comparison_report.html', viz_dir)
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nFinal Decision: {selection_data['selected_model']} selected for production")
    print(f"Production Ready: {'✓ YES' if selection_data['production_ready'] else '✗ NO'}")
    print("\nGenerated Outputs:")
    print(f"  - Production Model: {selection_data['production_model_path']}")
    print(f"  - JSON Report: {comparison_json_path}")
    print(f"  - Markdown Report: {models_dir / 'model_comparison.md'}")
    print(f"  - HTML Report: {models_dir / 'comparison_report.html'}")
    print(f"  - Visualizations: {viz_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    compare_models()
