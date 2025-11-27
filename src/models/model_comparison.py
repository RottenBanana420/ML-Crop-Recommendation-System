"""
Model Comparison: XGBoost vs Random Forest

This module compares the performance characteristics of XGBoost and Random Forest
models for crop recommendation, including accuracy, speed, and memory usage.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


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


def measure_prediction_time(model, X_test, n_runs=10):
    """
    Measure average prediction time.
    
    Args:
        model: Trained model
        X_test: Test features
        n_runs: Number of runs for averaging
        
    Returns:
        Average prediction time in seconds
    """
    times = []
    for _ in range(n_runs):
        start = time.time()
        model.predict(X_test)
        end = time.time()
        times.append(end - start)
    
    return np.mean(times)


def compare_models():
    """Compare XGBoost and Random Forest models."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'models'
    data_dir = project_root / 'data'
    
    print("=" * 80)
    print("MODEL COMPARISON: XGBoost vs Random Forest")
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
    
    # Load metadata
    with open(models_dir / 'xgboost_metadata.json', 'r') as f:
        xgb_metadata = json.load(f)
    
    with open(models_dir / 'model_metadata.json', 'r') as f:
        rf_metadata = json.load(f)
    
    # Load test data
    test_data = pd.read_csv(data_dir / 'processed' / 'preprocessed_test.csv')
    X_test = test_data.drop('label_encoded', axis=1)
    y_test = test_data['label_encoded'].values
    
    # Initialize comparison results
    comparison = {
        'accuracy_comparison': {},
        'speed_comparison': {},
        'memory_comparison': {},
        'cross_validation_comparison': {},
        'feature_importance_comparison': {},
        'summary': {}
    }
    
    # ==================== Accuracy Comparison ====================
    print("\n" + "-" * 80)
    print("ACCURACY COMPARISON")
    print("-" * 80)
    
    comparison['accuracy_comparison'] = {
        'xgboost': {
            'test_accuracy': xgb_metrics['test_accuracy'],
            'train_accuracy': xgb_metrics['train_accuracy'],
            'accuracy_gap': xgb_metrics['accuracy_gap'],
            'f1_weighted': xgb_metrics['test_f1_weighted'],
            'precision_weighted': xgb_metrics['test_precision_weighted'],
            'recall_weighted': xgb_metrics['test_recall_weighted']
        },
        'random_forest': {
            'test_accuracy': rf_metrics['test_accuracy'],
            'train_accuracy': rf_metrics['train_accuracy'],
            'accuracy_gap': rf_metrics['accuracy_gap'],
            'f1_weighted': rf_metrics['test_f1_weighted'],
            'precision_weighted': rf_metrics['test_precision_weighted'],
            'recall_weighted': rf_metrics['test_recall_weighted']
        }
    }
    
    print(f"XGBoost Test Accuracy:        {xgb_metrics['test_accuracy']:.4f}")
    print(f"Random Forest Test Accuracy:  {rf_metrics['test_accuracy']:.4f}")
    print(f"Difference:                   {xgb_metrics['test_accuracy'] - rf_metrics['test_accuracy']:.4f}")
    print()
    print(f"XGBoost F1 Score:             {xgb_metrics['test_f1_weighted']:.4f}")
    print(f"Random Forest F1 Score:       {rf_metrics['test_f1_weighted']:.4f}")
    print(f"Difference:                   {xgb_metrics['test_f1_weighted'] - rf_metrics['test_f1_weighted']:.4f}")
    
    # ==================== Speed Comparison ====================
    print("\n" + "-" * 80)
    print("SPEED COMPARISON")
    print("-" * 80)
    
    # Measure prediction times
    print("Measuring prediction times (10 runs)...")
    xgb_pred_time = measure_prediction_time(xgb_model, X_test)
    rf_pred_time = measure_prediction_time(rf_model, X_test)
    
    comparison['speed_comparison'] = {
        'xgboost': {
            'training_time': xgb_metrics['training_time'],
            'tuning_time': xgb_metrics['tuning_time'],
            'prediction_time': xgb_pred_time,
            'total_time': xgb_metrics['training_time'] + xgb_metrics['tuning_time']
        },
        'random_forest': {
            'training_time': rf_metrics['training_time'],
            'tuning_time': rf_metrics['tuning_time'],
            'prediction_time': rf_pred_time,
            'total_time': rf_metrics['training_time'] + rf_metrics['tuning_time']
        }
    }
    
    print(f"XGBoost Training Time:        {xgb_metrics['training_time']:.2f}s")
    print(f"Random Forest Training Time:  {rf_metrics['training_time']:.2f}s")
    print(f"Speedup:                      {rf_metrics['training_time'] / xgb_metrics['training_time']:.2f}x")
    print()
    print(f"XGBoost Tuning Time:          {xgb_metrics['tuning_time']:.2f}s")
    print(f"Random Forest Tuning Time:    {rf_metrics['tuning_time']:.2f}s")
    print()
    print(f"XGBoost Prediction Time:      {xgb_pred_time:.4f}s")
    print(f"Random Forest Prediction Time:{rf_pred_time:.4f}s")
    print(f"Speedup:                      {rf_pred_time / xgb_pred_time:.2f}x")
    
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
    print()
    print(f"XGBoost CV Scores:            {[f'{s:.4f}' for s in xgb_metrics['cv_scores']]}")
    print(f"Random Forest CV Scores:      {[f'{s:.4f}' for s in rf_metrics['cv_scores']]}")
    
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
    
    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Determine winner in each category
    accuracy_winner = "XGBoost" if xgb_metrics['test_accuracy'] > rf_metrics['test_accuracy'] else "Random Forest"
    speed_winner = "XGBoost" if xgb_pred_time < rf_pred_time else "Random Forest"
    memory_winner = "XGBoost" if xgb_size < rf_size else "Random Forest"
    stability_winner = "XGBoost" if xgb_metrics['cv_std'] < rf_metrics['cv_std'] else "Random Forest"
    
    comparison['summary'] = {
        'accuracy_winner': accuracy_winner,
        'speed_winner': speed_winner,
        'memory_winner': memory_winner,
        'stability_winner': stability_winner,
        'overall_recommendation': accuracy_winner  # Prioritize accuracy for production
    }
    
    print(f"\nAccuracy Winner:              {accuracy_winner}")
    print(f"Speed Winner (Prediction):    {speed_winner}")
    print(f"Memory Efficiency Winner:     {memory_winner}")
    print(f"Stability Winner (CV Std):    {stability_winner}")
    print()
    print(f"Overall Recommendation:       {accuracy_winner}")
    print("  (Based on test accuracy as primary metric)")
    
    # ==================== Save Comparison Results ====================
    print("\n" + "-" * 80)
    print("Saving comparison results...")
    
    # Save JSON
    comparison_json_path = models_dir / 'model_comparison.json'
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"JSON comparison saved to: {comparison_json_path}")
    
    # Save Markdown report
    comparison_md_path = models_dir / 'model_comparison.md'
    with open(comparison_md_path, 'w') as f:
        f.write("# Model Comparison: XGBoost vs Random Forest\n\n")
        
        f.write("## Accuracy Comparison\n\n")
        f.write("| Metric | XGBoost | Random Forest | Difference |\n")
        f.write("|--------|---------|---------------|------------|\n")
        f.write(f"| Test Accuracy | {xgb_metrics['test_accuracy']:.4f} | {rf_metrics['test_accuracy']:.4f} | {xgb_metrics['test_accuracy'] - rf_metrics['test_accuracy']:.4f} |\n")
        f.write(f"| F1 Score | {xgb_metrics['test_f1_weighted']:.4f} | {rf_metrics['test_f1_weighted']:.4f} | {xgb_metrics['test_f1_weighted'] - rf_metrics['test_f1_weighted']:.4f} |\n")
        f.write(f"| Precision | {xgb_metrics['test_precision_weighted']:.4f} | {rf_metrics['test_precision_weighted']:.4f} | {xgb_metrics['test_precision_weighted'] - rf_metrics['test_precision_weighted']:.4f} |\n")
        f.write(f"| Recall | {xgb_metrics['test_recall_weighted']:.4f} | {rf_metrics['test_recall_weighted']:.4f} | {xgb_metrics['test_recall_weighted'] - rf_metrics['test_recall_weighted']:.4f} |\n\n")
        
        f.write("## Speed Comparison\n\n")
        f.write("| Metric | XGBoost | Random Forest | Ratio |\n")
        f.write("|--------|---------|---------------|-------|\n")
        f.write(f"| Training Time | {xgb_metrics['training_time']:.2f}s | {rf_metrics['training_time']:.2f}s | {rf_metrics['training_time'] / xgb_metrics['training_time']:.2f}x |\n")
        f.write(f"| Tuning Time | {xgb_metrics['tuning_time']:.2f}s | {rf_metrics['tuning_time']:.2f}s | {rf_metrics['tuning_time'] / xgb_metrics['tuning_time']:.2f}x |\n")
        f.write(f"| Prediction Time | {xgb_pred_time:.4f}s | {rf_pred_time:.4f}s | {rf_pred_time / xgb_pred_time:.2f}x |\n\n")
        
        f.write("## Memory Comparison\n\n")
        f.write("| Metric | XGBoost | Random Forest | Ratio |\n")
        f.write("|--------|---------|---------------|-------|\n")
        f.write(f"| Model Size | {xgb_size:.2f} MB | {rf_size:.2f} MB | {rf_size / xgb_size:.2f}x |\n\n")
        
        f.write("## Cross-Validation Comparison\n\n")
        f.write("| Metric | XGBoost | Random Forest |\n")
        f.write("|--------|---------|---------------|\n")
        f.write(f"| CV Mean | {xgb_metrics['cv_mean']:.4f} | {rf_metrics['cv_mean']:.4f} |\n")
        f.write(f"| CV Std | {xgb_metrics['cv_std']:.4f} | {rf_metrics['cv_std']:.4f} |\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Accuracy Winner**: {accuracy_winner}\n")
        f.write(f"- **Speed Winner**: {speed_winner}\n")
        f.write(f"- **Memory Efficiency Winner**: {memory_winner}\n")
        f.write(f"- **Stability Winner**: {stability_winner}\n")
        f.write(f"- **Overall Recommendation**: {accuracy_winner}\n")
    
    print(f"Markdown report saved to: {comparison_md_path}")
    
    print("\n" + "=" * 80)
    print("Model comparison completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    compare_models()
