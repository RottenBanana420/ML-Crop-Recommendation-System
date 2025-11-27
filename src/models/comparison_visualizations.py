"""
Comparison Visualizations Module

This module generates comprehensive comparative visualizations for model comparison,
including radar charts, heatmaps, confusion matrices, and performance plots.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


def create_radar_chart(xgb_metrics, rf_metrics, output_path):
    """
    Create radar chart comparing models across multiple metrics.
    
    Args:
        xgb_metrics: XGBoost metrics dictionary
        rf_metrics: Random Forest metrics dictionary
        output_path: Path to save the chart
    """
    # Select metrics for comparison (normalized to 0-1 scale)
    metrics = {
        'Accuracy': (xgb_metrics['test_accuracy'], rf_metrics['test_accuracy']),
        'F1 Score': (xgb_metrics['test_f1_weighted'], rf_metrics['test_f1_weighted']),
        'Precision': (xgb_metrics['test_precision_weighted'], rf_metrics['test_precision_weighted']),
        'Recall': (xgb_metrics['test_recall_weighted'], rf_metrics['test_recall_weighted']),
        'CV Stability': (1 - xgb_metrics['cv_std'], 1 - rf_metrics['cv_std']),  # Invert std (higher is better)
    }
    
    categories = list(metrics.keys())
    xgb_values = [metrics[cat][0] for cat in categories]
    rf_values = [metrics[cat][1] for cat in categories]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    xgb_values += xgb_values[:1]  # Complete the circle
    rf_values += rf_values[:1]
    angles += angles[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, xgb_values, 'o-', linewidth=2, label='XGBoost', color='#2E86AB')
    ax.fill(angles, xgb_values, alpha=0.25, color='#2E86AB')
    
    ax.plot(angles, rf_values, 'o-', linewidth=2, label='Random Forest', color='#A23B72')
    ax.fill(angles, rf_values, alpha=0.25, color='#A23B72')
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    
    # Set y-axis limits
    ax.set_ylim(0.94, 1.0)
    
    # Add legend and title
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Multi-Metric Model Comparison', size=16, weight='bold', pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar chart saved to: {output_path}")


def create_per_class_heatmap(xgb_per_class, rf_per_class, crop_names, output_path):
    """
    Create heatmap showing per-class F1 scores for both models.
    
    Args:
        xgb_per_class: XGBoost per-class metrics
        rf_per_class: Random Forest per-class metrics
        crop_names: List of crop names
        output_path: Path to save the heatmap
    """
    # Create DataFrame for heatmap
    data = {
        'XGBoost F1': xgb_per_class['f1_scores'],
        'Random Forest F1': rf_per_class['f1_scores'],
        'Difference (XGB - RF)': [xgb - rf for xgb, rf in zip(xgb_per_class['f1_scores'], rf_per_class['f1_scores'])]
    }
    
    df = pd.DataFrame(data, index=crop_names)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    # XGBoost heatmap
    sns.heatmap(df[['XGBoost F1']], annot=True, fmt='.3f', cmap='YlGnBu', 
                vmin=0.90, vmax=1.0, ax=axes[0], cbar_kws={'label': 'F1 Score'})
    axes[0].set_title('XGBoost Per-Class F1 Scores', fontsize=14, weight='bold')
    axes[0].set_ylabel('Crop Type', fontsize=12)
    
    # Random Forest heatmap
    sns.heatmap(df[['Random Forest F1']], annot=True, fmt='.3f', cmap='YlGnBu',
                vmin=0.90, vmax=1.0, ax=axes[1], cbar_kws={'label': 'F1 Score'})
    axes[1].set_title('Random Forest Per-Class F1 Scores', fontsize=14, weight='bold')
    axes[1].set_ylabel('')
    
    # Difference heatmap
    sns.heatmap(df[['Difference (XGB - RF)']], annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, vmin=-0.05, vmax=0.05, ax=axes[2], cbar_kws={'label': 'Difference'})
    axes[2].set_title('Performance Difference (XGBoost - Random Forest)', fontsize=14, weight='bold')
    axes[2].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class heatmap saved to: {output_path}")


def create_confusion_matrix_comparison(xgb_cm, rf_cm, crop_names, output_path):
    """
    Create side-by-side confusion matrix comparison.
    
    Args:
        xgb_cm: XGBoost confusion matrix
        rf_cm: Random Forest confusion matrix
        crop_names: List of crop names
        output_path: Path to save the comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # XGBoost confusion matrix
    sns.heatmap(xgb_cm, annot=False, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=crop_names, yticklabels=crop_names, cbar_kws={'label': 'Count'})
    axes[0].set_title('XGBoost Confusion Matrix', fontsize=16, weight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='both', labelsize=8)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    plt.setp(axes[0].get_yticklabels(), rotation=0)
    
    # Random Forest confusion matrix
    sns.heatmap(rf_cm, annot=False, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=crop_names, yticklabels=crop_names, cbar_kws={'label': 'Count'})
    axes[1].set_title('Random Forest Confusion Matrix', fontsize=16, weight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    plt.setp(axes[1].get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix comparison saved to: {output_path}")


def create_inference_time_plot(xgb_times, rf_times, output_path):
    """
    Create box plot comparing inference time distributions.
    
    Args:
        xgb_times: List of XGBoost inference times
        rf_times: List of Random Forest inference times
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plot
    bp = ax.boxplot([xgb_times, rf_times], labels=['XGBoost', 'Random Forest'],
                     patch_artist=True, showmeans=True)
    
    # Customize colors
    colors = ['#2E86AB', '#A23B72']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Labels and title
    ax.set_ylabel('Inference Time (seconds)', fontsize=12)
    ax.set_title('Inference Time Distribution Comparison', fontsize=14, weight='bold')
    
    # Add mean values as text
    xgb_mean = np.mean(xgb_times)
    rf_mean = np.mean(rf_times)
    ax.text(1, xgb_mean, f'Mean: {xgb_mean:.4f}s', ha='center', va='bottom', fontsize=10)
    ax.text(2, rf_mean, f'Mean: {rf_mean:.4f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Inference time plot saved to: {output_path}")


def create_feature_importance_correlation(xgb_importance, rf_importance, output_path):
    """
    Create scatter plot showing correlation between feature importances.
    
    Args:
        xgb_importance: XGBoost feature importance dict
        rf_importance: Random Forest feature importance dict
        output_path: Path to save the plot
    """
    # Align features
    features = xgb_importance['features']
    xgb_imp = xgb_importance['importances']
    
    # Create mapping for RF importances
    rf_imp_dict = dict(zip(rf_importance['features'], rf_importance['importances']))
    rf_imp = [rf_imp_dict.get(feat, 0) for feat in features]
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(xgb_imp, rf_imp, alpha=0.6, s=100, color='#2E86AB', edgecolors='black', linewidth=1)
    
    # Add diagonal line (perfect correlation)
    max_val = max(max(xgb_imp), max(rf_imp))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Correlation', alpha=0.7)
    
    # Calculate correlation
    correlation = np.corrcoef(xgb_imp, rf_imp)[0, 1]
    
    # Labels and title
    ax.set_xlabel('XGBoost Feature Importance', fontsize=12)
    ax.set_ylabel('Random Forest Feature Importance', fontsize=12)
    ax.set_title(f'Feature Importance Correlation\n(Pearson r = {correlation:.3f})', 
                 fontsize=14, weight='bold')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Annotate top features
    for i in range(min(5, len(features))):
        ax.annotate(features[i], (xgb_imp[i], rf_imp[i]), 
                   fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance correlation plot saved to: {output_path}")


def create_performance_efficiency_scatter(comparison_data, output_path):
    """
    Create scatter plot showing accuracy vs inference speed trade-off.
    
    Args:
        comparison_data: Complete comparison data dictionary
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    models = ['XGBoost', 'Random Forest']
    accuracies = [
        comparison_data['accuracy_comparison']['xgboost']['test_accuracy'],
        comparison_data['accuracy_comparison']['random_forest']['test_accuracy']
    ]
    speeds = [
        comparison_data['speed_comparison']['xgboost']['prediction_time'],
        comparison_data['speed_comparison']['random_forest']['prediction_time']
    ]
    sizes = [
        comparison_data['memory_comparison']['xgboost']['model_size_mb'],
        comparison_data['memory_comparison']['random_forest']['model_size_mb']
    ]
    
    colors = ['#2E86AB', '#A23B72']
    
    # Create scatter plot with size representing model size
    for i, (model, acc, speed, size, color) in enumerate(zip(models, accuracies, speeds, sizes, colors)):
        ax.scatter(speed, acc, s=size*50, alpha=0.6, color=color, 
                  edgecolors='black', linewidth=2, label=f'{model} (Size: {size:.1f}MB)')
        ax.annotate(model, (speed, acc), fontsize=12, weight='bold',
                   xytext=(10, 10), textcoords='offset points')
    
    # Labels and title
    ax.set_xlabel('Inference Time (seconds) - Lower is Better', fontsize=12)
    ax.set_ylabel('Test Accuracy - Higher is Better', fontsize=12)
    ax.set_title('Performance-Efficiency Trade-off\n(Bubble size = Model size)', 
                 fontsize=14, weight='bold')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, loc='best')
    
    # Add quadrant lines
    ax.axhline(y=np.mean(accuracies), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=np.mean(speeds), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance-efficiency scatter plot saved to: {output_path}")


def save_all_visualizations(comparison_data, xgb_metrics, rf_metrics, 
                            xgb_cm, rf_cm, xgb_times, rf_times,
                            xgb_importance, rf_importance, crop_names, output_dir):
    """
    Generate and save all comparison visualizations.
    
    Args:
        comparison_data: Complete comparison data dictionary
        xgb_metrics: XGBoost metrics
        rf_metrics: Random Forest metrics
        xgb_cm: XGBoost confusion matrix
        rf_cm: Random Forest confusion matrix
        xgb_times: XGBoost inference times
        rf_times: Random Forest inference times
        xgb_importance: XGBoost feature importance
        rf_importance: Random Forest feature importance
        crop_names: List of crop names
        output_dir: Directory to save visualizations
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    # Generate all visualizations
    create_radar_chart(xgb_metrics, rf_metrics, output_dir / 'radar_chart.png')
    
    create_per_class_heatmap(
        comparison_data['per_class_comparison']['xgboost'],
        comparison_data['per_class_comparison']['random_forest'],
        crop_names,
        output_dir / 'per_class_heatmap.png'
    )
    
    create_confusion_matrix_comparison(xgb_cm, rf_cm, crop_names, 
                                      output_dir / 'confusion_matrices.png')
    
    create_inference_time_plot(xgb_times, rf_times, 
                              output_dir / 'inference_time_distribution.png')
    
    create_feature_importance_correlation(xgb_importance, rf_importance,
                                         output_dir / 'feature_importance_correlation.png')
    
    create_performance_efficiency_scatter(comparison_data,
                                         output_dir / 'performance_efficiency_scatter.png')
    
    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 80)
