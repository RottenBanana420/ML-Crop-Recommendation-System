"""
XGBoost Model Visualizations

This module creates confusion matrix visualizations and misclassification analysis
for the XGBoost crop recommendation model.
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def create_confusion_matrix_plot(y_true, y_pred, class_names, title, save_path):
    """
    Create and save a confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def analyze_misclassifications(y_true, y_pred, class_names, save_path):
    """
    Analyze and visualize most commonly confused crop pairs.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Find misclassifications (off-diagonal elements)
    misclassifications = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclassifications.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': cm[i, j]
                })
    
    # Sort by count
    misclassifications.sort(key=lambda x: x['count'], reverse=True)
    
    # Take top 10 most common misclassifications
    top_misclassifications = misclassifications[:10]
    
    if not top_misclassifications:
        print("No misclassifications found! Perfect predictions.")
        return
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = [f"{m['true_class']} → {m['predicted_class']}" for m in top_misclassifications]
    counts = [m['count'] for m in top_misclassifications]
    
    bars = ax.barh(labels, counts, color='coral')
    ax.set_xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class → Predicted Class', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Most Common Misclassifications', fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 0.1, i, str(count), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Misclassification analysis saved to {save_path}")
    
    # Print summary
    print("\nMost Commonly Confused Crop Pairs:")
    for i, m in enumerate(top_misclassifications, 1):
        print(f"{i}. {m['true_class']} confused with {m['predicted_class']}: {m['count']} times")


def create_comparison_confusion_matrices(rf_cm, xgb_cm, class_names, save_path):
    """
    Create side-by-side comparison of Random Forest and XGBoost confusion matrices.
    
    Args:
        rf_cm: Random Forest confusion matrix
        xgb_cm: XGBoost confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(28, 12))
    
    # Random Forest confusion matrix
    sns.heatmap(
        rf_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    # XGBoost confusion matrix
    sns.heatmap(
        xgb_cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar_kws={'label': 'Count'}
    )
    axes[1].set_title('XGBoost Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison confusion matrices saved to {save_path}")


def main():
    """Main execution function."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'models'
    data_dir = project_root / 'data'
    viz_dir = data_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading XGBoost model and data...")
    
    # Load XGBoost metrics
    with open(models_dir / 'xgboost_full_metrics.json', 'r') as f:
        xgb_metrics = json.load(f)
    
    # Load label encoder
    with open(models_dir / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    class_names = label_encoder.classes_
    
    # Load test data
    test_data = pd.read_csv(data_dir / 'processed' / 'preprocessed_test.csv')
    y_test = test_data['label_encoded'].values
    
    # Load XGBoost model and make predictions
    import joblib
    xgb_model = joblib.load(models_dir / 'xgboost_model.pkl')
    X_test = test_data.drop('label_encoded', axis=1)
    y_pred_xgb = xgb_model.predict(X_test)
    
    print("\nCreating XGBoost confusion matrix visualization...")
    create_confusion_matrix_plot(
        y_test,
        y_pred_xgb,
        class_names,
        'XGBoost Confusion Matrix - Crop Recommendation',
        viz_dir / 'xgboost_confusion_matrix.png'
    )
    
    print("\nAnalyzing XGBoost misclassifications...")
    analyze_misclassifications(
        y_test,
        y_pred_xgb,
        class_names,
        viz_dir / 'xgboost_misclassification_analysis.png'
    )
    
    # Create comparison with Random Forest if available
    rf_metrics_path = models_dir / 'full_metrics.json'
    if rf_metrics_path.exists():
        print("\nCreating comparison with Random Forest...")
        
        with open(rf_metrics_path, 'r') as f:
            rf_metrics = json.load(f)
        
        rf_cm = np.array(rf_metrics['confusion_matrix'])
        xgb_cm = np.array(xgb_metrics['confusion_matrix'])
        
        create_comparison_confusion_matrices(
            rf_cm,
            xgb_cm,
            class_names,
            viz_dir / 'model_comparison_confusion_matrices.png'
        )
    
    print("\n" + "=" * 80)
    print("Visualization generation completed successfully!")
    print(f"Visualizations saved to: {viz_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
