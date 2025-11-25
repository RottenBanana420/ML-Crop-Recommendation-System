"""
Visualization Module for Random Forest Model Performance

This module creates comprehensive visualizations for model evaluation including
confusion matrices, ROC curves, feature importance, and performance metrics.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
from sklearn.preprocessing import label_binarize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelVisualizer:
    """Create visualizations for Random Forest model performance."""
    
    def __init__(self, models_dir: Path, data_dir: Path, output_dir: Path):
        """
        Initialize the visualizer.
        
        Args:
            models_dir: Directory containing saved model and metadata
            data_dir: Directory containing test data
            output_dir: Directory to save visualizations
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and data
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.class_names = None
        self.metrics = None
        self.feature_importance = None
        
    def load_model_and_data(self):
        """Load trained model, test data, and metadata."""
        logger.info("Loading model and data...")
        
        # Load model
        model_path = self.models_dir / 'random_forest_model.pkl'
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load test data
        test_data = pd.read_csv(self.data_dir / 'processed' / 'preprocessed_test.csv')
        self.X_test = test_data.drop('label_encoded', axis=1)
        self.y_test = test_data['label_encoded'].values
        
        # Generate predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Load label encoder for class names
        import pickle
        with open(self.models_dir / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        self.class_names = label_encoder.classes_
        
        # Load metrics
        with open(self.models_dir / 'full_metrics.json', 'r') as f:
            self.metrics = json.load(f)
        
        # Load feature importance
        with open(self.models_dir / 'feature_importance.json', 'r') as f:
            self.feature_importance = json.load(f)
        
        logger.info(f"Test set size: {len(self.y_test)}")
        logger.info(f"Number of classes: {len(self.class_names)}")
    
    def plot_confusion_matrix(self):
        """Create and save confusion matrix heatmap."""
        logger.info("Creating confusion matrix visualization...")
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create confusion matrix display
        disp = ConfusionMatrixDisplay.from_predictions(
            self.y_test,
            self.y_pred,
            display_labels=self.class_names,
            cmap='Blues',
            ax=ax,
            colorbar=True
        )
        
        plt.title('Confusion Matrix - Random Forest Crop Recommendation', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Crop', fontsize=12, fontweight='bold')
        plt.ylabel('True Crop', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def plot_feature_importance(self, top_n: int = 15):
        """
        Create feature importance visualization.
        
        Args:
            top_n: Number of top features to display
        """
        logger.info(f"Creating feature importance visualization (top {top_n})...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top N features bar chart
        features = self.feature_importance['features'][:top_n]
        importances = self.feature_importance['importances'][:top_n]
        
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        ax1.barh(range(top_n), importances, color=colors)
        ax1.set_yticks(range(top_n))
        ax1.set_yticklabels(features)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(importances):
            ax1.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
        
        # Cumulative importance
        all_importances = np.array(self.feature_importance['importances'])
        cumsum = np.cumsum(all_importances)
        
        ax2.plot(range(1, len(cumsum) + 1), cumsum, marker='o', linewidth=2, markersize=4)
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold', linewidth=2)
        ax2.axhline(y=0.9, color='orange', linestyle='--', label='90% Threshold', linewidth=2)
        ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
    
    def plot_roc_curves(self):
        """Create ROC curves for multiclass classification (one-vs-rest)."""
        logger.info("Creating ROC curves...")
        
        # Binarize the labels for multiclass ROC
        y_test_bin = label_binarize(self.y_test, classes=range(len(self.class_names)))
        n_classes = y_test_bin.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], self.y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves for top classes (by sample count)
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get class distribution to plot most common classes
        unique, counts = np.unique(self.y_test, return_counts=True)
        top_classes_idx = np.argsort(counts)[-10:][::-1]  # Top 10 classes
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_classes_idx)))
        
        for idx, color in zip(top_classes_idx, colors):
            ax.plot(fpr[idx], tpr[idx], color=color, lw=2,
                   label=f'{self.class_names[idx]} (AUC = {roc_auc[idx]:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Top 10 Crops (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'roc_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {output_path}")
    
    def plot_performance_summary(self):
        """Create a summary visualization of model performance metrics."""
        logger.info("Creating performance summary visualization...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        accuracies = [
            self.metrics['train_accuracy'],
            self.metrics['test_accuracy'],
            self.metrics['cv_mean']
        ]
        labels = ['Train', 'Test', 'CV Mean']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = ax1.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_ylim([0.9, 1.0])
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Precision, Recall, F1 scores
        ax2 = fig.add_subplot(gs[0, 1])
        metrics_data = [
            self.metrics['test_precision_weighted'],
            self.metrics['test_recall_weighted'],
            self.metrics['test_f1_weighted']
        ]
        metric_labels = ['Precision', 'Recall', 'F1-Score']
        bars = ax2.bar(metric_labels, metrics_data, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Test Set Metrics (Weighted)', fontweight='bold')
        ax2.set_ylim([0.9, 1.0])
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, metrics_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Cross-validation scores
        ax3 = fig.add_subplot(gs[0, 2])
        cv_scores = self.metrics['cv_scores']
        ax3.boxplot([cv_scores], labels=['CV Scores'])
        ax3.scatter([1] * len(cv_scores), cv_scores, alpha=0.5, s=50)
        ax3.set_ylabel('Accuracy', fontweight='bold')
        ax3.set_title(f'Cross-Validation Scores\n(Mean: {self.metrics["cv_mean"]:.4f} ± {self.metrics["cv_std"]:.4f})',
                     fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Per-class F1 scores (top 10 classes)
        ax4 = fig.add_subplot(gs[1, :])
        class_report = self.metrics['classification_report']
        class_f1_scores = []
        class_labels = []
        
        for crop in self.class_names:
            if crop in class_report:
                class_f1_scores.append(class_report[crop]['f1-score'])
                class_labels.append(crop)
        
        # Sort by F1 score and take top 10
        sorted_indices = np.argsort(class_f1_scores)[::-1][:10]
        top_crops = [class_labels[i] for i in sorted_indices]
        top_f1_scores = [class_f1_scores[i] for i in sorted_indices]
        
        colors = plt.cm.RdYlGn(np.array(top_f1_scores))
        ax4.barh(range(len(top_crops)), top_f1_scores, color=colors, edgecolor='black', linewidth=1)
        ax4.set_yticks(range(len(top_crops)))
        ax4.set_yticklabels(top_crops)
        ax4.invert_yaxis()
        ax4.set_xlabel('F1-Score', fontweight='bold')
        ax4.set_title('Top 10 Crops by F1-Score', fontweight='bold')
        ax4.set_xlim([0.8, 1.0])
        ax4.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(top_f1_scores):
            ax4.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
        
        # 5. Training metrics summary
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = f"""
        Model Performance Summary
        {'=' * 80}
        
        Training Accuracy:        {self.metrics['train_accuracy']:.4f}
        Test Accuracy:            {self.metrics['test_accuracy']:.4f}
        Accuracy Gap:             {self.metrics['accuracy_gap']:.4f}
        
        Cross-Validation Mean:    {self.metrics['cv_mean']:.4f}
        Cross-Validation Std:     {self.metrics['cv_std']:.4f}
        
        Test Precision (Weighted): {self.metrics['test_precision_weighted']:.4f}
        Test Recall (Weighted):    {self.metrics['test_recall_weighted']:.4f}
        Test F1-Score (Weighted):  {self.metrics['test_f1_weighted']:.4f}
        
        Training Time:            {self.metrics.get('training_time', 0):.2f} seconds
        Tuning Time:              {self.metrics.get('tuning_time', 0):.2f} seconds
        """
        
        ax5.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle('Random Forest Model - Performance Summary', fontsize=18, fontweight='bold', y=0.98)
        
        output_path = self.output_dir / 'performance_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance summary saved to {output_path}")
    
    def create_all_visualizations(self):
        """Create all visualizations."""
        logger.info("Creating all visualizations...")
        
        self.load_model_and_data()
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.plot_roc_curves()
        self.plot_performance_summary()
        
        logger.info(f"All visualizations saved to {self.output_dir}")


def main():
    """Main execution function."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'models'
    data_dir = project_root / 'data'
    output_dir = data_dir / 'visualizations' / 'model_performance'
    
    # Create visualizations
    visualizer = ModelVisualizer(models_dir, data_dir, output_dir)
    visualizer.create_all_visualizations()
    
    logger.info("=" * 80)
    logger.info("All visualizations created successfully!")
    logger.info(f"Visualizations saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
