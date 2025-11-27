"""
XGBoost Classification Model for Crop Recommendation

This module implements an XGBoost classifier with hyperparameter tuning,
cross-validation, and comprehensive evaluation metrics.
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostCropModel:
    """XGBoost model for crop recommendation."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.label_encoder = None
        self.metrics = {}
        self.feature_importance = None
        self.n_classes = None
        
    def load_data(self, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load preprocessed training and testing data.
        
        Args:
            data_dir: Path to data directory
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Loading preprocessed data...")
        
        # Load training and testing data
        train_data = pd.read_csv(data_dir / 'processed' / 'preprocessed_train.csv')
        test_data = pd.read_csv(data_dir / 'processed' / 'preprocessed_test.csv')
        
        # Load feature names
        with open(data_dir.parent / 'models' / 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Load label encoder
        with open(data_dir.parent / 'models' / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Separate features and target
        X_train = train_data.drop('label_encoded', axis=1)
        y_train = train_data['label_encoded'].values
        X_test = test_data.drop('label_encoded', axis=1)
        y_test = test_data['label_encoded'].values
        
        self.n_classes = len(np.unique(y_train))
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Number of features: {len(self.feature_names)}")
        logger.info(f"Number of classes: {self.n_classes}")
        
        return X_train, X_test, y_train, y_test
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best hyperparameters
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Define parameter distribution with realistic, conservative values
        # to avoid overfitting and ensure generalizable results
        param_distributions = {
            'n_estimators': [100, 200, 300],  # Moderate number of trees
            'max_depth': [3, 5, 7, 9],  # Conservative depths to prevent overfitting
            'learning_rate': [0.05, 0.1, 0.2],  # Standard learning rates
            'subsample': [0.7, 0.8, 0.9],  # Prevent overfitting with subsampling
            'colsample_bytree': [0.7, 0.8, 0.9],  # Feature subsampling
            'gamma': [0, 0.1, 0.2],  # Minimum loss reduction for split
            'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight
            'reg_alpha': [0, 0.01, 0.1],  # L1 regularization
            'reg_lambda': [1, 1.5, 2],  # L2 regularization (default is 1)
        }
        
        # Initialize base model with multi-class configuration
        base_model = xgb.XGBClassifier(
            objective='multi:softprob',  # Multi-class classification with probabilities
            num_class=self.n_classes,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss'  # Multi-class log loss
        )
        
        # Perform randomized search with stratified CV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=30,  # Reasonable number of iterations
            cv=5,  # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        self.best_params = random_search.best_params_
        
        logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        # Store CV results
        self.metrics['cv_best_score'] = random_search.best_score_
        self.metrics['cv_std'] = random_search.cv_results_['std_test_score'][random_search.best_index_]
        self.metrics['tuning_time'] = tuning_time
        
        return self.best_params
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray, use_best_params: bool = True):
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_best_params: Whether to use tuned hyperparameters
        """
        logger.info("Training XGBoost model...")
        
        if use_best_params and self.best_params:
            params = {
                **self.best_params,
                'objective': 'multi:softprob',
                'num_class': self.n_classes,
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'mlogloss'
            }
        else:
            # Default parameters - conservative to avoid overfitting
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'min_child_weight': 3,
                'reg_alpha': 0.01,
                'reg_lambda': 1.5,
                'objective': 'multi:softprob',
                'num_class': self.n_classes,
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'mlogloss'
            }
        
        # Initialize and train model
        self.model = xgb.XGBClassifier(**params)
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.metrics['training_time'] = training_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Extract feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 important features:")
        for idx, row in self.feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def evaluate(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                 X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on training and test sets.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Training set predictions
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Test set predictions
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Calculate comprehensive metrics
        self.metrics.update({
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'accuracy_gap': train_accuracy - test_accuracy,
            'test_precision_macro': precision_score(y_test, y_test_pred, average='macro'),
            'test_precision_weighted': precision_score(y_test, y_test_pred, average='weighted'),
            'test_recall_macro': recall_score(y_test, y_test_pred, average='macro'),
            'test_recall_weighted': recall_score(y_test, y_test_pred, average='weighted'),
            'test_f1_macro': f1_score(y_test, y_test_pred, average='macro'),
            'test_f1_weighted': f1_score(y_test, y_test_pred, average='weighted'),
        })
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)
        self.metrics['classification_report'] = report
        
        # Cross-validation on full training set
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        self.metrics['cv_scores'] = cv_scores.tolist()
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()
        
        # Log results
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Accuracy Gap (Train - Test): {self.metrics['accuracy_gap']:.4f}")
        logger.info(f"Test F1 Score (Weighted): {self.metrics['test_f1_weighted']:.4f}")
        logger.info(f"Cross-Validation Mean: {self.metrics['cv_mean']:.4f} (+/- {self.metrics['cv_std']:.4f})")
        
        return self.metrics
    
    def save_model(self, output_dir: Path):
        """
        Save the trained model and metadata.
        
        Args:
            output_dir: Directory to save model files
        """
        logger.info("Saving model and metadata...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / 'xgboost_model.pkl'
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'XGBClassifier',
            'training_date': datetime.now().isoformat(),
            'hyperparameters': self.model.get_params(),
            'metrics': {k: v for k, v in self.metrics.items() 
                       if k not in ['confusion_matrix', 'classification_report', 'cv_scores']},
            'feature_count': len(self.feature_names),
            'class_count': self.n_classes,
        }
        
        metadata_path = output_dir / 'xgboost_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Save feature importance
        importance_data = {
            'features': self.feature_importance['feature'].tolist(),
            'importances': [float(x) for x in self.feature_importance['importance'].tolist()],  # Convert to Python float
            'importance_sum': float(self.feature_importance['importance'].sum()),  # Convert to Python float
        }
        
        importance_path = output_dir / 'xgboost_feature_importance.json'
        with open(importance_path, 'w') as f:
            json.dump(importance_data, f, indent=2)
        logger.info(f"Feature importance saved to {importance_path}")
        
        # Save full metrics including confusion matrix and classification report
        full_metrics_path = output_dir / 'xgboost_full_metrics.json'
        with open(full_metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Full metrics saved to {full_metrics_path}")


def main():
    """Main execution function."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    
    # Initialize model
    model = XGBoostCropModel(random_state=42)
    
    # Load data
    X_train, X_test, y_train, y_test = model.load_data(data_dir)
    
    # Tune hyperparameters
    best_params = model.tune_hyperparameters(X_train, y_train)
    
    # Train model with best parameters
    model.train(X_train, y_train, use_best_params=True)
    
    # Evaluate model
    metrics = model.evaluate(X_train, y_train, X_test, y_test)
    
    # Save model and results
    model.save_model(models_dir)
    
    logger.info("=" * 80)
    logger.info("XGBoost model training and evaluation completed successfully!")
    logger.info(f"Final Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Model saved to: {models_dir / 'xgboost_model.pkl'}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
