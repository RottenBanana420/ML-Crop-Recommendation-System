"""
Comprehensive Test Suite for Random Forest Model

This module contains stringent tests that FAIL if the model doesn't meet
production-quality standards. Tests should NEVER be modified to pass - 
instead, improve the model training code.
"""

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score


class TestRandomForestModel:
    """Test suite for Random Forest crop recommendation model."""
    
    @pytest.fixture(scope='class')
    def project_paths(self):
        """Set up project paths."""
        project_root = Path(__file__).parent.parent
        return {
            'models_dir': project_root / 'models',
            'data_dir': project_root / 'data',
        }
    
    @pytest.fixture(scope='class')
    def model_artifacts(self, project_paths):
        """Load model and related artifacts."""
        models_dir = project_paths['models_dir']
        
        # Load model
        model_path = models_dir / 'random_forest_model.pkl'
        assert model_path.exists(), "Model file does not exist. Run training first."
        model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = models_dir / 'model_metadata.json'
        assert metadata_path.exists(), "Metadata file does not exist."
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load full metrics
        metrics_path = models_dir / 'full_metrics.json'
        assert metrics_path.exists(), "Metrics file does not exist."
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load feature importance
        importance_path = models_dir / 'feature_importance.json'
        assert importance_path.exists(), "Feature importance file does not exist."
        with open(importance_path, 'r') as f:
            feature_importance = json.load(f)
        
        # Load label encoder
        encoder_path = models_dir / 'label_encoder.pkl'
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return {
            'model': model,
            'metadata': metadata,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'label_encoder': label_encoder,
        }
    
    @pytest.fixture(scope='class')
    def test_data(self, project_paths):
        """Load test data."""
        data_dir = project_paths['data_dir']
        test_data = pd.read_csv(data_dir / 'processed' / 'preprocessed_test.csv')
        
        X_test = test_data.drop('label_encoded', axis=1)
        y_test = test_data['label_encoded'].values
        
        return X_test, y_test
    
    # ==================== Test 1: Model Accuracy Thresholds ====================
    
    def test_minimum_test_accuracy(self, model_artifacts):
        """
        FAIL if test accuracy < 95%.
        
        This ensures the model meets production-quality standards.
        """
        test_accuracy = model_artifacts['metrics']['test_accuracy']
        
        assert test_accuracy >= 0.95, (
            f"Test accuracy {test_accuracy:.4f} is below minimum threshold of 0.95. "
            f"Model does not meet production standards. "
            f"Improve hyperparameter tuning or feature engineering."
        )
    
    def test_minimum_train_accuracy(self, model_artifacts):
        """
        FAIL if training accuracy < 95%.
        
        This ensures the model can learn the training data effectively.
        """
        train_accuracy = model_artifacts['metrics']['train_accuracy']
        
        assert train_accuracy >= 0.95, (
            f"Training accuracy {train_accuracy:.4f} is below minimum threshold of 0.95. "
            f"Model is underfitting. Consider more complex model or better features."
        )
    
    # ==================== Test 2: Overfitting Detection ====================
    
    def test_overfitting_train_test_gap(self, model_artifacts):
        """
        FAIL if (train_accuracy - test_accuracy) > 5%.
        
        This detects overfitting by checking the performance gap.
        """
        train_acc = model_artifacts['metrics']['train_accuracy']
        test_acc = model_artifacts['metrics']['test_accuracy']
        gap = train_acc - test_acc
        
        assert gap <= 0.05, (
            f"Train-test accuracy gap {gap:.4f} exceeds maximum of 0.05. "
            f"Model is overfitting. Reduce model complexity or add regularization."
        )
    
    def test_cross_validation_stability(self, model_artifacts):
        """
        FAIL if cross-validation standard deviation > 3%.
        
        This ensures the model performs consistently across different data splits.
        """
        cv_std = model_artifacts['metrics']['cv_std']
        
        assert cv_std <= 0.03, (
            f"Cross-validation std {cv_std:.4f} exceeds maximum of 0.03. "
            f"Model performance is unstable. Check for data issues or model instability."
        )
    
    # ==================== Test 3: Feature Importance Validation ====================
    
    def test_feature_importance_sum(self, model_artifacts):
        """
        FAIL if feature importances don't sum to 1.0 (±0.001 tolerance).
        
        Random Forest importances should always sum to 1.0.
        """
        importance_sum = model_artifacts['feature_importance']['importance_sum']
        
        assert abs(importance_sum - 1.0) <= 0.001, (
            f"Feature importances sum to {importance_sum:.6f}, expected 1.0. "
            f"This indicates a bug in feature importance calculation."
        )
    
    def test_no_negative_importances(self, model_artifacts):
        """
        FAIL if any feature importance is negative.
        
        Feature importances should always be non-negative.
        """
        importances = model_artifacts['feature_importance']['importances']
        
        assert all(imp >= 0 for imp in importances), (
            f"Found negative feature importances. "
            f"This should never happen with Random Forest."
        )
    
    def test_non_zero_importances(self, model_artifacts):
        """
        FAIL if all feature importances are zero.
        
        At least some features should have non-zero importance.
        """
        importances = model_artifacts['feature_importance']['importances']
        
        assert any(imp > 0 for imp in importances), (
            f"All feature importances are zero. "
            f"Model is not using any features."
        )
    
    def test_importance_array_length(self, model_artifacts):
        """
        FAIL if importance array length doesn't match feature count.
        
        Every feature should have an importance value.
        """
        importances = model_artifacts['feature_importance']['importances']
        features = model_artifacts['feature_importance']['features']
        
        assert len(importances) == len(features), (
            f"Importance array length {len(importances)} doesn't match "
            f"feature count {len(features)}."
        )
    
    # ==================== Test 4: Model Serialization ====================
    
    def test_model_file_exists(self, project_paths):
        """
        FAIL if model file doesn't exist after training.
        
        Model must be saved for deployment.
        """
        model_path = project_paths['models_dir'] / 'random_forest_model.pkl'
        
        assert model_path.exists(), (
            f"Model file not found at {model_path}. "
            f"Model was not saved properly."
        )
    
    def test_model_can_be_loaded(self, project_paths):
        """
        FAIL if saved model cannot be loaded.
        
        Model must be loadable for deployment.
        """
        model_path = project_paths['models_dir'] / 'random_forest_model.pkl'
        
        try:
            loaded_model = joblib.load(model_path)
            assert loaded_model is not None
        except Exception as e:
            pytest.fail(f"Failed to load model: {str(e)}")
    
    def test_loaded_model_predictions_match(self, model_artifacts, test_data, project_paths):
        """
        FAIL if loaded model predictions differ from original.
        
        Ensures model serialization preserves functionality.
        """
        X_test, y_test = test_data
        original_model = model_artifacts['model']
        
        # Load model fresh
        model_path = project_paths['models_dir'] / 'random_forest_model.pkl'
        loaded_model = joblib.load(model_path)
        
        # Compare predictions
        original_preds = original_model.predict(X_test)
        loaded_preds = loaded_model.predict(X_test)
        
        assert np.array_equal(original_preds, loaded_preds), (
            f"Loaded model predictions differ from original model. "
            f"Model serialization is broken."
        )
    
    # ==================== Test 5: Prediction Consistency ====================
    
    def test_prediction_determinism(self, model_artifacts, test_data):
        """
        FAIL if same input produces different outputs.
        
        With fixed random_state, predictions should be deterministic.
        """
        X_test, _ = test_data
        model = model_artifacts['model']
        
        # Make predictions multiple times
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        preds3 = model.predict(X_test)
        
        assert np.array_equal(preds1, preds2), (
            f"Predictions are not deterministic (run 1 vs 2). "
            f"Check random_state configuration."
        )
        
        assert np.array_equal(preds2, preds3), (
            f"Predictions are not deterministic (run 2 vs 3). "
            f"Check random_state configuration."
        )
    
    def test_prediction_probabilities_sum_to_one(self, model_artifacts, test_data):
        """
        FAIL if prediction probabilities don't sum to 1.0 for each sample.
        
        Probability distributions must be valid.
        """
        X_test, _ = test_data
        model = model_artifacts['model']
        
        proba = model.predict_proba(X_test)
        sums = proba.sum(axis=1)
        
        assert np.allclose(sums, 1.0, atol=1e-6), (
            f"Prediction probabilities don't sum to 1.0. "
            f"Found sums ranging from {sums.min():.6f} to {sums.max():.6f}."
        )
    
    # ==================== Test 6: Edge Cases ====================
    
    def test_predictions_are_valid_classes(self, model_artifacts, test_data):
        """
        FAIL if model produces invalid class predictions.
        
        All predictions must be within valid class range.
        """
        X_test, _ = test_data
        model = model_artifacts['model']
        label_encoder = model_artifacts['label_encoder']
        
        predictions = model.predict(X_test)
        n_classes = len(label_encoder.classes_)
        
        assert all(0 <= pred < n_classes for pred in predictions), (
            f"Model produced invalid class predictions. "
            f"Valid range is [0, {n_classes-1}], but found predictions outside this range."
        )
    
    def test_model_handles_boundary_values(self, model_artifacts, test_data):
        """
        FAIL if model cannot handle boundary values from training data.
        
        Model should handle min/max feature values gracefully.
        """
        X_test, _ = test_data
        model = model_artifacts['model']
        
        # Create synthetic data with min/max values
        X_min = pd.DataFrame([X_test.min().values], columns=X_test.columns)
        X_max = pd.DataFrame([X_test.max().values], columns=X_test.columns)
        
        try:
            pred_min = model.predict(X_min)
            pred_max = model.predict(X_max)
            
            assert len(pred_min) == 1
            assert len(pred_max) == 1
        except Exception as e:
            pytest.fail(f"Model failed on boundary values: {str(e)}")
    
    def test_model_handles_single_sample(self, model_artifacts, test_data):
        """
        FAIL if model cannot predict on a single sample.
        
        Model should work with batch size of 1.
        """
        X_test, _ = test_data
        model = model_artifacts['model']
        
        single_sample = X_test.iloc[[0]]
        
        try:
            prediction = model.predict(single_sample)
            assert len(prediction) == 1
        except Exception as e:
            pytest.fail(f"Model failed on single sample: {str(e)}")
    
    # ==================== Test 7: Cross-Validation Integrity ====================
    
    def test_cv_scores_not_unrealistic(self, model_artifacts):
        """
        FAIL if CV scores are unrealistically high (>99.8%).
        
        Perfect scores (100%) suggest data leakage or other issues.
        Note: 99-99.8% is achievable with excellent preprocessing and tuning.
        """
        cv_scores = model_artifacts['metrics']['cv_scores']
        
        assert all(score <= 0.998 for score in cv_scores), (
            f"Cross-validation scores are unrealistically high (>99.8%). "
            f"Check for data leakage or duplicate samples. Scores: {cv_scores}"
        )
    
    def test_cv_scores_reasonable_range(self, model_artifacts):
        """
        FAIL if CV scores vary wildly (range > 10%).
        
        Scores should be relatively consistent.
        """
        cv_scores = model_artifacts['metrics']['cv_scores']
        score_range = max(cv_scores) - min(cv_scores)
        
        assert score_range <= 0.10, (
            f"Cross-validation scores vary too much (range: {score_range:.4f}). "
            f"This suggests data distribution issues or model instability."
        )
    
    # ==================== Test 8: Model Configuration ====================
    
    def test_hyperparameters_saved(self, model_artifacts):
        """
        FAIL if hyperparameters are not saved correctly.
        
        All model configuration must be persisted.
        """
        metadata = model_artifacts['metadata']
        
        assert 'hyperparameters' in metadata, (
            f"Hyperparameters not found in metadata."
        )
        
        required_params = ['n_estimators', 'max_depth', 'min_samples_split', 
                          'min_samples_leaf', 'max_features', 'random_state']
        
        for param in required_params:
            assert param in metadata['hyperparameters'], (
                f"Required parameter '{param}' not found in saved hyperparameters."
            )
    
    def test_metadata_completeness(self, model_artifacts):
        """
        FAIL if model metadata is incomplete.
        
        Metadata must include all essential information.
        """
        metadata = model_artifacts['metadata']
        
        required_fields = [
            'model_type',
            'training_date',
            'hyperparameters',
            'metrics',
            'feature_count',
            'class_count'
        ]
        
        for field in required_fields:
            assert field in metadata, (
                f"Required metadata field '{field}' is missing."
            )
    
    def test_random_state_is_set(self, model_artifacts):
        """
        FAIL if random_state is not set.
        
        Random state must be set for reproducibility.
        """
        metadata = model_artifacts['metadata']
        random_state = metadata['hyperparameters'].get('random_state')
        
        assert random_state is not None, (
            f"random_state is not set. Model is not reproducible."
        )
        
        assert isinstance(random_state, int), (
            f"random_state should be an integer, got {type(random_state)}."
        )
    
    # ==================== Test 9: Performance Metrics ====================
    
    def test_f1_score_meets_threshold(self, model_artifacts):
        """
        FAIL if weighted F1-score < 0.95.
        
        F1-score should be high for balanced performance.
        """
        f1_weighted = model_artifacts['metrics']['test_f1_weighted']
        
        assert f1_weighted >= 0.95, (
            f"Weighted F1-score {f1_weighted:.4f} is below threshold of 0.95. "
            f"Model performance is insufficient."
        )
    
    def test_precision_recall_balance(self, model_artifacts):
        """
        FAIL if precision and recall differ by more than 10%.
        
        Model should have balanced precision and recall.
        """
        precision = model_artifacts['metrics']['test_precision_weighted']
        recall = model_artifacts['metrics']['test_recall_weighted']
        diff = abs(precision - recall)
        
        assert diff <= 0.10, (
            f"Precision ({precision:.4f}) and recall ({recall:.4f}) differ by {diff:.4f}. "
            f"Model is imbalanced. Difference should be <= 0.10."
        )


# ==================== Additional Validation Tests ====================

def test_all_required_files_exist():
    """
    FAIL if any required model files are missing.
    
    All artifacts must be saved.
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    
    required_files = [
        'random_forest_model.pkl',
        'model_metadata.json',
        'feature_importance.json',
        'full_metrics.json'
    ]
    
    for filename in required_files:
        filepath = models_dir / filename
        assert filepath.exists(), (
            f"Required file '{filename}' not found at {filepath}. "
            f"Model artifacts are incomplete."
        )


def test_model_actual_performance_on_test_set():
    """
    FAIL if model doesn't achieve expected accuracy on actual test set.
    
    This is an end-to-end test that loads everything and validates performance.
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    data_dir = project_root / 'data'
    
    # Load model
    model = joblib.load(models_dir / 'random_forest_model.pkl')
    
    # Load test data
    test_data = pd.read_csv(data_dir / 'processed' / 'preprocessed_test.csv')
    X_test = test_data.drop('label_encoded', axis=1)
    y_test = test_data['label_encoded'].values
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy >= 0.95, (
        f"End-to-end test accuracy {accuracy:.4f} is below threshold of 0.95. "
        f"Model does not meet production standards."
    )
