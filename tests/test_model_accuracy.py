"""
Strict Model Accuracy Validation Tests

These tests validate that the production model meets ALL accuracy requirements.
Tests are designed to FAIL if model performance degrades below thresholds.

CRITICAL: If any test fails, retrain/improve the MODEL, NOT the tests.
These tests define the MINIMUM acceptable performance.
"""

import pytest
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestProductionModelAccuracy:
    """Test production model accuracy requirements."""
    
    @pytest.fixture(scope='class')
    def test_data_and_model(self):
        """Load test data and production model."""
        # Load test data
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        
        # Load production model
        model = joblib.load('models/production_model.pkl')
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        return X_test, y_test, y_pred, y_proba, model
    
    def test_overall_accuracy_threshold(self, test_data_and_model):
        """Test that overall accuracy meets 95% threshold."""
        _, y_test, y_pred, _, _ = test_data_and_model
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # STRICT REQUIREMENT: >= 95% accuracy
        assert accuracy >= 0.95, (
            f"CRITICAL: Model accuracy {accuracy:.4f} ({accuracy*100:.2f}%) "
            f"is below 95% threshold. Model must be retrained."
        )
    
    def test_target_accuracy_99_percent(self, test_data_and_model):
        """Test that model achieves target 99% accuracy."""
        _, y_test, y_pred, _, _ = test_data_and_model
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # TARGET: >= 99% accuracy (current production standard)
        if accuracy < 0.99:
            pytest.warn(
                UserWarning(
                    f"Model accuracy {accuracy:.4f} ({accuracy*100:.2f}%) "
                    f"is below 99% target. Consider retraining for optimal performance."
                )
            )
    
    def test_per_class_f1_scores(self, test_data_and_model):
        """Test that all per-class F1-scores meet 95% threshold."""
        _, y_test, y_pred, _, _ = test_data_and_model
        
        # Calculate per-class F1-scores
        f1_scores = f1_score(y_test, y_pred, average=None)
        
        # Load crop names
        label_encoder = joblib.load('models/label_encoder.pkl')
        crop_names = label_encoder.classes_
        
        # Check each class
        failing_classes = []
        for i, (crop, f1) in enumerate(zip(crop_names, f1_scores)):
            if f1 < 0.95:
                failing_classes.append((crop, f1))
        
        # STRICT REQUIREMENT: ALL classes >= 95% F1-score
        assert len(failing_classes) == 0, (
            f"CRITICAL: {len(failing_classes)} crop(s) below 95% F1-score threshold:\n" +
            "\n".join([f"  - {crop}: {f1:.4f} ({f1*100:.2f}%)" for crop, f1 in failing_classes]) +
            "\nModel must be retrained to improve performance on these crops."
        )
    
    def test_per_class_precision(self, test_data_and_model):
        """Test that all per-class precision scores meet 90% threshold."""
        _, y_test, y_pred, _, _ = test_data_and_model
        
        # Calculate per-class precision
        precision_scores = precision_score(y_test, y_pred, average=None, zero_division=0)
        
        # Load crop names
        label_encoder = joblib.load('models/label_encoder.pkl')
        crop_names = label_encoder.classes_
        
        # Check each class
        failing_classes = []
        for crop, precision in zip(crop_names, precision_scores):
            if precision < 0.90:
                failing_classes.append((crop, precision))
        
        # STRICT REQUIREMENT: ALL classes >= 90% precision
        assert len(failing_classes) == 0, (
            f"CRITICAL: {len(failing_classes)} crop(s) below 90% precision threshold:\n" +
            "\n".join([f"  - {crop}: {precision:.4f} ({precision*100:.2f}%)" for crop, precision in failing_classes]) +
            "\nModel must be retrained to reduce false positives for these crops."
        )
    
    def test_per_class_recall(self, test_data_and_model):
        """Test that all per-class recall scores meet 90% threshold."""
        _, y_test, y_pred, _, _ = test_data_and_model
        
        # Calculate per-class recall
        recall_scores = recall_score(y_test, y_pred, average=None, zero_division=0)
        
        # Load crop names
        label_encoder = joblib.load('models/label_encoder.pkl')
        crop_names = label_encoder.classes_
        
        # Check each class
        failing_classes = []
        for crop, recall in zip(crop_names, recall_scores):
            if recall < 0.90:
                failing_classes.append((crop, recall))
        
        # STRICT REQUIREMENT: ALL classes >= 90% recall
        assert len(failing_classes) == 0, (
            f"CRITICAL: {len(failing_classes)} crop(s) below 90% recall threshold:\n" +
            "\n".join([f"  - {crop}: {recall:.4f} ({recall*100:.2f}%)" for crop, recall in failing_classes]) +
            "\nModel must be retrained to reduce false negatives for these crops."
        )
    
    def test_prediction_consistency(self, test_data_and_model):
        """Test that predictions are consistent across multiple runs."""
        X_test, y_test, y_pred_original, _, model = test_data_and_model
        
        # Make predictions again
        y_pred_new = model.predict(X_test)
        
        # Predictions should be identical
        consistency = (y_pred_original == y_pred_new).mean()
        
        assert consistency == 1.0, (
            f"CRITICAL: Model predictions are not consistent. "
            f"Only {consistency*100:.2f}% of predictions match. "
            f"This indicates non-deterministic behavior or model corruption."
        )
    
    def test_confidence_scores(self, test_data_and_model):
        """Test that confidence scores are reasonable."""
        _, y_test, y_pred, y_proba, _ = test_data_and_model
        
        # Get confidence for each prediction
        confidences = y_proba.max(axis=1)
        
        # Mean confidence should be reasonably high
        mean_confidence = confidences.mean()
        
        assert mean_confidence >= 0.80, (
            f"WARNING: Mean prediction confidence {mean_confidence:.4f} ({mean_confidence*100:.2f}%) "
            f"is below 80%. Model may be uncertain about predictions."
        )
        
        # For correct predictions, confidence should be even higher
        correct_mask = (y_pred == y_test)
        correct_confidences = confidences[correct_mask]
        mean_correct_confidence = correct_confidences.mean()
        
        assert mean_correct_confidence >= 0.85, (
            f"WARNING: Mean confidence for correct predictions {mean_correct_confidence:.4f} "
            f"({mean_correct_confidence*100:.2f}%) is below 85%."
        )


class TestCrossValidationPerformance:
    """Test cross-validation performance."""
    
    def test_cv_score_threshold(self):
        """Test that cross-validation score meets threshold."""
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        cv_mean = metadata['metrics']['cv_mean']
        
        # STRICT REQUIREMENT: CV score >= 95%
        assert cv_mean >= 0.95, (
            f"CRITICAL: Cross-validation score {cv_mean:.4f} ({cv_mean*100:.2f}%) "
            f"is below 95% threshold. Model generalization is insufficient."
        )
    
    def test_cv_stability(self):
        """Test that cross-validation is stable (low variance)."""
        # Load full metrics
        with open('models/full_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        cv_scores = metrics['cv_scores']
        cv_std = np.std(cv_scores)
        
        # REQUIREMENT: CV std <= 1% (indicates stable model)
        assert cv_std <= 0.01, (
            f"WARNING: Cross-validation standard deviation {cv_std:.4f} ({cv_std*100:.2f}%) "
            f"exceeds 1%. Model performance is unstable across folds."
        )


class TestModelMetadataValidity:
    """Test that model metadata is valid and accurate."""
    
    def test_metadata_accuracy_matches_actual(self):
        """Test that metadata accuracy matches actual test accuracy."""
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        metadata_accuracy = metadata['metrics']['test_accuracy']
        
        # Calculate actual accuracy
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        model = joblib.load('models/production_model.pkl')
        y_pred = model.predict(X_test)
        actual_accuracy = accuracy_score(y_test, y_pred)
        
        # Should match (within floating point precision)
        assert abs(metadata_accuracy - actual_accuracy) < 0.0001, (
            f"CRITICAL: Metadata accuracy {metadata_accuracy:.4f} does not match "
            f"actual accuracy {actual_accuracy:.4f}. Metadata is stale or incorrect."
        )
    
    def test_production_model_metadata_exists(self):
        """Test that production model metadata exists and is valid."""
        with open('models/production_model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Check required fields
        assert 'selected_model' in metadata, "Missing selected_model field"
        assert 'production_ready' in metadata, "Missing production_ready field"
        assert 'meets_accuracy_threshold' in metadata, "Missing meets_accuracy_threshold"
        assert 'meets_per_class_threshold' in metadata, "Missing meets_per_class_threshold"
        assert 'meets_latency_threshold' in metadata, "Missing meets_latency_threshold"
        assert 'meets_size_threshold' in metadata, "Missing meets_size_threshold"
        
        # All criteria must be met
        assert metadata['production_ready'] is True, "Model not marked as production ready"
        assert metadata['meets_accuracy_threshold'] is True, "Accuracy threshold not met"
        assert metadata['meets_per_class_threshold'] is True, "Per-class threshold not met"
        assert metadata['meets_latency_threshold'] is True, "Latency threshold not met"
        assert metadata['meets_size_threshold'] is True, "Size threshold not met"


class TestNoOverfitting:
    """Test that model is not overfitting."""
    
    def test_train_test_gap(self):
        """Test that train-test accuracy gap is acceptable."""
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        train_accuracy = metadata['metrics']['train_accuracy']
        test_accuracy = metadata['metrics']['test_accuracy']
        accuracy_gap = train_accuracy - test_accuracy
        
        # REQUIREMENT: Gap <= 2% (indicates minimal overfitting)
        assert accuracy_gap <= 0.02, (
            f"WARNING: Train-test accuracy gap {accuracy_gap:.4f} ({accuracy_gap*100:.2f}%) "
            f"exceeds 2%. Model may be overfitting."
        )
    
    def test_perfect_training_accuracy_with_good_test(self):
        """Test that perfect training accuracy doesn't indicate overfitting."""
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        train_accuracy = metadata['metrics']['train_accuracy']
        test_accuracy = metadata['metrics']['test_accuracy']
        
        # If training accuracy is perfect, test accuracy should still be very high
        if train_accuracy == 1.0:
            assert test_accuracy >= 0.95, (
                f"CRITICAL: Perfect training accuracy (100%) but test accuracy "
                f"only {test_accuracy:.4f} ({test_accuracy*100:.2f}%). "
                f"This indicates severe overfitting."
            )


class TestPredictionDistribution:
    """Test that predictions are well-distributed."""
    
    def test_no_class_ignored(self):
        """Test that model predicts all classes (no class is ignored)."""
        X_test = np.load('data/processed/X_test.npy')
        model = joblib.load('models/production_model.pkl')
        y_pred = model.predict(X_test)
        
        # Get unique predictions
        unique_predictions = np.unique(y_pred)
        
        # Should predict at least 15 out of 22 classes on test set
        # (some rare classes might not appear in test set)
        assert len(unique_predictions) >= 15, (
            f"WARNING: Model only predicts {len(unique_predictions)} out of 22 classes. "
            f"Some classes may be ignored by the model."
        )
    
    def test_no_single_class_dominance(self):
        """Test that no single class dominates predictions."""
        X_test = np.load('data/processed/X_test.npy')
        model = joblib.load('models/production_model.pkl')
        y_pred = model.predict(X_test)
        
        # Count predictions per class
        unique, counts = np.unique(y_pred, return_counts=True)
        max_count = counts.max()
        total_count = len(y_pred)
        
        # No class should be > 50% of predictions
        max_percentage = max_count / total_count
        
        assert max_percentage <= 0.50, (
            f"WARNING: One class accounts for {max_percentage*100:.2f}% of predictions. "
            f"Model may be biased towards a single class."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
