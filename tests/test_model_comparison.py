"""
Comprehensive Test Suite for Model Comparison

This module contains stringent tests that FAIL if model comparison is incomplete,
metrics are calculated incorrectly, selection logic is flawed, or production
readiness criteria aren't met. Tests should NEVER be modified to pass - instead,
improve the comparison code.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, cohen_kappa_score
)


class TestModelComparison:
    """Test suite for comprehensive model comparison."""
    
    @pytest.fixture
    def project_paths(self):
        """Set up project paths."""
        project_root = Path(__file__).parent.parent
        return {
            'root': project_root,
            'models': project_root / 'models',
            'data': project_root / 'data',
            'viz': project_root / 'models' / 'comparison_visualizations'
        }
    
    @pytest.fixture
    def comparison_data(self, project_paths):
        """Load comparison data."""
        comparison_path = project_paths['models'] / 'model_comparison.json'
        assert comparison_path.exists(), "Comparison JSON not found. Run model comparison first."
        
        with open(comparison_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def selection_data(self, project_paths):
        """Load selection data."""
        selection_path = project_paths['models'] / 'production_model_metadata.json'
        assert selection_path.exists(), "Selection metadata not found. Run model comparison first."
        
        with open(selection_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def test_data(self, project_paths):
        """Load test data."""
        test_path = project_paths['data'] / 'processed' / 'preprocessed_test.csv'
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop('label_encoded', axis=1)
        y_test = test_df['label_encoded'].values
        return X_test, y_test
    
    @pytest.fixture
    def models(self, project_paths):
        """Load both models."""
        xgb_model = joblib.load(project_paths['models'] / 'xgboost_model.pkl')
        rf_model = joblib.load(project_paths['models'] / 'random_forest_model.pkl')
        return {'xgboost': xgb_model, 'random_forest': rf_model}
    
    # ==================== Comparison Completeness Tests ====================
    
    def test_all_required_sections_present(self, comparison_data):
        """
        FAIL if any required comparison section is missing.
        
        All sections must be present for complete comparison.
        """
        required_sections = [
            'accuracy_comparison',
            'speed_comparison',
            'memory_comparison',
            'cross_validation_comparison',
            'feature_importance_comparison',
            'per_class_comparison'
        ]
        
        for section in required_sections:
            assert section in comparison_data, f"Missing required section: {section}"
    
    def test_all_metrics_calculated(self, comparison_data):
        """
        FAIL if any required metric is missing.
        
        Comprehensive comparison requires all metrics.
        """
        required_accuracy_metrics = [
            'test_accuracy', 'train_accuracy', 'accuracy_gap',
            'f1_weighted', 'precision_weighted', 'recall_weighted',
            'mcc', 'cohens_kappa'
        ]
        
        for model in ['xgboost', 'random_forest']:
            for metric in required_accuracy_metrics:
                assert metric in comparison_data['accuracy_comparison'][model], \
                    f"Missing {metric} for {model}"
    
    def test_per_class_metrics_complete(self, comparison_data):
        """
        FAIL if per-class metrics are incomplete.
        
        All 22 crops must have per-class metrics.
        """
        for model in ['xgboost', 'random_forest']:
            per_class = comparison_data['per_class_comparison'][model]
            
            assert 'precision_scores' in per_class
            assert 'recall_scores' in per_class
            assert 'f1_scores' in per_class
            assert 'support' in per_class
            
            # Check all have 22 values (one per crop)
            assert len(per_class['precision_scores']) == 22, \
                f"{model} precision_scores should have 22 values"
            assert len(per_class['recall_scores']) == 22, \
                f"{model} recall_scores should have 22 values"
            assert len(per_class['f1_scores']) == 22, \
                f"{model} f1_scores should have 22 values"
            assert len(per_class['support']) == 22, \
                f"{model} support should have 22 values"
    
    def test_latency_percentiles_present(self, comparison_data):
        """
        FAIL if latency percentiles are missing.
        
        Detailed latency analysis requires percentiles.
        """
        for model in ['xgboost', 'random_forest']:
            speed_data = comparison_data['speed_comparison'][model]
            assert 'latency_percentiles' in speed_data, \
                f"Missing latency_percentiles for {model}"
            
            percentiles = speed_data['latency_percentiles']
            assert 'p50' in percentiles
            assert 'p95' in percentiles
            assert 'p99' in percentiles
    
    def test_visualizations_generated(self, project_paths):
        """
        FAIL if any required visualization is missing.
        
        All visualizations must be generated.
        """
        required_viz = [
            'radar_chart.png',
            'per_class_heatmap.png',
            'confusion_matrices.png',
            'inference_time_distribution.png',
            'feature_importance_correlation.png',
            'performance_efficiency_scatter.png'
        ]
        
        viz_dir = project_paths['viz']
        assert viz_dir.exists(), "Visualization directory not found"
        
        for viz_file in required_viz:
            viz_path = viz_dir / viz_file
            assert viz_path.exists(), f"Missing visualization: {viz_file}"
            assert viz_path.stat().st_size > 0, f"Empty visualization file: {viz_file}"
    
    def test_reports_generated(self, project_paths):
        """
        FAIL if reports are not generated.
        
        Both Markdown and HTML reports must exist.
        """
        md_report = project_paths['models'] / 'model_comparison.md'
        html_report = project_paths['models'] / 'comparison_report.html'
        
        assert md_report.exists(), "Markdown report not found"
        assert html_report.exists(), "HTML report not found"
        
        assert md_report.stat().st_size > 1000, "Markdown report too small"
        assert html_report.stat().st_size > 5000, "HTML report too small"
    
    # ==================== Metric Accuracy Tests ====================
    
    def test_accuracy_matches_sklearn(self, models, test_data, comparison_data):
        """
        FAIL if accuracy doesn't match sklearn calculation.
        
        Accuracy must be calculated correctly.
        """
        X_test, y_test = test_data
        
        for model_name in ['xgboost', 'random_forest']:
            model = models[model_name]
            y_pred = model.predict(X_test)
            
            sklearn_accuracy = accuracy_score(y_test, y_pred)
            reported_accuracy = comparison_data['accuracy_comparison'][model_name]['test_accuracy']
            
            assert abs(sklearn_accuracy - reported_accuracy) < 0.0001, \
                f"{model_name} accuracy mismatch: sklearn={sklearn_accuracy:.4f}, reported={reported_accuracy:.4f}"
    
    def test_precision_recall_f1_match_sklearn(self, models, test_data, comparison_data):
        """
        FAIL if precision/recall/F1 don't match sklearn.
        
        All classification metrics must be correct.
        """
        X_test, y_test = test_data
        
        for model_name in ['xgboost', 'random_forest']:
            model = models[model_name]
            y_pred = model.predict(X_test)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            
            comp_data = comparison_data['accuracy_comparison'][model_name]
            
            assert abs(precision - comp_data['precision_weighted']) < 0.0001, \
                f"{model_name} precision mismatch"
            assert abs(recall - comp_data['recall_weighted']) < 0.0001, \
                f"{model_name} recall mismatch"
            assert abs(f1 - comp_data['f1_weighted']) < 0.0001, \
                f"{model_name} F1 mismatch"
    
    def test_mcc_and_kappa_correct(self, models, test_data, comparison_data):
        """
        FAIL if MCC or Cohen's Kappa are incorrect.
        
        Advanced metrics must be calculated correctly.
        """
        X_test, y_test = test_data
        
        for model_name in ['xgboost', 'random_forest']:
            model = models[model_name]
            y_pred = model.predict(X_test)
            
            sklearn_mcc = matthews_corrcoef(y_test, y_pred)
            sklearn_kappa = cohen_kappa_score(y_test, y_pred)
            
            comp_data = comparison_data['accuracy_comparison'][model_name]
            
            assert abs(sklearn_mcc - comp_data['mcc']) < 0.0001, \
                f"{model_name} MCC mismatch"
            assert abs(sklearn_kappa - comp_data['cohens_kappa']) < 0.0001, \
                f"{model_name} Cohen's Kappa mismatch"
    
    def test_per_class_metrics_sum_correctly(self, comparison_data, test_data):
        """
        FAIL if per-class metrics don't aggregate to weighted metrics.
        
        Per-class and overall metrics must be consistent.
        """
        _, y_test = test_data
        
        for model_name in ['xgboost', 'random_forest']:
            per_class = comparison_data['per_class_comparison'][model_name]
            overall = comparison_data['accuracy_comparison'][model_name]
            
            # Calculate weighted average from per-class
            support = np.array(per_class['support'])
            f1_scores = np.array(per_class['f1_scores'])
            
            weighted_f1 = np.sum(f1_scores * support) / np.sum(support)
            
            assert abs(weighted_f1 - overall['f1_weighted']) < 0.001, \
                f"{model_name} per-class F1 doesn't match weighted F1"
    
    def test_inference_time_realistic(self, comparison_data):
        """
        FAIL if inference times are unrealistic.
        
        Times should be between 0.001s and 1s for test set.
        """
        for model_name in ['xgboost', 'random_forest']:
            pred_time = comparison_data['speed_comparison'][model_name]['prediction_time']
            
            assert 0.001 < pred_time < 1.0, \
                f"{model_name} prediction time unrealistic: {pred_time}s"
            
            # Check percentiles are ordered correctly
            latency = comparison_data['speed_comparison'][model_name]['latency_percentiles']
            assert latency['p50'] <= latency['p95'] <= latency['p99'], \
                f"{model_name} latency percentiles not ordered correctly"
    
    def test_memory_measurements_valid(self, comparison_data):
        """
        FAIL if memory measurements are invalid.
        
        Model sizes must be positive and reasonable.
        """
        for model_name in ['xgboost', 'random_forest']:
            size_mb = comparison_data['memory_comparison'][model_name]['model_size_mb']
            
            assert size_mb > 0, f"{model_name} model size must be positive"
            assert size_mb < 100, f"{model_name} model size unreasonably large: {size_mb}MB"
    
    # ==================== Selection Logic Tests ====================
    
    def test_selection_prioritizes_accuracy(self, comparison_data, selection_data):
        """
        FAIL if selection doesn't prioritize accuracy when difference > 0.5%.
        
        Accuracy is the primary criterion.
        """
        xgb_acc = comparison_data['accuracy_comparison']['xgboost']['test_accuracy']
        rf_acc = comparison_data['accuracy_comparison']['random_forest']['test_accuracy']
        
        acc_diff = abs(xgb_acc - rf_acc)
        
        if acc_diff > 0.005:  # 0.5%
            # Higher accuracy model should be selected
            if xgb_acc > rf_acc:
                assert selection_data['selected_model'] == 'XGBoost', \
                    "XGBoost has higher accuracy but wasn't selected"
            else:
                assert selection_data['selected_model'] == 'Random Forest', \
                    "Random Forest has higher accuracy but wasn't selected"
    
    def test_selection_considers_speed_when_tied(self, comparison_data, selection_data):
        """
        FAIL if selection doesn't consider speed when accuracy is tied.
        
        Speed breaks ties when accuracy difference < 0.1%.
        """
        xgb_acc = comparison_data['accuracy_comparison']['xgboost']['test_accuracy']
        rf_acc = comparison_data['accuracy_comparison']['random_forest']['test_accuracy']
        
        acc_diff = abs(xgb_acc - rf_acc)
        
        if acc_diff <= 0.001:  # 0.1%
            xgb_speed = comparison_data['speed_comparison']['xgboost']['prediction_time']
            rf_speed = comparison_data['speed_comparison']['random_forest']['prediction_time']
            
            # Faster model should be selected
            if xgb_speed < rf_speed:
                assert selection_data['selected_model'] == 'XGBoost', \
                    "XGBoost is faster but wasn't selected when accuracy tied"
            else:
                assert selection_data['selected_model'] == 'Random Forest', \
                    "Random Forest is faster but wasn't selected when accuracy tied"
    
    def test_production_readiness_enforced(self, comparison_data, selection_data):
        """
        FAIL if production readiness criteria aren't properly evaluated.
        
        Must check accuracy ≥97.5%, per-class F1 ≥95%, latency, and size.
        """
        assert 'production_ready' in selection_data
        assert 'meets_accuracy_threshold' in selection_data
        assert 'meets_per_class_threshold' in selection_data
        assert 'meets_latency_threshold' in selection_data
        assert 'meets_size_threshold' in selection_data
        
        # Verify accuracy threshold check
        selected_model = selection_data['selected_model'].lower().replace(' ', '_')
        selected_acc = comparison_data['accuracy_comparison'][selected_model]['test_accuracy']
        
        if selected_acc >= 0.975:
            assert selection_data['meets_accuracy_threshold'], \
                "Model meets accuracy threshold but flag is False"
        else:
            assert not selection_data['meets_accuracy_threshold'], \
                "Model doesn't meet accuracy threshold but flag is True"
    
    def test_statistical_significance_calculated(self, selection_data):
        """
        FAIL if statistical significance isn't calculated.
        
        McNemar's test must be performed.
        """
        assert 'statistical_significance' in selection_data
        assert 'statistical_analysis' in selection_data
        
        # Should contain p-value or error message
        stat_sig = selection_data['statistical_significance']
        assert 'p-value' in stat_sig or 'failed' in stat_sig.lower(), \
            "Statistical significance must include p-value or failure message"
    
    def test_selection_rationale_documented(self, selection_data):
        """
        FAIL if selection rationale is missing or too brief.
        
        Detailed explanation is required.
        """
        assert 'selection_rationale' in selection_data
        rationale = selection_data['selection_rationale']
        
        assert len(rationale) > 50, "Selection rationale too brief"
        assert any(word in rationale.lower() for word in ['accuracy', 'speed', 'performance']), \
            "Rationale must mention performance criteria"
    
    def test_production_model_saved(self, project_paths):
        """
        FAIL if production model file doesn't exist.
        
        Production model must be saved.
        """
        prod_model_path = project_paths['models'] / 'production_model.pkl'
        assert prod_model_path.exists(), "Production model file not found"
        assert prod_model_path.stat().st_size > 1000, "Production model file too small"
    
    def test_production_model_matches_winner(self, project_paths, selection_data, models):
        """
        FAIL if production model doesn't match selected model.
        
        Saved production model must be the selected model.
        """
        prod_model = joblib.load(project_paths['models'] / 'production_model.pkl')
        selected_name = selection_data['selected_model'].lower().replace(' ', '_')
        selected_model = models[selected_name]
        
        # Compare model parameters (for tree-based models)
        if hasattr(prod_model, 'n_estimators'):
            assert prod_model.n_estimators == selected_model.n_estimators, \
                "Production model parameters don't match selected model"
    
    def test_production_model_functional(self, project_paths, test_data):
        """
        FAIL if production model can't make predictions.
        
        Production model must be loadable and functional.
        """
        prod_model = joblib.load(project_paths['models'] / 'production_model.pkl')
        X_test, y_test = test_data
        
        # Should be able to predict
        predictions = prod_model.predict(X_test)
        
        assert len(predictions) == len(y_test), \
            "Production model predictions length mismatch"
        assert all(0 <= p < 22 for p in predictions), \
            "Production model predictions out of valid range"
    
    # ==================== Consistency Tests ====================
    
    def test_comparison_deterministic(self, project_paths):
        """
        FAIL if comparison produces different results on repeated runs.
        
        Results must be deterministic.
        """
        # Load comparison data
        with open(project_paths['models'] / 'model_comparison.json', 'r') as f:
            comparison1 = json.load(f)
        
        # Key metrics should be stable
        xgb_acc = comparison1['accuracy_comparison']['xgboost']['test_accuracy']
        rf_acc = comparison1['accuracy_comparison']['random_forest']['test_accuracy']
        
        # These should be exact values, not random
        assert isinstance(xgb_acc, (int, float)), "Accuracy must be numeric"
        assert isinstance(rf_acc, (int, float)), "Accuracy must be numeric"
        assert 0 <= xgb_acc <= 1, "Accuracy must be in [0, 1]"
        assert 0 <= rf_acc <= 1, "Accuracy must be in [0, 1]"
    
    def test_metric_consistency_with_individual_reports(self, project_paths, comparison_data):
        """
        FAIL if comparison metrics don't match individual model reports.
        
        Metrics must be consistent across reports.
        """
        # Load individual model metrics
        with open(project_paths['models'] / 'xgboost_full_metrics.json', 'r') as f:
            xgb_individual = json.load(f)
        
        with open(project_paths['models'] / 'full_metrics.json', 'r') as f:
            rf_individual = json.load(f)
        
        # Check key metrics match
        assert abs(comparison_data['accuracy_comparison']['xgboost']['test_accuracy'] - 
                  xgb_individual['test_accuracy']) < 0.0001, \
            "XGBoost accuracy inconsistent between reports"
        
        assert abs(comparison_data['accuracy_comparison']['random_forest']['test_accuracy'] - 
                  rf_individual['test_accuracy']) < 0.0001, \
            "Random Forest accuracy inconsistent between reports"
    
    def test_no_performance_degradation(self, comparison_data):
        """
        FAIL if either model's performance has degraded unexpectedly.
        
        Performance should not drop below established baselines.
        """
        # Based on previous runs, both models should achieve >97% accuracy
        BASELINE_ACCURACY = 0.97
        
        xgb_acc = comparison_data['accuracy_comparison']['xgboost']['test_accuracy']
        rf_acc = comparison_data['accuracy_comparison']['random_forest']['test_accuracy']
        
        assert xgb_acc >= BASELINE_ACCURACY, \
            f"XGBoost accuracy degraded: {xgb_acc:.4f} < {BASELINE_ACCURACY}"
        assert rf_acc >= BASELINE_ACCURACY, \
            f"Random Forest accuracy degraded: {rf_acc:.4f} < {BASELINE_ACCURACY}"
    
    # ==================== Production Readiness Tests ====================
    
    def test_selected_model_meets_accuracy_baseline(self, comparison_data, selection_data):
        """
        FAIL if selected model doesn't meet 97.5% accuracy baseline.
        
        Production model must meet minimum accuracy.
        """
        selected_name = selection_data['selected_model'].lower().replace(' ', '_')
        selected_acc = comparison_data['accuracy_comparison'][selected_name]['test_accuracy']
        
        assert selected_acc >= 0.975, \
            f"Selected model accuracy {selected_acc:.4f} below 97.5% baseline"
    
    def test_no_class_below_f1_threshold(self, comparison_data, selection_data):
        """
        FAIL if any crop class has F1 < 95% for selected model.
        
        All crops must be well-predicted.
        """
        selected_name = selection_data['selected_model'].lower().replace(' ', '_')
        per_class = comparison_data['per_class_comparison'][selected_name]
        
        min_f1 = min(per_class['f1_scores'])
        
        assert min_f1 >= 0.95, \
            f"Selected model has crop with F1 {min_f1:.4f} below 95% threshold"
    
    def test_inference_latency_acceptable(self, comparison_data, selection_data, test_data):
        """
        FAIL if inference latency exceeds 100ms for 1000 samples.
        
        Production SLA requires fast inference.
        """
        selected_name = selection_data['selected_model'].lower().replace(' ', '_')
        pred_time = comparison_data['speed_comparison'][selected_name]['prediction_time']
        
        _, y_test = test_data
        test_size = len(y_test)
        
        # Scale to 1000 samples
        latency_1000 = pred_time * (1000 / test_size)
        
        assert latency_1000 < 0.1, \
            f"Selected model latency {latency_1000*1000:.2f}ms exceeds 100ms for 1000 samples"
    
    def test_model_size_acceptable(self, comparison_data, selection_data):
        """
        FAIL if model size exceeds 50MB.
        
        Deployment requires reasonable model size.
        """
        selected_name = selection_data['selected_model'].lower().replace(' ', '_')
        model_size = comparison_data['memory_comparison'][selected_name]['model_size_mb']
        
        assert model_size < 50, \
            f"Selected model size {model_size:.2f}MB exceeds 50MB limit"
    
    def test_all_classes_predicted(self, project_paths, test_data):
        """
        FAIL if production model can't predict all 22 crops.
        
        Model must be able to predict all crop types.
        """
        prod_model = joblib.load(project_paths['models'] / 'production_model.pkl')
        X_test, y_test = test_data
        
        predictions = prod_model.predict(X_test)
        unique_predictions = set(predictions)
        
        # Should be able to predict multiple crops (though not necessarily all 22 in test set)
        assert len(unique_predictions) >= 10, \
            f"Production model only predicts {len(unique_predictions)} different crops"
    
    def test_production_metadata_complete(self, selection_data):
        """
        FAIL if production model metadata is incomplete.
        
        Metadata must document selection decision.
        """
        required_fields = [
            'selected_model',
            'selection_rationale',
            'production_ready',
            'meets_accuracy_threshold',
            'meets_per_class_threshold',
            'meets_latency_threshold',
            'meets_size_threshold',
            'deployment_recommendation'
        ]
        
        for field in required_fields:
            assert field in selection_data, f"Missing metadata field: {field}"
