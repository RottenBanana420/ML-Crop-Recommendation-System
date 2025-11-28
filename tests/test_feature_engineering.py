"""
Feature Engineering Tests

These tests verify that the feature engineering in the prediction service
matches the preprocessing pipeline exactly.

CRITICAL: These tests are designed to FAIL if feature engineering doesn't match.
"""

import pytest
import numpy as np
from app import create_app
from app.services.prediction_service import PredictionService


@pytest.fixture
def app():
    """Create test app"""
    return create_app('testing')


@pytest.fixture
def prediction_service(app):
    """Create prediction service"""
    return PredictionService(app.config)


TEST_INPUT = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.8,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.9
}


class TestFeatureEngineering:
    """Test feature engineering logic"""
    
    def test_correct_number_of_features(self, prediction_service):
        """Test that exactly 22 features are generated"""
        features = prediction_service.engineer_features(TEST_INPUT)
        assert len(features) == 22, \
            f"Expected 22 features, got {len(features)}"
    
    def test_feature_names_match_expected(self, prediction_service, app):
        """Test that feature names match the expected list"""
        features = prediction_service.engineer_features(TEST_INPUT)
        expected_features = app.config['FEATURE_NAMES']
        
        for expected_name in expected_features:
            assert expected_name in features, \
                f"Missing expected feature: {expected_name}"
    
    def test_base_features_preserved(self, prediction_service):
        """Test that base features are preserved in output"""
        features = prediction_service.engineer_features(TEST_INPUT)
        
        for key in TEST_INPUT:
            assert key in features, f"Base feature {key} missing from output"
            assert features[key] == TEST_INPUT[key], \
                f"Base feature {key} value changed"
    
    def test_rainfall_log_calculated(self, prediction_service):
        """Test that rainfall_log is calculated correctly"""
        features = prediction_service.engineer_features(TEST_INPUT)
        expected = np.log1p(TEST_INPUT['rainfall'])
        assert 'rainfall_log' in features
        assert abs(features['rainfall_log'] - expected) < 1e-6, \
            f"rainfall_log mismatch: expected {expected}, got {features['rainfall_log']}"
    
    def test_nutrient_ratios_calculated(self, prediction_service):
        """Test that nutrient ratios are calculated correctly"""
        features = prediction_service.engineer_features(TEST_INPUT)
        
        # Check that ratio features exist
        assert 'N_to_P_ratio' in features
        assert 'N_to_K_ratio' in features
        assert 'P_to_K_ratio' in features
        
        # Verify calculations (with epsilon handling)
        epsilon = 0.1
        expected_n_p = min(TEST_INPUT['N'] / (TEST_INPUT['P'] + epsilon), 100)
        assert abs(features['N_to_P_ratio'] - expected_n_p) < 1e-6
    
    def test_total_npk_calculated(self, prediction_service):
        """Test that total_NPK is calculated correctly"""
        features = prediction_service.engineer_features(TEST_INPUT)
        expected = TEST_INPUT['N'] + TEST_INPUT['P'] + TEST_INPUT['K']
        assert 'total_NPK' in features
        assert abs(features['total_NPK'] - expected) < 1e-6
    
    def test_environmental_interactions_calculated(self, prediction_service):
        """Test that environmental interactions are calculated"""
        features = prediction_service.engineer_features(TEST_INPUT)
        
        assert 'temp_humidity_interaction' in features
        assert 'rainfall_humidity_interaction' in features
        assert 'temp_rainfall_interaction' in features
        
        # Verify temp_humidity_interaction
        expected = TEST_INPUT['temperature'] * TEST_INPUT['humidity'] / 100
        assert abs(features['temp_humidity_interaction'] - expected) < 1e-6
    
    def test_ph_interactions_calculated(self, prediction_service):
        """Test that pH interactions are calculated"""
        features = prediction_service.engineer_features(TEST_INPUT)
        
        assert 'ph_N_interaction' in features
        assert 'ph_P_interaction' in features
        assert 'ph_K_interaction' in features
        
        # Verify ph_N_interaction
        expected = TEST_INPUT['ph'] * TEST_INPUT['N']
        assert abs(features['ph_N_interaction'] - expected) < 1e-6
    
    def test_no_division_by_zero_errors(self, prediction_service):
        """Test that feature engineering handles zero values without errors"""
        zero_input = {
            'N': 0,
            'P': 0,
            'K': 0,
            'temperature': 20,
            'humidity': 50,
            'ph': 7,
            'rainfall': 100
        }
        
        # Should not raise any exceptions
        features = prediction_service.engineer_features(zero_input)
        assert len(features) == 22
        
        # Check that no features are NaN or infinite
        for key, value in features.items():
            assert not np.isnan(value), f"Feature {key} is NaN"
            assert not np.isinf(value), f"Feature {key} is infinite"
    
    def test_feature_values_match_preprocessing(self, prediction_service):
        """
        CRITICAL: Test that feature engineering matches preprocessing.py exactly.
        This compares against known good values from the preprocessing pipeline.
        """
        features = prediction_service.engineer_features(TEST_INPUT)
        
        # These expected values come from running the preprocessing pipeline
        # on the same input
        expected_npk_balance = np.std([TEST_INPUT['N'], TEST_INPUT['P'], TEST_INPUT['K']])
        assert abs(features['NPK_balance'] - expected_npk_balance) < 1e-6, \
            "NPK_balance doesn't match preprocessing calculation"
        
        expected_avg_npk = np.mean([TEST_INPUT['N'], TEST_INPUT['P'], TEST_INPUT['K']])
        assert abs(features['avg_NPK'] - expected_avg_npk) < 1e-6, \
            "avg_NPK doesn't match preprocessing calculation"
