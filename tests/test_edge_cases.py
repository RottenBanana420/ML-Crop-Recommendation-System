"""
Edge Case and Boundary Condition Tests

These tests validate system behavior with extreme and unusual inputs.
Tests are designed to FAIL if edge cases are not handled properly.

CRITICAL: If any test fails, fix the CODEBASE to handle edge cases, NOT the tests.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import json


class TestExtremeInputValues:
    """Test handling of extreme input values."""
    
    @pytest.fixture(scope='class')
    def model_components(self):
        """Load model components."""
        model = joblib.load('models/production_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        return model, scaler, label_encoder
    
    def test_minimum_values(self, model_components):
        """Test with minimum possible values."""
        model, scaler, _ = model_components
        
        # Minimum values for each feature (after engineering)
        X_min = np.array([[0.0] * 22])  # All zeros
        
        # Should not crash
        prediction = model.predict(X_min)
        assert len(prediction) == 1, "Prediction failed with minimum values"
        assert 0 <= prediction[0] < 22, "Invalid prediction with minimum values"
    
    def test_maximum_values(self, model_components):
        """Test with maximum possible values."""
        model, scaler, _ = model_components
        
        # Very large values (but not infinite)
        X_max = np.array([[1000.0] * 22])
        
        # Should not crash
        prediction = model.predict(X_max)
        assert len(prediction) == 1, "Prediction failed with maximum values"
        assert 0 <= prediction[0] < 22, "Invalid prediction with maximum values"
    
    def test_negative_values(self, model_components):
        """Test with negative values (invalid but should handle gracefully)."""
        model, scaler, _ = model_components
        
        # Negative values
        X_neg = np.array([[-10.0] * 22])
        
        # Should not crash (model should handle or scaler should clip)
        prediction = model.predict(X_neg)
        assert len(prediction) == 1, "Prediction failed with negative values"
    
    def test_very_small_values(self, model_components):
        """Test with very small but non-zero values."""
        model, scaler, _ = model_components
        
        # Very small values
        X_small = np.array([[0.0001] * 22])
        
        prediction = model.predict(X_small)
        assert len(prediction) == 1, "Prediction failed with very small values"
        assert 0 <= prediction[0] < 22, "Invalid prediction with very small values"
    
    def test_mixed_extreme_values(self, model_components):
        """Test with mix of extreme values."""
        model, scaler, _ = model_components
        
        # Mix of very small and very large
        X_mixed = np.array([[0.001, 1000, 0.1, 500, 0.01, 100, 0.5, 200, 1, 50, 0.2, 
                            300, 0.05, 150, 0.3, 400, 0.001, 600, 0.4, 250, 0.6, 350]])
        
        prediction = model.predict(X_mixed)
        assert len(prediction) == 1, "Prediction failed with mixed extreme values"


class TestInvalidInputTypes:
    """Test handling of invalid input types."""
    
    def test_string_input(self):
        """Test that string inputs are rejected."""
        model = joblib.load('models/production_model.pkl')
        
        # String input
        X_str = np.array([["invalid"] * 22])
        
        with pytest.raises((ValueError, TypeError)):
            model.predict(X_str)
    
    def test_none_input(self):
        """Test that None input is rejected."""
        model = joblib.load('models/production_model.pkl')
        
        with pytest.raises((ValueError, TypeError, AttributeError)):
            model.predict(None)
    
    def test_empty_input(self):
        """Test that empty input is rejected."""
        model = joblib.load('models/production_model.pkl')
        
        X_empty = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            model.predict(X_empty)
    
    def test_wrong_dimensions(self):
        """Test that wrong dimensions are rejected."""
        model = joblib.load('models/production_model.pkl')
        
        # Wrong number of features
        X_wrong = np.array([[1.0, 2.0, 3.0]])  # Only 3 features instead of 22
        
        with pytest.raises((ValueError, IndexError)):
            model.predict(X_wrong)
    
    def test_1d_array(self):
        """Test that 1D array is handled."""
        model = joblib.load('models/production_model.pkl')
        
        # 1D array (should be 2D)
        X_1d = np.array([1.0] * 22)
        
        # Should either reshape or raise error
        try:
            prediction = model.predict(X_1d.reshape(1, -1))
            assert len(prediction) == 1
        except (ValueError, IndexError):
            # Acceptable to raise error for wrong shape
            pass


class TestNaNAndInfValues:
    """Test handling of NaN and Inf values."""
    
    def test_nan_values(self):
        """Test that NaN values are handled."""
        model = joblib.load('models/production_model.pkl')
        
        # NaN values
        X_nan = np.array([[np.nan] * 22])
        
        # Should either handle or raise clear error
        with pytest.raises((ValueError, RuntimeError)):
            model.predict(X_nan)
    
    def test_inf_values(self):
        """Test that Inf values are handled."""
        model = joblib.load('models/production_model.pkl')
        
        # Inf values
        X_inf = np.array([[np.inf] * 22])
        
        # Should either handle or raise clear error
        with pytest.raises((ValueError, RuntimeError)):
            model.predict(X_inf)
    
    def test_mixed_nan_values(self):
        """Test with some NaN values mixed with valid values."""
        model = joblib.load('models/production_model.pkl')
        
        # Mix of valid and NaN
        X_mixed = np.random.rand(1, 22)
        X_mixed[0, 5] = np.nan
        X_mixed[0, 10] = np.nan
        
        with pytest.raises((ValueError, RuntimeError)):
            model.predict(X_mixed)


class TestFlaskAPIEdgeCases:
    """Test Flask API edge cases."""
    
    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        from app import create_app
        app = create_app('testing')
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_missing_required_field(self, client):
        """Test API with missing required field."""
        payload = {
            'N': 90,
            'P': 42,
            # Missing K
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        assert response.status_code == 400, "Should reject missing field"
    
    def test_extra_fields(self, client):
        """Test API with extra unexpected fields."""
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93,
            'extra_field': 'should_be_ignored'
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        # Should either accept (ignoring extra) or reject
        assert response.status_code in [200, 400], "Unexpected status code for extra fields"
    
    def test_string_instead_of_number(self, client):
        """Test API with string instead of number."""
        payload = {
            'N': "ninety",  # String instead of number
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        assert response.status_code == 400, "Should reject string instead of number"
    
    def test_negative_values_api(self, client):
        """Test API with negative values."""
        payload = {
            'N': -10,  # Negative value
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        assert response.status_code == 400, "Should reject negative values"
    
    def test_out_of_range_humidity(self, client):
        """Test API with out-of-range humidity."""
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 150.00,  # > 100
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        assert response.status_code == 400, "Should reject humidity > 100"
    
    def test_out_of_range_ph(self, client):
        """Test API with out-of-range pH."""
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 15.0,  # > 14
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        assert response.status_code == 400, "Should reject pH > 14"
    
    def test_empty_json(self, client):
        """Test API with empty JSON."""
        response = client.post('/predict/api/crop',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 400, "Should reject empty JSON"
    
    def test_malformed_json(self, client):
        """Test API with malformed JSON."""
        response = client.post('/predict/api/crop',
                              data="not valid json",
                              content_type='application/json')
        
        assert response.status_code == 400, "Should reject malformed JSON"
    
    def test_null_values(self, client):
        """Test API with null values."""
        payload = {
            'N': None,  # null
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        assert response.status_code == 400, "Should reject null values"


class TestBoundaryConditions:
    """Test boundary conditions for features."""
    
    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        from app import create_app
        app = create_app('testing')
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_zero_nitrogen(self, client):
        """Test with zero nitrogen."""
        payload = {
            'N': 0,  # Boundary: zero
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        # Should either accept or reject gracefully
        assert response.status_code in [200, 400], "Unexpected response for zero nitrogen"
    
    def test_zero_rainfall(self, client):
        """Test with zero rainfall."""
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 0.0  # Boundary: zero rainfall
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        # Should handle (some crops tolerate drought)
        assert response.status_code in [200, 400], "Unexpected response for zero rainfall"
    
    def test_extreme_temperature_cold(self, client):
        """Test with extreme cold temperature."""
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': -10.0,  # Very cold
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        # Should reject or handle
        assert response.status_code in [200, 400], "Unexpected response for extreme cold"
    
    def test_extreme_temperature_hot(self, client):
        """Test with extreme hot temperature."""
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 60.0,  # Very hot
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        # Should reject or handle
        assert response.status_code in [200, 400], "Unexpected response for extreme heat"


class TestUnicodeAndSpecialCharacters:
    """Test handling of unicode and special characters."""
    
    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        from app import create_app
        app = create_app('testing')
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_unicode_in_request(self, client):
        """Test that unicode characters are handled."""
        # This shouldn't happen in normal use, but test robustness
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93,
            'comment': '测试'  # Chinese characters
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json; charset=utf-8')
        
        # Should handle gracefully
        assert response.status_code in [200, 400], "Failed to handle unicode"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
