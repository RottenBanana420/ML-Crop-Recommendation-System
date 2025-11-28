"""
API Contract Tests

These tests verify that the prediction API endpoints return correct
status codes, response formats, and that predictions match direct model calls.

CRITICAL: These tests are designed to FAIL if the API doesn't work correctly.
"""

import pytest
import json
import joblib
import numpy as np
from app import create_app
from app.services.model_service import get_model_service


@pytest.fixture
def app():
    """Create test app"""
    app = create_app('testing')
    with app.app_context():
        model_service = get_model_service()
        model_service.load_models(app.config)
    yield app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


# Valid test input
VALID_INPUT = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.8,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.9
}


class TestPredictionAPI:
    """Test prediction API endpoints"""
    
    def test_valid_prediction_returns_200(self, client):
        """Test that valid input returns 200 status code"""
        response = client.post(
            '/predict/api',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}. Response: {response.get_json()}"
    
    def test_response_has_required_fields(self, client):
        """Test that response includes all required fields"""
        response = client.post(
            '/predict/api',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        data = response.get_json()
        
        assert 'success' in data, "Response missing 'success' field"
        assert data['success'] is True, "Success should be True"
        assert 'result' in data, "Response missing 'result' field"
        
        result = data['result']
        assert 'crop' in result, "Result missing 'crop' field"
        assert 'confidence' in result, "Result missing 'confidence' field"
        assert 'alternatives' in result, "Result missing 'alternatives' field"
    
    def test_confidence_in_valid_range(self, client):
        """Test that confidence score is between 0 and 1"""
        response = client.post(
            '/predict/api',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        data = response.get_json()
        confidence = data['result']['confidence']
        
        assert 0 <= confidence <= 1, \
            f"Confidence {confidence} not in range [0, 1]"
    
    def test_alternatives_are_valid(self, client):
        """Test that alternatives list contains valid crops"""
        response = client.post(
            '/predict/api',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        data = response.get_json()
        alternatives = data['result']['alternatives']
        
        assert isinstance(alternatives, list), "Alternatives should be a list"
        assert len(alternatives) > 0, "Alternatives list should not be empty"
        
        for alt in alternatives:
            assert 'crop' in alt, "Alternative missing 'crop' field"
            assert 'confidence' in alt, "Alternative missing 'confidence' field"
            assert 0 <= alt['confidence'] <= 1, \
                f"Alternative confidence {alt['confidence']} not in range [0, 1]"
    
    def test_invalid_input_returns_400(self, client):
        """Test that invalid input returns 400 status code"""
        invalid_input = VALID_INPUT.copy()
        invalid_input['N'] = -10  # Invalid: below minimum
        
        response = client.post(
            '/predict/api',
            data=json.dumps(invalid_input),
            content_type='application/json'
        )
        assert response.status_code == 400, \
            f"Expected 400 for invalid input, got {response.status_code}"
    
    def test_missing_field_returns_400(self, client):
        """Test that missing required field returns 400"""
        incomplete_input = VALID_INPUT.copy()
        del incomplete_input['N']
        
        response = client.post(
            '/predict/api',
            data=json.dumps(incomplete_input),
            content_type='application/json'
        )
        assert response.status_code == 400, \
            f"Expected 400 for missing field, got {response.status_code}"
        
        data = response.get_json()
        assert 'errors' in data or 'error' in data, \
            "Error response should include error details"
    
    def test_malformed_json_returns_400(self, client):
        """Test that malformed JSON returns 400"""
        response = client.post(
            '/predict/api',
            data='not valid json',
            content_type='application/json'
        )
        assert response.status_code == 400, \
            f"Expected 400 for malformed JSON, got {response.status_code}"
    
    def test_prediction_matches_direct_model_call(self, app, client):
        """
        CRITICAL TEST: Verify API predictions match direct model calls.
        This ensures the API is using the same logic as the model.
        """
        # Get prediction from API
        response = client.post(
            '/predict/api',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        api_result = response.get_json()['result']
        api_crop = api_result['crop']
        
        # Make direct model prediction
        with app.app_context():
            from app.services.prediction_service import PredictionService
            prediction_service = PredictionService(app.config)
            direct_result = prediction_service.predict(VALID_INPUT)
            direct_crop = direct_result['crop']
        
        assert api_crop == direct_crop, \
            f"API prediction '{api_crop}' doesn't match direct model prediction '{direct_crop}'"
    
    def test_non_numeric_value_returns_400(self, client):
        """Test that non-numeric values are rejected"""
        invalid_input = VALID_INPUT.copy()
        invalid_input['N'] = 'abc'
        
        response = client.post(
            '/predict/api',
            data=json.dumps(invalid_input),
            content_type='application/json'
        )
        assert response.status_code == 400, \
            f"Expected 400 for non-numeric value, got {response.status_code}"
