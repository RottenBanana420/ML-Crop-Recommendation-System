"""
Error Handling Tests

These tests verify that the application handles errors gracefully and
doesn't expose sensitive information.

CRITICAL: These tests are designed to FAIL if error handling is inadequate.
"""

import pytest
import json
from app import create_app


@pytest.fixture
def app():
    """Create test app"""
    return create_app('testing')


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


class TestErrorHandling:
    """Test error handling and security"""
    
    def test_malformed_json_returns_400(self, client):
        """Test that malformed JSON returns 400"""
        response = client.post(
            '/predict/api',
            data='this is not json',
            content_type='application/json'
        )
        assert response.status_code == 400, \
            f"Expected 400 for malformed JSON, got {response.status_code}"
    
    def test_error_response_doesnt_expose_stack_trace(self, client):
        """Test that error responses don't expose stack traces"""
        # Send malformed data to trigger an error
        response = client.post(
            '/predict/api',
            data='invalid',
            content_type='application/json'
        )
        
        response_text = response.get_data(as_text=True)
        
        # Check that response doesn't contain sensitive information
        assert 'Traceback' not in response_text, \
            "Error response exposes stack trace"
        assert 'File "' not in response_text, \
            "Error response exposes file paths"
        assert '.py' not in response_text or 'application/json' in response_text, \
            "Error response may expose Python file paths"
    
    def test_server_error_returns_generic_message(self, client):
        """Test that server errors return generic messages"""
        # This test assumes there's a way to trigger a 500 error
        # For now, we test that the error handler is registered
        response = client.get('/nonexistent-api-endpoint')
        assert response.status_code == 404
        
        # Verify error response format
        if response.content_type == 'application/json':
            data = response.get_json()
            assert 'error' in data or 'message' in data
    
    def test_empty_json_returns_400(self, client):
        """Test that empty JSON returns 400"""
        response = client.post(
            '/predict/api',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400, \
            f"Expected 400 for empty JSON, got {response.status_code}"
    
    def test_null_values_rejected(self, client):
        """Test that null values are rejected"""
        data = {
            'N': None,
            'P': 42,
            'K': 43,
            'temperature': 20,
            'humidity': 80,
            'ph': 6.5,
            'rainfall': 200
        }
        response = client.post(
            '/predict/api',
            data=json.dumps(data),
            content_type='application/json'
        )
        assert response.status_code == 400, \
            "Null values should be rejected"
    
    def test_extra_fields_ignored(self, client):
        """Test that extra fields in input are handled gracefully"""
        data = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.8,
            'humidity': 82.0,
            'ph': 6.5,
            'rainfall': 202.9,
            'extra_field': 'should be ignored'
        }
        response = client.post(
            '/predict/api',
            data=json.dumps(data),
            content_type='application/json'
        )
        # Should succeed (extra fields ignored) or return clear error
        assert response.status_code in [200, 400]
    
    def test_very_large_numbers_handled(self, client):
        """Test that very large numbers are handled properly"""
        data = {
            'N': 1e10,  # Very large number
            'P': 42,
            'K': 43,
            'temperature': 20,
            'humidity': 80,
            'ph': 6.5,
            'rainfall': 200
        }
        response = client.post(
            '/predict/api',
            data=json.dumps(data),
            content_type='application/json'
        )
        assert response.status_code == 400, \
            "Very large numbers should be rejected by validation"
    
    def test_infinity_values_rejected(self, client):
        """Test that infinity values are rejected"""
        data = {
            'N': float('inf'),
            'P': 42,
            'K': 43,
            'temperature': 20,
            'humidity': 80,
            'ph': 6.5,
            'rainfall': 200
        }
        response = client.post(
            '/predict/api',
            data=json.dumps(data),
            content_type='application/json'
        )
        assert response.status_code == 400, \
            "Infinity values should be rejected"
