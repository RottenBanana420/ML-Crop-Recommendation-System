"""
Performance Tests

These tests verify that the application meets performance requirements.

CRITICAL: These tests are designed to FAIL if performance is inadequate.
"""

import pytest
import time
import json
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


VALID_INPUT = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.8,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.9
}


class TestPerformance:
    """Test application performance"""
    
    def test_single_prediction_under_100ms(self, client):
        """Test that single prediction completes in under 100ms"""
        start_time = time.time()
        
        response = client.post(
            '/predict/api',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert response.status_code == 200, "Prediction should succeed"
        assert elapsed_ms < 100, \
            f"Prediction took {elapsed_ms:.2f}ms, expected < 100ms"
    
    def test_prediction_time_reported(self, client):
        """Test that prediction time is reported in response"""
        response = client.post(
            '/predict/api',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        
        data = response.get_json()
        assert 'result' in data
        assert 'prediction_time_ms' in data['result'], \
            "Response should include prediction_time_ms"
        
        prediction_time = data['result']['prediction_time_ms']
        assert prediction_time > 0, "Prediction time should be positive"
        assert prediction_time < 100, \
            f"Prediction time {prediction_time}ms exceeds 100ms threshold"
    
    def test_multiple_predictions_consistent_performance(self, client):
        """Test that multiple predictions have consistent performance"""
        times = []
        
        for _ in range(10):
            start_time = time.time()
            response = client.post(
                '/predict/api',
                data=json.dumps(VALID_INPUT),
                content_type='application/json'
            )
            elapsed_ms = (time.time() - start_time) * 1000
            times.append(elapsed_ms)
            assert response.status_code == 200
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        assert avg_time < 100, \
            f"Average prediction time {avg_time:.2f}ms exceeds 100ms"
        assert max_time < 200, \
            f"Maximum prediction time {max_time:.2f}ms is too high"
    
    def test_health_check_fast(self, client):
        """Test that health check endpoint is fast"""
        start_time = time.time()
        response = client.get('/health')
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert elapsed_ms < 50, \
            f"Health check took {elapsed_ms:.2f}ms, expected < 50ms"
