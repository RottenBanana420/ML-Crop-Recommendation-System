"""
Flask Application Integration Tests

These tests verify that the Flask app is properly configured and all
components work together correctly.
"""

import pytest
from app import create_app
from app.services.model_service import get_model_service


@pytest.fixture
def app():
    """Create and configure a test app instance"""
    app = create_app('testing')
    
    # Load models for testing
    with app.app_context():
        model_service = get_model_service()
        model_service.load_models(app.config)
    
    yield app


@pytest.fixture
def client(app):
    """Create a test client"""
    return app.test_client()


class TestFlaskApp:
    """Test Flask application setup and configuration"""
    
    def test_app_creation(self, app):
        """Test that app is created correctly"""
        assert app is not None
        assert app.config['TESTING'] is True
    
    def test_blueprints_registered(self, app):
        """Test that all blueprints are registered"""
        blueprint_names = [bp.name for bp in app.blueprints.values()]
        assert 'main' in blueprint_names
        assert 'prediction' in blueprint_names
        assert 'performance' in blueprint_names
    
    def test_health_endpoint(self, client):
        """Test health check endpoint returns 200"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert data['models_loaded'] is True
    
    def test_404_error_handler(self, client):
        """Test 404 error handler returns correct status"""
        response = client.get('/nonexistent-page')
        assert response.status_code == 404
    
    def test_home_page_loads(self, client):
        """Test home page loads successfully"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Crop Recommendation' in response.data
    
    def test_predict_page_loads(self, client):
        """Test prediction page loads successfully"""
        response = client.get('/predict/')
        assert response.status_code == 200
        assert b'Get Crop Recommendation' in response.data
    
    def test_performance_page_loads(self, client):
        """Test performance page loads successfully"""
        response = client.get('/performance/')
        assert response.status_code == 200
        assert b'Performance' in response.data
