"""
Comprehensive Integration Tests for ML Crop Recommendation System

These tests validate end-to-end workflows and component interactions.
Tests are designed to FAIL if any integration issues exist.

CRITICAL: If any test fails, modify the CODEBASE, NOT the tests.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import json
import os
from pathlib import Path


class TestDataToModelPipeline:
    """Test complete pipeline from raw data to model predictions."""
    
    def test_complete_prediction_pipeline(self):
        """Test end-to-end pipeline: data → preprocessing → model → prediction."""
        from src.data.loader import CropDataLoader
        from src.features.preprocessing import CropPreprocessor
        
        # Load data
        loader = CropDataLoader()
        df = loader.load_data()
        
        # Preprocess
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
        
        # Load production model
        model = joblib.load('models/production_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Validate
        assert len(predictions) == len(y_test), "Prediction count mismatch"
        assert all(0 <= p < 22 for p in predictions), "Invalid prediction values"
        
        # Decode predictions
        crop_names = label_encoder.inverse_transform(predictions)
        assert all(isinstance(name, str) for name in crop_names), "Invalid crop names"
        
        # Check accuracy
        accuracy = (predictions == y_test).mean()
        assert accuracy >= 0.95, f"Pipeline accuracy {accuracy:.2%} below 95% threshold"
    
    def test_preprocessing_consistency(self):
        """Test that preprocessing produces consistent results."""
        from src.data.loader import CropDataLoader
        from src.features.preprocessing import CropPreprocessor
        
        loader = CropDataLoader()
        df = loader.load_data()
        
        # Run preprocessing twice with same random state
        prep1 = CropPreprocessor(test_size=0.2, random_state=42)
        X_train1, X_test1, y_train1, y_test1 = prep1.fit_transform(df)
        
        prep2 = CropPreprocessor(test_size=0.2, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = prep2.fit_transform(df)
        
        # Verify consistency
        np.testing.assert_array_equal(X_train1, X_train2, err_msg="Training data inconsistent")
        np.testing.assert_array_equal(X_test1, X_test2, err_msg="Test data inconsistent")
        np.testing.assert_array_equal(y_train1, y_train2, err_msg="Training labels inconsistent")
        np.testing.assert_array_equal(y_test1, y_test2, err_msg="Test labels inconsistent")
    
    def test_model_persistence_and_loading(self):
        """Test that saved models can be loaded and produce same results."""
        from src.data.loader import CropDataLoader
        from src.features.preprocessing import CropPreprocessor
        
        loader = CropDataLoader()
        df = loader.load_data()
        
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
        
        # Load model
        model = joblib.load('models/production_model.pkl')
        predictions1 = model.predict(X_test)
        
        # Reload model
        model_reloaded = joblib.load('models/production_model.pkl')
        predictions2 = model_reloaded.predict(X_test)
        
        # Verify same predictions
        np.testing.assert_array_equal(predictions1, predictions2, 
                                     err_msg="Model predictions inconsistent after reload")


class TestFlaskAPIIntegration:
    """Test Flask application integration."""
    
    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        from app import create_app
        app = create_app('testing')
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/predict/api/health')
        assert response.status_code == 200, "Health check failed"
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy', "Service not healthy"
        assert data['model_loaded'] is True, "Model not loaded"
    
    def test_prediction_endpoint_valid_input(self, client):
        """Test prediction endpoint with valid input."""
        payload = {
            'N': 90,
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
        
        assert response.status_code == 200, f"Prediction failed: {response.data}"
        
        data = json.loads(response.data)
        assert 'prediction' in data, "No prediction in response"
        assert 'confidence' in data, "No confidence in response"
        assert 'probabilities' in data, "No probabilities in response"
        
        # Validate prediction
        assert isinstance(data['prediction'], str), "Prediction not a string"
        assert 0 <= data['confidence'] <= 1, "Invalid confidence value"
        assert len(data['probabilities']) == 22, "Wrong number of probabilities"
    
    def test_prediction_endpoint_invalid_input(self, client):
        """Test prediction endpoint with invalid input."""
        # Missing required field
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
        
        assert response.status_code == 400, "Should reject invalid input"
    
    def test_prediction_endpoint_out_of_range(self, client):
        """Test prediction endpoint with out-of-range values."""
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 150.00,  # Invalid: > 100
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        response = client.post('/predict/api/crop',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        assert response.status_code == 400, "Should reject out-of-range values"
    
    def test_rate_limiting(self, client):
        """Test that rate limiting works."""
        # Make multiple requests
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        # First few requests should succeed
        for _ in range(5):
            response = client.post('/predict/api/crop',
                                  data=json.dumps(payload),
                                  content_type='application/json')
            assert response.status_code == 200, "Valid request failed"
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get('/predict/api/health')
        assert 'Access-Control-Allow-Origin' in response.headers, "CORS headers missing"


class TestModelComparisonWorkflow:
    """Test model comparison workflow."""
    
    def test_model_comparison_files_exist(self):
        """Test that model comparison files exist."""
        assert os.path.exists('models/model_comparison.json'), "Comparison JSON missing"
        assert os.path.exists('models/production_model_metadata.json'), "Production metadata missing"
    
    def test_model_comparison_data_validity(self):
        """Test that model comparison data is valid."""
        with open('models/model_comparison.json', 'r') as f:
            comparison = json.load(f)
        
        # Check required sections
        assert 'accuracy_comparison' in comparison, "Missing accuracy comparison"
        assert 'speed_comparison' in comparison, "Missing speed comparison"
        assert 'memory_comparison' in comparison, "Missing memory comparison"
        
        # Check both models present
        assert 'random_forest' in comparison['accuracy_comparison'], "Missing Random Forest data"
        assert 'xgboost' in comparison['accuracy_comparison'], "Missing XGBoost data"
        
        # Validate accuracy values
        rf_acc = comparison['accuracy_comparison']['random_forest']['test_accuracy']
        xgb_acc = comparison['accuracy_comparison']['xgboost']['test_accuracy']
        
        assert 0.95 <= rf_acc <= 1.0, f"Random Forest accuracy {rf_acc} out of range"
        assert 0.95 <= xgb_acc <= 1.0, f"XGBoost accuracy {xgb_acc} out of range"
    
    def test_production_model_selection(self):
        """Test that production model was properly selected."""
        with open('models/production_model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        assert 'selected_model' in metadata, "No selected model"
        assert metadata['selected_model'] in ['Random Forest', 'XGBoost'], "Invalid model selection"
        assert metadata['production_ready'] is True, "Model not production ready"
        
        # Verify all criteria met
        assert metadata['meets_accuracy_threshold'] is True, "Accuracy threshold not met"
        assert metadata['meets_per_class_threshold'] is True, "Per-class threshold not met"
        assert metadata['meets_latency_threshold'] is True, "Latency threshold not met"
        assert metadata['meets_size_threshold'] is True, "Size threshold not met"


class TestErrorPropagation:
    """Test error handling across the stack."""
    
    def test_invalid_data_handling(self):
        """Test that invalid data is handled gracefully."""
        from src.data.loader import CropDataLoader
        
        loader = CropDataLoader()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            loader.data_path = 'nonexistent.csv'
            loader.load_data()
    
    def test_invalid_model_path(self):
        """Test handling of invalid model path."""
        with pytest.raises(FileNotFoundError):
            joblib.load('models/nonexistent_model.pkl')
    
    def test_prediction_with_wrong_features(self):
        """Test prediction with wrong number of features."""
        model = joblib.load('models/production_model.pkl')
        
        # Wrong number of features (should be 22)
        X_wrong = np.random.rand(10, 10)
        
        with pytest.raises((ValueError, IndexError)):
            model.predict(X_wrong)


class TestDataPersistence:
    """Test data persistence and retrieval."""
    
    def test_preprocessed_data_saved(self):
        """Test that preprocessed data is saved correctly."""
        assert os.path.exists('data/processed/X_train.npy'), "X_train not saved"
        assert os.path.exists('data/processed/X_test.npy'), "X_test not saved"
        assert os.path.exists('data/processed/y_train.npy'), "y_train not saved"
        assert os.path.exists('data/processed/y_test.npy'), "y_test not saved"
    
    def test_model_metadata_completeness(self):
        """Test that model metadata is complete."""
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        required_fields = [
            'model_type',
            'training_date',
            'hyperparameters',
            'metrics',
            'feature_count',
            'class_count'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        # Validate metrics
        metrics = metadata['metrics']
        assert 'test_accuracy' in metrics, "Missing test accuracy"
        assert 'train_accuracy' in metrics, "Missing train accuracy"
        assert 'cv_mean' in metrics, "Missing CV mean"
    
    def test_feature_importance_saved(self):
        """Test that feature importance is saved."""
        assert os.path.exists('models/feature_importance.json'), "Feature importance not saved"
        
        with open('models/feature_importance.json', 'r') as f:
            importance = json.load(f)
        
        assert len(importance) == 22, "Wrong number of features"
        
        # Verify importance values sum to ~1.0
        total_importance = sum(imp for _, imp in importance)
        assert 0.99 <= total_importance <= 1.01, f"Feature importance sum {total_importance} != 1.0"


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""
    
    def test_new_farmer_prediction_scenario(self):
        """Simulate a new farmer getting a crop recommendation."""
        # Farmer's soil and climate data
        farmer_data = {
            'N': 85,
            'P': 58,
            'K': 41,
            'temperature': 21.5,
            'humidity': 75.0,
            'ph': 6.8,
            'rainfall': 180.0
        }
        
        # Load models
        model = joblib.load('models/production_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        # Preprocess (need to engineer features)
        from src.features.preprocessing import CropPreprocessor
        preprocessor = CropPreprocessor()
        
        # Create DataFrame
        df = pd.DataFrame([farmer_data])
        
        # Engineer features
        df_engineered = preprocessor._engineer_features(df)
        
        # Scale
        X_scaled = scaler.transform(df_engineered)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Decode
        crop = label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities.max()
        
        # Validate
        assert isinstance(crop, str), "Invalid crop type"
        assert crop in label_encoder.classes_, "Unknown crop"
        assert 0 <= confidence <= 1, "Invalid confidence"
        assert confidence >= 0.5, "Low confidence prediction"
    
    def test_batch_prediction_scenario(self):
        """Test batch predictions for multiple farmers."""
        # Multiple farmers' data
        batch_data = pd.DataFrame([
            {'N': 85, 'P': 58, 'K': 41, 'temperature': 21.5, 'humidity': 75.0, 'ph': 6.8, 'rainfall': 180.0},
            {'N': 120, 'P': 40, 'K': 50, 'temperature': 28.0, 'humidity': 65.0, 'ph': 7.2, 'rainfall': 220.0},
            {'N': 60, 'P': 70, 'K': 30, 'temperature': 18.0, 'humidity': 85.0, 'ph': 6.0, 'rainfall': 150.0},
        ])
        
        # Load models
        model = joblib.load('models/production_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        # Preprocess
        from src.features.preprocessing import CropPreprocessor
        preprocessor = CropPreprocessor()
        
        df_engineered = preprocessor._engineer_features(batch_data)
        X_scaled = scaler.transform(df_engineered)
        
        # Predict
        predictions = model.predict(X_scaled)
        crops = label_encoder.inverse_transform(predictions)
        
        # Validate
        assert len(crops) == 3, "Wrong number of predictions"
        assert all(isinstance(crop, str) for crop in crops), "Invalid crop types"
        assert all(crop in label_encoder.classes_ for crop in crops), "Unknown crops"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
