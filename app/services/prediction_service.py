"""
Prediction Service - ML Prediction and Feature Engineering

This module handles feature engineering and prediction generation,
matching the preprocessing pipeline exactly.
"""

import time
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from app.services.model_service import get_model_service


class PredictionService:
    """Service for generating crop predictions with feature engineering"""
    
    def __init__(self, config):
        """
        Initialize prediction service.
        
        Args:
            config: Flask app configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_service = get_model_service()
    
    def engineer_features(self, input_data: Dict[str, float]) -> Dict[str, float]:
        """
        Engineer features from raw input data.
        
        This method replicates the feature engineering from preprocessing.py exactly.
        
        Args:
            input_data: Dictionary with base features (N, P, K, temperature, humidity, ph, rainfall)
        
        Returns:
            Dictionary with all 22 engineered features
        """
        # Extract base features
        N = input_data['N']
        P = input_data['P']
        K = input_data['K']
        temp = input_data['temperature']
        humidity = input_data['humidity']
        ph = input_data['ph']
        rainfall = input_data['rainfall']
        
        # Initialize features dictionary with base features
        features = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temp,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        
        # 1. Log transform rainfall
        features['rainfall_log'] = np.log1p(rainfall)
        
        # 2. Nutrient ratios (with epsilon to avoid division by zero)
        epsilon = 0.1
        features['N_to_P_ratio'] = min(N / (P + epsilon), 100)
        features['N_to_K_ratio'] = min(N / (K + epsilon), 100)
        features['P_to_K_ratio'] = min(P / (K + epsilon), 100)
        
        # 3. Nutrient totals and balance
        features['total_NPK'] = N + P + K
        features['NPK_balance'] = np.std([N, P, K])
        features['avg_NPK'] = np.mean([N, P, K])
        
        # 4. Environmental interactions
        features['temp_humidity_interaction'] = temp * humidity / 100
        features['rainfall_humidity_interaction'] = rainfall * humidity / 100
        features['temp_rainfall_interaction'] = temp * np.log1p(rainfall)
        
        # 5. pH interactions
        features['ph_N_interaction'] = ph * N
        features['ph_P_interaction'] = ph * P
        features['ph_K_interaction'] = ph * K
        
        # 6. Composite indicators
        features['moisture_index'] = (humidity / 100) * np.log1p(rainfall)
        
        # Growing conditions index (normalized)
        # Note: Using reasonable min/max values for normalization
        temp_min, temp_max = 8.8, 43.7
        temp_norm = (temp - temp_min) / (temp_max - temp_min + 1e-10)
        humidity_norm = humidity / 100
        features['growing_conditions_index'] = (temp_norm + humidity_norm) / 2
        
        return features
    
    def predict(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate crop prediction from input data.
        
        Args:
            input_data: Dictionary with base input features
        
        Returns:
            Dictionary with prediction results:
            {
                'crop': str,
                'confidence': float,
                'alternatives': List[Dict],
                'prediction_time_ms': float,
                'input_features': Dict,
                'engineered_features': Dict
            }
        """
        start_time = time.time()
        
        try:
            # Engineer features
            engineered_features = self.engineer_features(input_data)
            
            # Convert to DataFrame with correct feature order
            feature_names = self.model_service.get_feature_names()
            feature_array = np.array([[engineered_features[name] for name in feature_names]])
            
            # Scale features
            scaler = self.model_service.get_scaler()
            scaled_features = scaler.transform(feature_array)
            
            # Make prediction
            model = self.model_service.get_model()
            prediction = model.predict(scaled_features)[0]
            probabilities = model.predict_proba(scaled_features)[0]
            
            # Get crop name
            label_encoder = self.model_service.get_label_encoder()
            crop_name = label_encoder.inverse_transform([prediction])[0]
            confidence = float(probabilities[prediction])
            
            # Get top alternatives (excluding the main prediction)
            top_indices = np.argsort(probabilities)[-4:][::-1]  # Top 4
            alternatives = []
            for idx in top_indices[1:]:  # Skip first (main prediction)
                alt_crop = label_encoder.inverse_transform([idx])[0]
                alt_confidence = float(probabilities[idx])
                alternatives.append({
                    'crop': alt_crop,
                    'confidence': alt_confidence,
                    'confidence_percent': round(alt_confidence * 100, 2)
                })
            
            # Calculate prediction time
            prediction_time_ms = (time.time() - start_time) * 1000
            
            # Log performance warning if too slow
            if prediction_time_ms > self.config['MAX_PREDICTION_TIME_MS']:
                self.logger.warning(
                    f"Prediction took {prediction_time_ms:.2f}ms "
                    f"(threshold: {self.config['MAX_PREDICTION_TIME_MS']}ms)"
                )
            
            return {
                'crop': crop_name,
                'confidence': confidence,
                'confidence_percent': round(confidence * 100, 2),
                'alternatives': alternatives,
                'prediction_time_ms': round(prediction_time_ms, 2),
                'input_features': input_data,
                'engineered_features': {k: round(v, 4) for k, v in engineered_features.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, batch_data: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Generate predictions for a batch of inputs.
        
        Args:
            batch_data: List of input dictionaries
        
        Returns:
            List of prediction result dictionaries
        """
        results = []
        start_time = time.time()
        
        for idx, input_data in enumerate(batch_data):
            try:
                result = self.predict(input_data)
                result['batch_index'] = idx
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch prediction error at index {idx}: {e}")
                results.append({
                    'batch_index': idx,
                    'error': str(e),
                    'success': False
                })
        
        total_time_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"Batch prediction completed: {len(batch_data)} samples in {total_time_ms:.2f}ms "
            f"({total_time_ms/len(batch_data):.2f}ms per sample)"
        )
        
        return results
    
    def get_feature_importance(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            List of dictionaries with feature names and importance scores
        """
        try:
            model = self.model_service.get_model()
            feature_names = self.model_service.get_feature_names()
            
            # Get feature importances (works for Random Forest and XGBoost)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                self.logger.warning("Model does not have feature_importances_ attribute")
                return []
            
            # Create list of (feature, importance) tuples
            feature_importance = [
                {
                    'feature': name,
                    'importance': float(imp),
                    'importance_percent': round(float(imp) * 100, 2)
                }
                for name, imp in zip(feature_names, importances)
            ]
            
            # Sort by importance and return top N
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            return feature_importance[:top_n]
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return []
