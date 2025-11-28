"""
Model Service - Singleton for ML Model Management

This module provides thread-safe singleton access to the trained ML models,
scaler, and label encoder. Models are loaded once on startup and cached.
"""

import json
import joblib
import logging
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, Any

import numpy as np


class ModelService:
    """
    Singleton service for managing ML models and preprocessing components.
    
    Thread-safe implementation ensures models are loaded once and shared
    across all requests for optimal performance.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the model service (only runs once)"""
        if self._initialized:
            return
        
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.feature_names = None
        self.model_comparison = None
        self._initialized = True
        
        self.logger.info("ModelService initialized")
    
    def load_models(self, config) -> bool:
        """
        Load all model components from disk.
        
        Args:
            config: Flask app configuration object
        
        Returns:
            True if all models loaded successfully, False otherwise
        """
        try:
            self.logger.info("Loading ML models and preprocessing components...")
            
            # Load production model
            model_path = config['PRODUCTION_MODEL_PATH']
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Production model not found: {model_path}")
            self.model = joblib.load(model_path)
            self.logger.info(f"Loaded production model from {model_path}")
            
            # Load scaler
            scaler_path = config['SCALER_PATH']
            if not Path(scaler_path).exists():
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Loaded scaler from {scaler_path}")
            
            # Load label encoder
            encoder_path = config['LABEL_ENCODER_PATH']
            if not Path(encoder_path).exists():
                raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
            self.label_encoder = joblib.load(encoder_path)
            self.logger.info(f"Loaded label encoder from {encoder_path}")
            
            # Load metadata
            metadata_path = config['METADATA_PATH']
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.logger.info(f"Loaded model metadata from {metadata_path}")
            else:
                self.logger.warning(f"Metadata file not found: {metadata_path}")
                self.metadata = {}
            
            # Load feature names
            feature_names_path = config['FEATURE_NAMES_PATH']
            if Path(feature_names_path).exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                self.logger.info(f"Loaded feature names: {len(self.feature_names)} features")
            else:
                # Fallback to config
                self.feature_names = config['FEATURE_NAMES']
                self.logger.warning(f"Using feature names from config")
            
            # Load model comparison data
            comparison_path = config['MODEL_COMPARISON_PATH']
            if Path(comparison_path).exists():
                with open(comparison_path, 'r') as f:
                    self.model_comparison = json.load(f)
                self.logger.info(f"Loaded model comparison data")
            else:
                self.logger.warning(f"Model comparison file not found: {comparison_path}")
                self.model_comparison = {}
            
            # Verify model is ready
            self._verify_model()
            
            self.logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}", exc_info=True)
            return False
    
    def _verify_model(self):
        """Verify that loaded model is functional"""
        try:
            # Create dummy input with correct number of features
            n_features = len(self.feature_names)
            dummy_input = np.zeros((1, n_features))
            
            # Test scaling
            scaled = self.scaler.transform(dummy_input)
            
            # Test prediction
            prediction = self.model.predict(scaled)
            probabilities = self.model.predict_proba(scaled)
            
            # Verify shapes
            assert prediction.shape == (1,), "Prediction shape mismatch"
            assert probabilities.shape[0] == 1, "Probability shape mismatch"
            assert probabilities.shape[1] == len(self.label_encoder.classes_), "Class count mismatch"
            
            self.logger.info("Model verification successful")
            
        except Exception as e:
            raise RuntimeError(f"Model verification failed: {e}")
    
    def get_model(self):
        """Get the loaded model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_models() first.")
        return self.model
    
    def get_scaler(self):
        """Get the loaded scaler"""
        if self.scaler is None:
            raise RuntimeError("Scaler not loaded. Call load_models() first.")
        return self.scaler
    
    def get_label_encoder(self):
        """Get the loaded label encoder"""
        if self.label_encoder is None:
            raise RuntimeError("Label encoder not loaded. Call load_models() first.")
        return self.label_encoder
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata"""
        return self.metadata or {}
    
    def get_feature_names(self) -> list:
        """Get list of feature names"""
        return self.feature_names or []
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get model comparison data"""
        return self.model_comparison or {}
    
    def get_crop_classes(self) -> list:
        """Get list of all crop classes"""
        if self.label_encoder is None:
            return []
        return self.label_encoder.classes_.tolist()
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return all([
            self.model is not None,
            self.scaler is not None,
            self.label_encoder is not None
        ])


# Global singleton instance
_model_service = ModelService()


def get_model_service() -> ModelService:
    """
    Get the global ModelService instance.
    
    Returns:
        ModelService singleton instance
    """
    return _model_service
