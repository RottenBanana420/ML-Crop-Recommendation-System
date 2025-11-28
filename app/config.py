"""
Flask Application Configuration

This module provides environment-based configuration for the Flask application.
Includes settings for development, production, and testing environments.
"""

import os
from pathlib import Path


class Config:
    """Base configuration with common settings"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / 'models'
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production-immediately'
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Model paths
    PRODUCTION_MODEL_PATH = MODELS_DIR / 'production_model.pkl'
    SCALER_PATH = MODELS_DIR / 'scaler.pkl'
    LABEL_ENCODER_PATH = MODELS_DIR / 'label_encoder.pkl'
    METADATA_PATH = MODELS_DIR / 'production_model_metadata.json'
    FEATURE_NAMES_PATH = MODELS_DIR / 'feature_names.json'
    MODEL_COMPARISON_PATH = MODELS_DIR / 'model_comparison.json'
    
    # Feature names (22 engineered features from preprocessing pipeline)
    FEATURE_NAMES = [
        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
        'rainfall_log', 'N_to_P_ratio', 'N_to_K_ratio', 'P_to_K_ratio',
        'total_NPK', 'NPK_balance', 'avg_NPK',
        'temp_humidity_interaction', 'rainfall_humidity_interaction',
        'temp_rainfall_interaction', 'ph_N_interaction', 'ph_P_interaction',
        'ph_K_interaction', 'moisture_index', 'growing_conditions_index'
    ]
    
    # Original input features (7 base features)
    INPUT_FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Input validation ranges (from EDA analysis)
    VALIDATION_RANGES = {
        'N': {'min': 0, 'max': 140, 'unit': 'kg/ha'},
        'P': {'min': 5, 'max': 145, 'unit': 'kg/ha'},
        'K': {'min': 5, 'max': 205, 'unit': 'kg/ha'},
        'temperature': {'min': 8.8, 'max': 43.7, 'unit': '°C'},
        'humidity': {'min': 14.3, 'max': 99.9, 'unit': '%'},
        'ph': {'min': 3.5, 'max': 9.9, 'unit': 'pH'},
        'rainfall': {'min': 20.2, 'max': 298.6, 'unit': 'mm'}
    }
    
    # Rate limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_DEFAULT = '100 per hour'
    RATELIMIT_PREDICTION = '10 per minute'
    
    # CORS settings
    CORS_ORIGINS = ['http://localhost:5000', 'http://127.0.0.1:5000']
    
    # Performance settings
    MAX_PREDICTION_TIME_MS = 100  # Maximum acceptable prediction time
    MAX_CONCURRENT_REQUESTS = 50  # Maximum concurrent requests to handle
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False  # Allow HTTP in development
    RATELIMIT_ENABLED = False  # Disable rate limiting in development
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    # In production, SECRET_KEY MUST be set via environment variable
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Production CORS origins should be configured via environment
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',') if os.environ.get('CORS_ORIGINS') else []
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    SESSION_COOKIE_SECURE = False
    RATELIMIT_ENABLED = False  # Disable rate limiting in tests
    LOG_LEVEL = 'DEBUG'
    
    # Use test-specific paths if needed
    WTF_CSRF_ENABLED = False  # Disable CSRF for testing


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """
    Get configuration object for specified environment.
    
    Args:
        env: Environment name ('development', 'production', 'testing')
             If None, uses FLASK_ENV environment variable or 'default'
    
    Returns:
        Configuration class for the specified environment
    """
    if env is None:
        env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])
