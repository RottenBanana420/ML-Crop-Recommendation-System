"""
Main Routes Blueprint

Handles landing page, about page, and health check endpoints.
"""

from flask import Blueprint, render_template, jsonify, current_app

from app.services.model_service import get_model_service


main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Landing page with hero section and features"""
    model_service = get_model_service()
    
    # Get model metadata for stats
    metadata = model_service.get_metadata()
    
    stats = {
        'accuracy': metadata.get('test_accuracy', 99.32),
        'crop_types': len(model_service.get_crop_classes()),
        'prediction_speed': metadata.get('avg_prediction_time_ms', 8.56),
        'tests_passed': '200+'
    }
    
    return render_template('pages/index.html', stats=stats)


@main_bp.route('/about')
def about():
    """About page with project information"""
    model_service = get_model_service()
    
    context = {
        'model_type': model_service.get_metadata().get('selected_model', 'Random Forest'),
        'accuracy': model_service.get_metadata().get('test_accuracy', 99.32),
        'crops': model_service.get_crop_classes(),
        'features_count': len(model_service.get_feature_names())
    }
    
    return render_template('pages/about.html', **context)


@main_bp.route('/health')
@main_bp.route('/api/health')
def health():
    """Health check endpoint for monitoring"""
    model_service = get_model_service()
    
    health_status = {
        'status': 'healthy' if model_service.is_loaded() else 'unhealthy',
        'models_loaded': model_service.is_loaded(),
        'crop_classes': len(model_service.get_crop_classes()),
        'features': len(model_service.get_feature_names())
    }
    
    status_code = 200 if model_service.is_loaded() else 503
    return jsonify(health_status), status_code


@main_bp.route('/api/docs')
def api_docs():
    """API documentation endpoint"""
    model_service = get_model_service()
    
    docs = {
        'title': 'ML Crop Recommendation System API',
        'version': '1.0.0',
        'description': 'RESTful API for crop recommendation based on soil and environmental conditions',
        'base_url': '/api',
        'endpoints': {
            'health': {
                'path': '/api/health',
                'method': 'GET',
                'description': 'Health check endpoint for monitoring system status',
                'response': {
                    'status': 'string (healthy|unhealthy)',
                    'models_loaded': 'boolean',
                    'crop_classes': 'integer',
                    'features': 'integer'
                }
            },
            'predict': {
                'path': '/predict/api',
                'method': 'POST',
                'description': 'Single crop prediction endpoint',
                'request_body': {
                    'N': 'float (0-140) - Nitrogen content ratio',
                    'P': 'float (5-145) - Phosphorus content ratio',
                    'K': 'float (5-205) - Potassium content ratio',
                    'temperature': 'float (8.8-43.7) - Temperature in °C',
                    'humidity': 'float (14.3-99.9) - Relative humidity %',
                    'ph': 'float (3.5-9.9) - Soil pH level',
                    'rainfall': 'float (20.2-298.6) - Annual rainfall in mm'
                },
                'response': {
                    'success': 'boolean',
                    'result': {
                        'crop': 'string - Recommended crop name',
                        'confidence': 'float - Confidence score (0-1)',
                        'confidence_percent': 'float - Confidence as percentage',
                        'alternatives': 'array - Top alternative crops with confidence',
                        'prediction_time_ms': 'float - Prediction latency'
                    }
                }
            },
            'batch_predict': {
                'path': '/predict/batch',
                'method': 'POST',
                'description': 'Batch crop prediction endpoint',
                'request_body': {
                    'inputs': 'array - Array of input objects with same structure as single prediction'
                },
                'response': {
                    'success': 'boolean',
                    'results': 'array - Array of prediction results',
                    'summary': {
                        'total': 'integer - Total predictions',
                        'successful': 'integer - Successful predictions',
                        'failed': 'integer - Failed predictions'
                    }
                }
            },
            'metrics': {
                'path': '/performance/api/metrics',
                'method': 'GET',
                'description': 'Get model performance metrics',
                'response': {
                    'success': 'boolean',
                    'metrics': 'object - Model performance metrics'
                }
            },
            'comparison': {
                'path': '/performance/api/comparison',
                'method': 'GET',
                'description': 'Get model comparison data (Random Forest vs XGBoost)',
                'response': {
                    'success': 'boolean',
                    'comparison': 'object - Model comparison data'
                }
            },
            'feature_importance': {
                'path': '/performance/api/feature-importance',
                'method': 'GET',
                'description': 'Get feature importance rankings',
                'response': {
                    'success': 'boolean',
                    'features': 'array - Feature importance data'
                }
            }
        },
        'supported_crops': model_service.get_crop_classes(),
        'features': model_service.get_feature_names(),
        'model_info': {
            'type': model_service.get_metadata().get('selected_model', 'Random Forest'),
            'accuracy': model_service.get_metadata().get('test_accuracy', 99.32),
            'version': model_service.get_metadata().get('version', '1.0.0')
        },
        'rate_limiting': {
            'enabled': current_app.config.get('RATELIMIT_ENABLED', False),
            'default_limit': current_app.config.get('RATELIMIT_DEFAULT', '100 per hour')
        },
        'example_request': {
            'url': 'POST /predict/api',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': {
                'N': 90,
                'P': 42,
                'K': 43,
                'temperature': 20.87,
                'humidity': 82.00,
                'ph': 6.50,
                'rainfall': 202.93
            }
        }
    }
    
    return jsonify(docs), 200
