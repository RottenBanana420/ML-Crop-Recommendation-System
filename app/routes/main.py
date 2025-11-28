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
