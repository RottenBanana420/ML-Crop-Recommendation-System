"""
Performance Routes Blueprint

Handles model performance dashboard and metrics API endpoints.
"""

from flask import Blueprint, render_template, jsonify, current_app

from app.services.model_service import get_model_service
from app.services.prediction_service import PredictionService


performance_bp = Blueprint('performance', __name__)


@performance_bp.route('/')
def performance_page():
    """Performance dashboard page"""
    model_service = get_model_service()
    metadata = model_service.get_metadata()
    comparison = model_service.get_model_comparison()
    
    context = {
        'metadata': metadata,
        'comparison': comparison
    }
    
    return render_template('pages/performance.html', **context)


@performance_bp.route('/api/metrics')
def metrics_api():
    """
    API endpoint for model performance metrics.
    
    Response JSON:
        {
            "success": bool,
            "metrics": {
                "model_type": str,
                "test_accuracy": float,
                "training_time_s": float,
                "avg_prediction_time_ms": float,
                "model_size_mb": float,
                ...
            }
        }
    """
    try:
        model_service = get_model_service()
        metadata = model_service.get_metadata()
        
        return jsonify({
            'success': True,
            'metrics': metadata
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching metrics: {e}")
        return jsonify({
            'success': False,
            'error': 'Error fetching metrics',
            'message': str(e)
        }), 500


@performance_bp.route('/api/comparison')
def comparison_api():
    """
    API endpoint for model comparison data.
    
    Response JSON:
        {
            "success": bool,
            "comparison": {
                "random_forest": {...},
                "xgboost": {...},
                "winner": str
            }
        }
    """
    try:
        model_service = get_model_service()
        comparison = model_service.get_model_comparison()
        
        return jsonify({
            'success': True,
            'comparison': comparison
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching comparison: {e}")
        return jsonify({
            'success': False,
            'error': 'Error fetching comparison data',
            'message': str(e)
        }), 500


@performance_bp.route('/api/feature-importance')
def feature_importance_api():
    """
    API endpoint for feature importance data.
    
    Response JSON:
        {
            "success": bool,
            "features": [
                {"feature": str, "importance": float, "importance_percent": float},
                ...
            ]
        }
    """
    try:
        prediction_service = PredictionService(current_app.config)
        features = prediction_service.get_feature_importance(top_n=10)
        
        return jsonify({
            'success': True,
            'features': features
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching feature importance: {e}")
        return jsonify({
            'success': False,
            'error': 'Error fetching feature importance',
            'message': str(e)
        }), 500
