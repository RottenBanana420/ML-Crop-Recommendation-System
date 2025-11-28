"""
Prediction Routes Blueprint

Handles prediction form page and API endpoints for crop recommendations.
"""

from flask import Blueprint, render_template, request, jsonify, current_app

from app.services.validation_service import ValidationService
from app.services.prediction_service import PredictionService


prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/')
def predict_page():
    """Prediction form page"""
    validation_service = ValidationService(current_app.config)
    field_info = validation_service.get_all_field_info()
    
    return render_template('pages/predict.html', fields=field_info)


@prediction_bp.route('/api', methods=['POST'])
def predict_api():
    """
    API endpoint for crop predictions.
    
    Request JSON:
        {
            "N": float,
            "P": float,
            "K": float,
            "temperature": float,
            "humidity": float,
            "ph": float,
            "rainfall": float
        }
    
    Response JSON:
        {
            "success": bool,
            "result": {
                "crop": str,
                "confidence": float,
                "confidence_percent": float,
                "alternatives": [{"crop": str, "confidence": float}, ...],
                "prediction_time_ms": float
            }
        }
    """
    # Apply rate limiting if enabled
    if current_app.limiter:
        # This will be handled by the decorator in production
        pass
    
    # Get JSON data
    try:
        data = request.get_json()
        if data is None:
            return jsonify({
                'success': False,
                'error': 'Invalid JSON',
                'message': 'Request must contain valid JSON data'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Invalid JSON',
            'message': str(e)
        }), 400
    
    # Validate input
    validation_service = ValidationService(current_app.config)
    errors = validation_service.validate_input(data)
    
    if errors:
        return jsonify({
            'success': False,
            'error': 'Validation Error',
            'errors': errors,
            'message': 'Please check your input values'
        }), 400
    
    # Sanitize input
    try:
        sanitized_data = validation_service.sanitize_input(data)
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid Input',
            'message': str(e)
        }), 400
    
    # Make prediction
    try:
        prediction_service = PredictionService(current_app.config)
        result = prediction_service.predict(sanitized_data)
        
        return jsonify({
            'success': True,
            'result': result
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Prediction Error',
            'message': 'An error occurred while generating prediction'
        }), 500


@prediction_bp.route('/batch', methods=['POST'])
def predict_batch():
    """
    API endpoint for batch predictions.
    
    Request JSON:
        {
            "inputs": [
                {"N": float, "P": float, ...},
                {"N": float, "P": float, ...},
                ...
            ]
        }
    
    Response JSON:
        {
            "success": bool,
            "results": [prediction_result, ...],
            "summary": {
                "total": int,
                "successful": int,
                "failed": int
            }
        }
    """
    # Get JSON data
    try:
        data = request.get_json()
        if data is None or 'inputs' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid JSON',
                'message': 'Request must contain "inputs" array'
            }), 400
        
        inputs = data['inputs']
        if not isinstance(inputs, list):
            return jsonify({
                'success': False,
                'error': 'Invalid Input',
                'message': '"inputs" must be an array'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Invalid JSON',
            'message': str(e)
        }), 400
    
    # Validate batch
    validation_service = ValidationService(current_app.config)
    validation_results = validation_service.validate_batch(inputs)
    
    if not validation_results['valid']:
        return jsonify({
            'success': False,
            'error': 'Validation Error',
            'message': f"{validation_results['invalid_count']} inputs failed validation",
            'errors': validation_results['errors']
        }), 400
    
    # Make batch predictions
    try:
        prediction_service = PredictionService(current_app.config)
        
        # Sanitize all inputs
        sanitized_inputs = [validation_service.sanitize_input(inp) for inp in inputs]
        
        # Generate predictions
        results = prediction_service.predict_batch(sanitized_inputs)
        
        # Count successes and failures
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total': len(results),
                'successful': successful,
                'failed': failed
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Prediction Error',
            'message': 'An error occurred during batch prediction'
        }), 500


# Apply rate limiting to prediction endpoints if enabled
def init_rate_limiting(app):
    """Initialize rate limiting for prediction endpoints"""
    if app.limiter:
        limiter = app.limiter
        limiter.limit(app.config['RATELIMIT_PREDICTION'])(predict_api)
        limiter.limit(app.config['RATELIMIT_PREDICTION'])(predict_batch)
