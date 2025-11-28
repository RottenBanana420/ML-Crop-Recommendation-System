"""
Flask Application Factory

This module implements the application factory pattern for creating
Flask application instances with proper configuration and extensions.
"""

import logging
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from app.config import get_config


def create_app(config_name=None):
    """
    Application factory for creating Flask app instances.
    
    Args:
        config_name: Configuration environment ('development', 'production', 'testing')
                    If None, uses FLASK_ENV environment variable
    
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    config_class = get_config(config_name)
    app.config.from_object(config_class)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, app.config['LOG_LEVEL']),
        format=app.config['LOG_FORMAT']
    )
    app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))
    
    # Initialize extensions
    init_extensions(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Log startup
    app.logger.info(f"Flask app created with config: {config_name or 'default'}")
    app.logger.info(f"Debug mode: {app.config['DEBUG']}")
    
    return app


def init_extensions(app):
    """Initialize Flask extensions"""
    
    # CORS - Cross-Origin Resource Sharing
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Rate Limiting
    if app.config['RATELIMIT_ENABLED']:
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            storage_uri=app.config['RATELIMIT_STORAGE_URL'],
            default_limits=[app.config['RATELIMIT_DEFAULT']]
        )
        app.limiter = limiter
        app.logger.info("Rate limiting enabled")
    else:
        app.limiter = None
        app.logger.info("Rate limiting disabled")


def register_blueprints(app):
    """Register application blueprints"""
    
    from app.routes.main import main_bp
    from app.routes.prediction import prediction_bp
    from app.routes.performance import performance_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp, url_prefix='/predict')
    app.register_blueprint(performance_bp, url_prefix='/performance')
    
    app.logger.info("Blueprints registered: main, prediction, performance")


def register_error_handlers(app):
    """Register custom error handlers"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors"""
        if hasattr(error, 'description'):
            message = error.description
        else:
            message = "Bad request. Please check your input and try again."
        
        # Return JSON for API requests, HTML for web requests
        if is_api_request():
            return jsonify({
                'success': False,
                'error': 'Bad Request',
                'message': message
            }), 400
        return render_template('errors/400.html', error=message), 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors"""
        if is_api_request():
            return jsonify({
                'success': False,
                'error': 'Not Found',
                'message': 'The requested resource was not found.'
            }), 404
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(429)
    def ratelimit_exceeded(error):
        """Handle 429 Too Many Requests errors"""
        if is_api_request():
            return jsonify({
                'success': False,
                'error': 'Rate Limit Exceeded',
                'message': 'Too many requests. Please try again later.'
            }), 429
        return render_template('errors/429.html'), 429
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error"""
        # Log the error but don't expose details to user
        app.logger.error(f"Internal server error: {error}", exc_info=True)
        
        if is_api_request():
            return jsonify({
                'success': False,
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred. Please try again later.'
            }), 500
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle uncaught exceptions"""
        # Log the full exception
        app.logger.error(f"Unhandled exception: {error}", exc_info=True)
        
        # Return generic error to user (don't expose stack trace)
        if is_api_request():
            return jsonify({
                'success': False,
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred. Please try again later.'
            }), 500
        return render_template('errors/500.html'), 500


def is_api_request():
    """
    Determine if the current request is an API request.
    
    Returns:
        True if request is for API endpoint, False otherwise
    """
    from flask import request
    return request.path.startswith('/predict/api') or \
           request.path.startswith('/performance/api') or \
           request.accept_mimetypes.accept_json and \
           not request.accept_mimetypes.accept_html
