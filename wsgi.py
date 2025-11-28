"""
WSGI Entry Point for Production

Use this file with Gunicorn or other WSGI servers.

Example:
    gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
"""

import os
from app import create_app
from app.services.model_service import get_model_service


# Create Flask app for production
app = create_app(os.getenv('FLASK_ENV', 'production'))

# Load models on startup
with app.app_context():
    model_service = get_model_service()
    if not model_service.load_models(app.config):
        app.logger.error("Failed to load models")
        raise RuntimeError("Model loading failed")
    app.logger.info("Models loaded successfully for production")
