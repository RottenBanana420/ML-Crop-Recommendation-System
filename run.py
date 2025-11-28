"""
Application Entry Point for Development

Run this file to start the Flask development server.
"""

import os
from app import create_app
from app.services.model_service import get_model_service


# Create Flask app
app = create_app(os.getenv('FLASK_ENV', 'development'))

# Load models on startup
with app.app_context():
    model_service = get_model_service()
    if not model_service.load_models(app.config):
        app.logger.error("Failed to load models. Application may not function correctly.")
    else:
        app.logger.info("Models loaded successfully")


if __name__ == '__main__':
    # Run development server
    app.run(
        host='0.0.0.0',
        port=5001,  
        debug=app.config['DEBUG']
    )
