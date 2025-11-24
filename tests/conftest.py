"""
Pytest configuration and fixtures for crop recommendation tests.
"""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_path(project_root):
    """Get the path to the test dataset."""
    return project_root / 'data' / 'raw' / 'Crop_recommendation.csv'


@pytest.fixture
def expected_columns():
    """Get the expected column names."""
    return ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']


@pytest.fixture
def expected_labels():
    """Get the expected crop labels."""
    return {
        'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 
        'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 
        'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 
        'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
    }
