"""
Input Validation Tests

These tests verify that the validation service correctly validates input data
and rejects invalid values.

CRITICAL: These tests are designed to FAIL if validation doesn't work properly.
"""

import pytest
from app import create_app
from app.services.validation_service import ValidationService


@pytest.fixture
def app():
    """Create test app"""
    return create_app('testing')


@pytest.fixture
def validation_service(app):
    """Create validation service"""
    return ValidationService(app.config)


class TestValidationService:
    """Test input validation"""
    
    def test_valid_input_passes(self, validation_service):
        """Test that valid input passes validation"""
        valid_data = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.8,
            'humidity': 82.0,
            'ph': 6.5,
            'rainfall': 202.9
        }
        errors = validation_service.validate_input(valid_data)
        assert errors is None, f"Valid input should not have errors: {errors}"
    
    def test_out_of_range_nitrogen_rejected(self, validation_service):
        """Test that N value outside range is rejected"""
        # Test below minimum
        data = {'N': -10, 'P': 42, 'K': 43, 'temperature': 20, 'humidity': 80, 'ph': 6.5, 'rainfall': 200}
        errors = validation_service.validate_input(data)
        assert errors is not None, "Negative N should be rejected"
        assert 'N' in errors, "Error should mention N field"
        
        # Test above maximum
        data['N'] = 1000
        errors = validation_service.validate_input(data)
        assert errors is not None, "N=1000 should be rejected (max is 140)"
        assert 'N' in errors, "Error should mention N field"
    
    def test_out_of_range_temperature_rejected(self, validation_service):
        """Test that temperature outside range is rejected"""
        data = {'N': 90, 'P': 42, 'K': 43, 'temperature': -50, 'humidity': 80, 'ph': 6.5, 'rainfall': 200}
        errors = validation_service.validate_input(data)
        assert errors is not None, "Temperature=-50 should be rejected"
        assert 'temperature' in errors, "Error should mention temperature field"
    
    def test_non_numeric_value_rejected(self, validation_service):
        """Test that non-numeric values are rejected"""
        data = {'N': 'abc', 'P': 42, 'K': 43, 'temperature': 20, 'humidity': 80, 'ph': 6.5, 'rainfall': 200}
        errors = validation_service.validate_input(data)
        assert errors is not None, "Non-numeric N should be rejected"
        assert 'N' in errors, "Error should mention N field"
    
    def test_missing_required_field_rejected(self, validation_service):
        """Test that missing required fields are rejected"""
        data = {'P': 42, 'K': 43, 'temperature': 20, 'humidity': 80, 'ph': 6.5, 'rainfall': 200}
        # Missing N
        errors = validation_service.validate_input(data)
        assert errors is not None, "Missing N field should be rejected"
        assert 'N' in errors, "Error should mention missing N field"
    
    def test_edge_case_minimum_values(self, validation_service):
        """Test that minimum valid values are accepted"""
        data = {
            'N': 0,  # Min is 0
            'P': 5,  # Min is 5
            'K': 5,  # Min is 5
            'temperature': 8.8,  # Min is 8.8
            'humidity': 14.3,  # Min is 14.3
            'ph': 3.5,  # Min is 3.5
            'rainfall': 20.2  # Min is 20.2
        }
        errors = validation_service.validate_input(data)
        assert errors is None, f"Minimum valid values should pass: {errors}"
    
    def test_edge_case_maximum_values(self, validation_service):
        """Test that maximum valid values are accepted"""
        data = {
            'N': 140,  # Max is 140
            'P': 145,  # Max is 145
            'K': 205,  # Max is 205
            'temperature': 43.7,  # Max is 43.7
            'humidity': 99.9,  # Max is 99.9
            'ph': 9.9,  # Max is 9.9
            'rainfall': 298.6  # Max is 298.6
        }
        errors = validation_service.validate_input(data)
        assert errors is None, f"Maximum valid values should pass: {errors}"
    
    def test_validation_error_messages_descriptive(self, validation_service):
        """Test that validation error messages are descriptive"""
        data = {'N': -10, 'P': 42, 'K': 43, 'temperature': 20, 'humidity': 80, 'ph': 6.5, 'rainfall': 200}
        errors = validation_service.validate_input(data)
        assert errors is not None
        assert 'N' in errors
        error_msg = errors['N']
        # Error message should mention the valid range
        assert 'between' in error_msg.lower() or 'range' in error_msg.lower(), \
            f"Error message should mention valid range: {error_msg}"
    
    def test_multiple_errors_reported(self, validation_service):
        """Test that multiple validation errors are all reported"""
        data = {
            'N': -10,  # Invalid
            'P': 1000,  # Invalid
            'K': 43,
            'temperature': 20,
            'humidity': 80,
            'ph': 6.5,
            'rainfall': 200
        }
        errors = validation_service.validate_input(data)
        assert errors is not None
        assert 'N' in errors, "Should report N error"
        assert 'P' in errors, "Should report P error"
    
    def test_sanitize_input_converts_types(self, validation_service):
        """Test that sanitize_input converts values to float"""
        data = {
            'N': '90',  # String
            'P': 42,
            'K': 43,
            'temperature': '20.8',  # String
            'humidity': 82.0,
            'ph': 6.5,
            'rainfall': 202.9
        }
        sanitized = validation_service.sanitize_input(data)
        assert isinstance(sanitized['N'], float), "N should be converted to float"
        assert isinstance(sanitized['temperature'], float), "temperature should be converted to float"
        assert sanitized['N'] == 90.0
