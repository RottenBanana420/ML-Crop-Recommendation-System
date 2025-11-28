"""
Validation Service - Input Data Validation

This module provides comprehensive validation for user input data,
ensuring data quality and security before processing.
"""

import logging
from typing import Dict, Any, Optional, List


class ValidationService:
    """Service for validating crop recommendation input data"""
    
    def __init__(self, config):
        """
        Initialize validation service with configuration.
        
        Args:
            config: Flask app configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_ranges = config['VALIDATION_RANGES']
        self.required_fields = config['INPUT_FEATURES']
    
    def validate_input(self, data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Validate input data for crop prediction.
        
        Args:
            data: Dictionary containing input features
        
        Returns:
            Dictionary of errors (field -> error message) if validation fails,
            None if validation passes
        """
        errors = {}
        
        # Check for required fields
        for field in self.required_fields:
            if field not in data:
                errors[field] = f"{field} is required"
                continue
            
            # Validate field value
            field_errors = self._validate_field(field, data[field])
            if field_errors:
                errors[field] = field_errors
        
        return errors if errors else None
    
    def _validate_field(self, field_name: str, value: Any) -> Optional[str]:
        """
        Validate a single field value.
        
        Args:
            field_name: Name of the field
            value: Value to validate
        
        Returns:
            Error message if validation fails, None otherwise
        """
        # Check if value is numeric
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return f"{field_name} must be a number"
        
        # Check for NaN or infinity
        if not self._is_finite(numeric_value):
            return f"{field_name} must be a finite number"
        
        # Check range
        if field_name in self.validation_ranges:
            range_info = self.validation_ranges[field_name]
            min_val = range_info['min']
            max_val = range_info['max']
            unit = range_info.get('unit', '')
            
            if numeric_value < min_val or numeric_value > max_val:
                return f"{field_name} must be between {min_val} and {max_val} {unit}"
        
        return None
    
    def _is_finite(self, value: float) -> bool:
        """Check if value is finite (not NaN or infinity)"""
        import math
        return not (math.isnan(value) or math.isinf(value))
    
    def sanitize_input(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Sanitize and convert input data to proper types.
        
        Args:
            data: Raw input data
        
        Returns:
            Sanitized data with float values
        
        Raises:
            ValueError: If data cannot be sanitized
        """
        sanitized = {}
        
        for field in self.required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            
            try:
                sanitized[field] = float(data[field])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {field}: {data[field]}")
        
        return sanitized
    
    def validate_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of input data.
        
        Args:
            batch_data: List of input dictionaries
        
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List of error dictionaries (one per input),
                'valid_count': int,
                'invalid_count': int
            }
        """
        results = {
            'valid': True,
            'errors': [],
            'valid_count': 0,
            'invalid_count': 0
        }
        
        for idx, data in enumerate(batch_data):
            errors = self.validate_input(data)
            if errors:
                results['valid'] = False
                results['invalid_count'] += 1
                results['errors'].append({
                    'index': idx,
                    'errors': errors
                })
            else:
                results['valid_count'] += 1
                results['errors'].append(None)
        
        return results
    
    def get_field_info(self, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Get validation information for a specific field.
        
        Args:
            field_name: Name of the field
        
        Returns:
            Dictionary with field validation info, or None if field not found
        """
        if field_name not in self.validation_ranges:
            return None
        
        range_info = self.validation_ranges[field_name]
        return {
            'name': field_name,
            'min': range_info['min'],
            'max': range_info['max'],
            'unit': range_info.get('unit', ''),
            'required': field_name in self.required_fields
        }
    
    def get_all_field_info(self) -> List[Dict[str, Any]]:
        """
        Get validation information for all fields.
        
        Returns:
            List of field validation info dictionaries
        """
        return [self.get_field_info(field) for field in self.required_fields]
