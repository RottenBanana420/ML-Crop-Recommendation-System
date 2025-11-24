"""
Comprehensive test suite for CropDataLoader.

These tests are designed to FAIL if there are any issues with data loading.
Tests validate data integrity, column presence, data types, missing values,
duplicate rows, and value ranges.

CRITICAL: If any test fails, modify the CODEBASE (loader.py), NOT the tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import CropDataLoader


class TestDataLoading:
    """Tests for basic data loading functionality."""
    
    def test_data_file_exists(self, data_path):
        """Test that the data file exists and is readable."""
        assert data_path.exists(), f"Data file not found at {data_path}"
        assert data_path.is_file(), f"Path {data_path} is not a file"
        assert data_path.suffix == '.csv', f"File must be CSV, got {data_path.suffix}"
    
    def test_loader_initialization_default_path(self):
        """Test that loader can be initialized with default path."""
        loader = CropDataLoader()
        assert loader.data_path.exists(), "Default data path does not exist"
    
    def test_loader_initialization_custom_path(self, data_path):
        """Test that loader can be initialized with custom path."""
        loader = CropDataLoader(str(data_path))
        assert loader.data_path == data_path
    
    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a pandas DataFrame."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert isinstance(df, pd.DataFrame), "load_data must return a pandas DataFrame"
    
    def test_load_data_without_validation(self):
        """Test that data can be loaded without validation."""
        loader = CropDataLoader()
        df = loader.load_data(validate=False)
        assert df is not None
        assert len(df) > 0
    
    def test_data_not_empty(self):
        """Test that the dataset is not empty."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert len(df) > 0, "Dataset must not be empty"
        assert df.shape[0] > 0, "Dataset must have at least one row"
        assert df.shape[1] > 0, "Dataset must have at least one column"
    
    def test_minimum_rows_requirement(self):
        """Test that dataset has minimum required rows."""
        loader = CropDataLoader()
        df = loader.load_data()
        min_rows = 100
        assert len(df) >= min_rows, f"Dataset must have at least {min_rows} rows, got {len(df)}"


class TestColumnValidation:
    """Tests for column presence and structure."""
    
    def test_all_required_columns_present(self, expected_columns):
        """Test that all required columns are present."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        for col in expected_columns:
            assert col in df.columns, f"Required column '{col}' is missing"
    
    def test_no_extra_columns(self, expected_columns):
        """Test that there are no unexpected extra columns."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        extra_cols = set(df.columns) - set(expected_columns)
        assert len(extra_cols) == 0, f"Unexpected extra columns found: {extra_cols}"
    
    def test_exact_column_match(self, expected_columns):
        """Test that columns exactly match expected columns."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        assert set(df.columns) == set(expected_columns), \
            f"Column mismatch. Expected: {expected_columns}, Got: {list(df.columns)}"
    
    def test_column_order(self, expected_columns):
        """Test that columns are in the expected order."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        assert list(df.columns) == expected_columns, \
            f"Column order mismatch. Expected: {expected_columns}, Got: {list(df.columns)}"
    
    def test_get_feature_names(self):
        """Test get_feature_names method."""
        loader = CropDataLoader()
        loader.load_data()
        
        features = loader.get_feature_names()
        expected_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        assert features == expected_features, \
            f"Feature names mismatch. Expected: {expected_features}, Got: {features}"
    
    def test_get_target_name(self):
        """Test get_target_name method."""
        loader = CropDataLoader()
        loader.load_data()
        
        target = loader.get_target_name()
        assert target == 'label', f"Target name must be 'label', got '{target}'"


class TestDataTypes:
    """Tests for data type validation."""
    
    def test_n_is_numeric(self):
        """Test that N column is numeric."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert pd.api.types.is_numeric_dtype(df['N']), "Column 'N' must be numeric"
    
    def test_p_is_numeric(self):
        """Test that P column is numeric."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert pd.api.types.is_numeric_dtype(df['P']), "Column 'P' must be numeric"
    
    def test_k_is_numeric(self):
        """Test that K column is numeric."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert pd.api.types.is_numeric_dtype(df['K']), "Column 'K' must be numeric"
    
    def test_temperature_is_numeric(self):
        """Test that temperature column is numeric."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert pd.api.types.is_numeric_dtype(df['temperature']), \
            "Column 'temperature' must be numeric"
    
    def test_humidity_is_numeric(self):
        """Test that humidity column is numeric."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert pd.api.types.is_numeric_dtype(df['humidity']), \
            "Column 'humidity' must be numeric"
    
    def test_ph_is_numeric(self):
        """Test that ph column is numeric."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert pd.api.types.is_numeric_dtype(df['ph']), "Column 'ph' must be numeric"
    
    def test_rainfall_is_numeric(self):
        """Test that rainfall column is numeric."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert pd.api.types.is_numeric_dtype(df['rainfall']), \
            "Column 'rainfall' must be numeric"
    
    def test_label_is_string(self):
        """Test that label column is string/object type."""
        loader = CropDataLoader()
        df = loader.load_data()
        assert pd.api.types.is_object_dtype(df['label']) or \
               pd.api.types.is_string_dtype(df['label']), \
            "Column 'label' must be string/object type"
    
    def test_no_mixed_types_in_numeric_columns(self):
        """Test that numeric columns don't have mixed types."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for col in numeric_cols:
            # All values should be numeric
            assert df[col].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating))).all(), \
                f"Column '{col}' contains non-numeric values"
    
    def test_get_data_types_method(self):
        """Test get_data_types method."""
        loader = CropDataLoader()
        loader.load_data()
        
        dtypes = loader.get_data_types()
        assert isinstance(dtypes, dict), "get_data_types must return a dictionary"
        assert len(dtypes) == 8, "Must return data types for all 8 columns"


class TestMissingValues:
    """Tests for missing value detection."""
    
    def test_no_missing_values_in_any_column(self):
        """Test that there are no missing values in any column."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        missing = df.isnull().sum()
        assert missing.sum() == 0, f"Found missing values: {missing[missing > 0].to_dict()}"
    
    def test_no_nan_values(self):
        """Test that there are no NaN values."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        for col in df.columns:
            nan_count = df[col].isna().sum()
            assert nan_count == 0, f"Column '{col}' has {nan_count} NaN values"
    
    def test_no_none_values(self):
        """Test that there are no None values."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        for col in df.columns:
            none_count = df[col].isnull().sum()
            assert none_count == 0, f"Column '{col}' has {none_count} None values"
    
    def test_no_empty_strings_in_label(self):
        """Test that label column has no empty strings."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        empty_count = (df['label'] == '').sum()
        assert empty_count == 0, f"Found {empty_count} empty strings in label column"
    
    def test_check_missing_values_method(self):
        """Test check_missing_values method."""
        loader = CropDataLoader()
        loader.load_data()
        
        missing = loader.check_missing_values()
        assert isinstance(missing, dict), "check_missing_values must return a dictionary"
        assert all(count == 0 for count in missing.values()), \
            f"Found missing values: {missing}"


class TestDuplicates:
    """Tests for duplicate row detection."""
    
    def test_no_duplicate_rows(self):
        """Test that there are no duplicate rows."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        duplicates = df.duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate rows in the dataset"
    
    def test_check_duplicates_method(self):
        """Test check_duplicates method."""
        loader = CropDataLoader()
        loader.load_data()
        
        duplicates = loader.check_duplicates()
        assert duplicates == 0, f"Found {duplicates} duplicate rows"
    
    def test_data_integrity(self):
        """Test overall data integrity."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        # No duplicates
        assert df.duplicated().sum() == 0, "Data must not contain duplicates"
        
        # No missing values
        assert df.isnull().sum().sum() == 0, "Data must not contain missing values"


class TestValueRanges:
    """Tests for value range validation."""
    
    def test_n_values_non_negative(self):
        """Test that N values are non-negative."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        assert (df['N'] >= 0).all(), f"N values must be non-negative. Min: {df['N'].min()}"
    
    def test_p_values_non_negative(self):
        """Test that P values are non-negative."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        assert (df['P'] >= 0).all(), f"P values must be non-negative. Min: {df['P'].min()}"
    
    def test_k_values_non_negative(self):
        """Test that K values are non-negative."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        assert (df['K'] >= 0).all(), f"K values must be non-negative. Min: {df['K'].min()}"
    
    def test_temperature_values_reasonable(self):
        """Test that temperature values are in reasonable range (0-50°C)."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        assert (df['temperature'] >= 0).all(), \
            f"Temperature must be >= 0°C. Min: {df['temperature'].min()}"
        assert (df['temperature'] <= 50).all(), \
            f"Temperature must be <= 50°C. Max: {df['temperature'].max()}"
    
    def test_humidity_values_percentage(self):
        """Test that humidity values are valid percentages (0-100)."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        assert (df['humidity'] >= 0).all(), \
            f"Humidity must be >= 0%. Min: {df['humidity'].min()}"
        assert (df['humidity'] <= 100).all(), \
            f"Humidity must be <= 100%. Max: {df['humidity'].max()}"
    
    def test_ph_values_valid(self):
        """Test that pH values are in valid range (0-14)."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        assert (df['ph'] >= 0).all(), f"pH must be >= 0. Min: {df['ph'].min()}"
        assert (df['ph'] <= 14).all(), f"pH must be <= 14. Max: {df['ph'].max()}"
    
    def test_rainfall_values_non_negative(self):
        """Test that rainfall values are non-negative."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        assert (df['rainfall'] >= 0).all(), \
            f"Rainfall must be non-negative. Min: {df['rainfall'].min()}"
    
    def test_no_infinite_values(self):
        """Test that there are no infinite values in numeric columns."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for col in numeric_cols:
            assert not (df[col] == float('inf')).any(), \
                f"Column '{col}' contains positive infinity"
            assert not (df[col] == float('-inf')).any(), \
                f"Column '{col}' contains negative infinity"
    
    def test_get_value_ranges_method(self):
        """Test get_value_ranges method."""
        loader = CropDataLoader()
        loader.load_data()
        
        ranges = loader.get_value_ranges()
        assert isinstance(ranges, dict), "get_value_ranges must return a dictionary"
        
        # Check that all ranges are valid tuples
        for col, (min_val, max_val) in ranges.items():
            assert isinstance(min_val, (int, float)), f"Min value for '{col}' must be numeric"
            assert isinstance(max_val, (int, float)), f"Max value for '{col}' must be numeric"
            assert min_val <= max_val, f"Min must be <= max for '{col}'"


class TestLabelValidation:
    """Tests for label column validation."""
    
    def test_labels_from_expected_set(self, expected_labels):
        """Test that all labels are from the expected set."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        unique_labels = set(df['label'].unique())
        unexpected = unique_labels - expected_labels
        
        assert len(unexpected) == 0, \
            f"Found unexpected labels: {unexpected}. Expected: {expected_labels}"
    
    def test_no_empty_labels(self):
        """Test that there are no empty labels."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        empty_count = (df['label'].str.strip() == '').sum()
        assert empty_count == 0, f"Found {empty_count} empty labels"
    
    def test_label_distribution_reasonable(self):
        """Test that label distribution is reasonable (no class has < 10 samples)."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        distribution = df['label'].value_counts()
        min_samples = 10
        
        for label, count in distribution.items():
            assert count >= min_samples, \
                f"Label '{label}' has only {count} samples (minimum: {min_samples})"
    
    def test_get_class_distribution_method(self):
        """Test get_class_distribution method."""
        loader = CropDataLoader()
        loader.load_data()
        
        distribution = loader.get_class_distribution()
        assert isinstance(distribution, dict), "get_class_distribution must return a dictionary"
        assert len(distribution) > 0, "Distribution must not be empty"
        assert all(count > 0 for count in distribution.values()), \
            "All classes must have at least one sample"


class TestDatasetInfo:
    """Tests for dataset information methods."""
    
    def test_get_dataset_info_returns_dict(self):
        """Test that get_dataset_info returns a dictionary."""
        loader = CropDataLoader()
        loader.load_data()
        
        info = loader.get_dataset_info()
        assert isinstance(info, dict), "get_dataset_info must return a dictionary"
    
    def test_dataset_info_contains_required_keys(self):
        """Test that dataset info contains all required keys."""
        loader = CropDataLoader()
        loader.load_data()
        
        info = loader.get_dataset_info()
        required_keys = [
            'shape', 'num_rows', 'num_columns', 'columns', 
            'feature_columns', 'target_column', 'data_types',
            'missing_values', 'duplicate_rows', 'class_distribution'
        ]
        
        for key in required_keys:
            assert key in info, f"Dataset info must contain '{key}'"
    
    def test_dataset_shape_correct(self):
        """Test that dataset shape is reported correctly."""
        loader = CropDataLoader()
        df = loader.load_data()
        
        info = loader.get_dataset_info()
        assert info['shape'] == df.shape, "Shape mismatch in dataset info"
        assert info['num_rows'] == len(df), "Row count mismatch"
        assert info['num_columns'] == len(df.columns), "Column count mismatch"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_data_property_before_loading_raises_error(self):
        """Test that accessing data property before loading raises error."""
        loader = CropDataLoader()
        
        with pytest.raises(RuntimeError, match="Data must be loaded first"):
            _ = loader.data
    
    def test_validate_before_loading_raises_error(self):
        """Test that validating before loading raises error."""
        loader = CropDataLoader()
        
        with pytest.raises(RuntimeError, match="Data must be loaded before validation"):
            loader.validate_data()
    
    def test_get_info_before_loading_raises_error(self):
        """Test that getting info before loading raises error."""
        loader = CropDataLoader()
        
        with pytest.raises(RuntimeError, match="Data must be loaded first"):
            loader.get_dataset_info()
    
    def test_data_property_returns_copy(self):
        """Test that data property returns a copy, not reference."""
        loader = CropDataLoader()
        loader.load_data()
        
        df1 = loader.data
        df2 = loader.data
        
        # Modify df1
        df1.iloc[0, 0] = -9999
        
        # df2 should not be affected
        assert df1.iloc[0, 0] != df2.iloc[0, 0], \
            "Data property must return a copy, not a reference"
    
    def test_nonexistent_file_raises_error(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        loader = CropDataLoader('/nonexistent/path/to/file.csv')
        
        with pytest.raises(FileNotFoundError):
            loader.load_data()


class TestValidationMethod:
    """Tests for the validate_data method."""
    
    def test_validate_data_passes_on_good_data(self):
        """Test that validate_data passes on good data."""
        loader = CropDataLoader()
        df = loader.load_data(validate=True)
        
        # If we got here without exception, validation passed
        assert df is not None
    
    def test_validate_data_method_callable(self):
        """Test that validate_data method can be called explicitly."""
        loader = CropDataLoader()
        loader.load_data(validate=False)
        
        # Should not raise any exception
        loader.validate_data()
