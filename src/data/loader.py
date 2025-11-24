"""
Data loader module for Crop Recommendation System.

This module provides the CropDataLoader class for loading, validating,
and providing information about the crop recommendation dataset.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CropDataLoader:
    """
    A robust data loader for the crop recommendation dataset.
    
    This class handles loading the CSV data, validating its integrity,
    and providing various methods to inspect the dataset.
    """
    
    # Expected columns in the dataset
    EXPECTED_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    TARGET_COLUMN = 'label'
    
    # Expected crop labels (all 22 crops in the dataset)
    EXPECTED_LABELS = {
        'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 
        'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 
        'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 
        'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
    }
    
    # Value ranges for validation
    VALUE_RANGES = {
        'N': (0, float('inf')),
        'P': (0, float('inf')),
        'K': (0, float('inf')),
        'temperature': (0, 50),  # Celsius
        'humidity': (0, 100),     # Percentage
        'ph': (0, 14),            # pH scale
        'rainfall': (0, float('inf'))  # mm
    }
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the CropDataLoader.
        
        Args:
            data_path: Path to the CSV file. If None, uses default path.
        """
        if data_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent
            data_path = project_root / 'data' / 'raw' / 'Crop_recommendation.csv'
        
        self.data_path = Path(data_path)
        self._data: Optional[pd.DataFrame] = None
        
    def load_data(self, validate: bool = True) -> pd.DataFrame:
        """
        Load the crop recommendation dataset from CSV.
        
        Args:
            validate: Whether to validate the data after loading.
            
        Returns:
            pandas DataFrame containing the dataset.
            
        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If the data is invalid and validate=True.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {self.data_path}\n"
                f"Please ensure the CSV file exists at this location."
            )
        
        try:
            # Load CSV with proper handling of whitespace and encoding
            self._data = pd.read_csv(self.data_path)
            
            # Strip whitespace from column names
            self._data.columns = self._data.columns.str.strip()
            
            # Strip whitespace from string columns
            for col in self._data.select_dtypes(include=['object']).columns:
                self._data[col] = self._data[col].str.strip()
            
            if validate:
                self.validate_data()
                
            return self._data
            
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"The CSV file at {self.data_path} is empty.")
        except Exception as e:
            raise ValueError(f"Error loading data from {self.data_path}: {str(e)}")
    
    def validate_data(self) -> None:
        """
        Validate the loaded dataset for integrity and correctness.
        
        Raises:
            ValueError: If any validation check fails.
            RuntimeError: If data hasn't been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Data must be loaded before validation. Call load_data() first.")
        
        # Check if dataset is empty
        if len(self._data) == 0:
            raise ValueError("Dataset is empty (0 rows).")
        
        # Check minimum number of rows
        if len(self._data) < 100:
            raise ValueError(f"Dataset has only {len(self._data)} rows. Expected at least 100 rows.")
        
        # Check for correct columns
        missing_cols = set(self.EXPECTED_COLUMNS) - set(self._data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        extra_cols = set(self._data.columns) - set(self.EXPECTED_COLUMNS)
        if extra_cols:
            raise ValueError(f"Unexpected extra columns found: {extra_cols}")
        
        # Check column order
        if list(self._data.columns) != self.EXPECTED_COLUMNS:
            raise ValueError(
                f"Column order mismatch.\n"
                f"Expected: {self.EXPECTED_COLUMNS}\n"
                f"Got: {list(self._data.columns)}"
            )
        
        # Check for missing values
        missing_values = self._data.isnull().sum()
        if missing_values.any():
            cols_with_missing = missing_values[missing_values > 0].to_dict()
            raise ValueError(f"Missing values found in columns: {cols_with_missing}")
        
        # Check for duplicate rows
        duplicates = self._data.duplicated().sum()
        if duplicates > 0:
            raise ValueError(f"Found {duplicates} duplicate rows in the dataset.")
        
        # Validate data types for numerical columns
        for col in self.FEATURE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(self._data[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric, but got dtype: {self._data[col].dtype}"
                )
            
            # Check for infinite values
            if not pd.Series(self._data[col]).apply(lambda x: pd.isna(x) or pd.api.types.is_number(x)).all():
                raise ValueError(f"Column '{col}' contains non-numeric values.")
            
            if (self._data[col] == float('inf')).any() or (self._data[col] == float('-inf')).any():
                raise ValueError(f"Column '{col}' contains infinite values.")
        
        # Validate label column
        if not pd.api.types.is_object_dtype(self._data[self.TARGET_COLUMN]) and \
           not pd.api.types.is_string_dtype(self._data[self.TARGET_COLUMN]):
            raise ValueError(
                f"Target column '{self.TARGET_COLUMN}' must be string/object type, "
                f"but got: {self._data[self.TARGET_COLUMN].dtype}"
            )
        
        # Check for empty labels
        if (self._data[self.TARGET_COLUMN] == '').any():
            raise ValueError("Found empty strings in label column.")
        
        # Validate label values
        unique_labels = set(self._data[self.TARGET_COLUMN].unique())
        unexpected_labels = unique_labels - self.EXPECTED_LABELS
        if unexpected_labels:
            raise ValueError(f"Unexpected crop labels found: {unexpected_labels}")
        
        # Validate value ranges
        for col, (min_val, max_val) in self.VALUE_RANGES.items():
            col_min = self._data[col].min()
            col_max = self._data[col].max()
            
            if col_min < min_val:
                raise ValueError(
                    f"Column '{col}' has values below minimum ({min_val}). "
                    f"Found minimum: {col_min}"
                )
            
            if max_val != float('inf') and col_max > max_val:
                raise ValueError(
                    f"Column '{col}' has values above maximum ({max_val}). "
                    f"Found maximum: {col_max}"
                )
    
    def get_dataset_info(self) -> Dict:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics and information.
            
        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        return {
            'shape': self._data.shape,
            'num_rows': len(self._data),
            'num_columns': len(self._data.columns),
            'columns': list(self._data.columns),
            'feature_columns': self.FEATURE_COLUMNS,
            'target_column': self.TARGET_COLUMN,
            'data_types': self._data.dtypes.to_dict(),
            'missing_values': self._data.isnull().sum().to_dict(),
            'duplicate_rows': self._data.duplicated().sum(),
            'memory_usage_mb': self._data.memory_usage(deep=True).sum() / 1024**2,
            'numerical_summary': self._data[self.FEATURE_COLUMNS].describe().to_dict(),
            'class_distribution': self._data[self.TARGET_COLUMN].value_counts().to_dict(),
            'unique_labels': sorted(self._data[self.TARGET_COLUMN].unique().tolist())
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.FEATURE_COLUMNS.copy()
    
    def get_target_name(self) -> str:
        """Get the target column name."""
        return self.TARGET_COLUMN
    
    def get_data_types(self) -> Dict[str, str]:
        """
        Get data types of all columns.
        
        Returns:
            Dictionary mapping column names to their data types.
            
        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        return {col: str(dtype) for col, dtype in self._data.dtypes.items()}
    
    def check_missing_values(self) -> Dict[str, int]:
        """
        Check for missing values in each column.
        
        Returns:
            Dictionary mapping column names to count of missing values.
            
        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        return self._data.isnull().sum().to_dict()
    
    def check_duplicates(self) -> int:
        """
        Check for duplicate rows in the dataset.
        
        Returns:
            Number of duplicate rows.
            
        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        return self._data.duplicated().sum()
    
    def get_value_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get minimum and maximum values for numerical columns.
        
        Returns:
            Dictionary mapping column names to (min, max) tuples.
            
        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        ranges = {}
        for col in self.FEATURE_COLUMNS:
            ranges[col] = (float(self._data[col].min()), float(self._data[col].max()))
        
        return ranges
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of crop labels.
        
        Returns:
            Dictionary mapping crop labels to their counts.
            
        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        return self._data[self.TARGET_COLUMN].value_counts().to_dict()
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Get the loaded DataFrame.
        
        Returns:
            The loaded pandas DataFrame.
            
        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        return self._data.copy()
