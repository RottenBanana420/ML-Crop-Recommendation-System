"""
Robust Data Preprocessing Pipeline for Crop Recommendation System.

This module provides comprehensive preprocessing functionality including:
- Train-test splitting with stratification
- Feature scaling using StandardScaler
- Label encoding for categorical targets
- Agricultural domain feature engineering (nutrient ratios, environmental interactions)
- Pipeline persistence for reusability
- Comprehensive validation and error handling

Author: Crop Recommendation System
Date: 2025-11-24
"""

import os
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CropPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive preprocessing pipeline for crop recommendation data.
    
    This class handles:
    - Train-test splitting with stratification
    - Feature scaling (StandardScaler)
    - Label encoding
    - Agricultural feature engineering
    - Pipeline persistence
    
    Attributes:
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        scaling_method: Scaling method to use (default: 'standard')
        apply_log_to_rainfall: Whether to log-transform rainfall (default: True)
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        scaling_method: str = 'standard',
        apply_log_to_rainfall: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            test_size: Proportion of data for testing (0.0 to 1.0)
            random_state: Random seed for reproducibility
            scaling_method: Scaling method ('standard' or 'minmax')
            apply_log_to_rainfall: Whether to apply log transform to rainfall
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaling_method = scaling_method
        self.apply_log_to_rainfall = apply_log_to_rainfall
        
        # Fitted components (initialized during fit)
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.original_features_ = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.target_column_ = 'label'
        
        # Validation flags
        self._is_fitted = False
    
    def _validate_input_data(self, data: pd.DataFrame, require_target: bool = True) -> None:
        """
        Validate input data structure and types.
        
        Args:
            data: Input DataFrame
            require_target: Whether target column is required
            
        Raises:
            ValueError: If data validation fails
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame, got {type(data)}")
        
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Check for required features
        missing_features = set(self.original_features_) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check target column if required
        if require_target and self.target_column_ not in data.columns:
            raise ValueError(f"Target column '{self.target_column_}' not found in data")
        
        # Check data types for numerical features
        for feature in self.original_features_:
            if not pd.api.types.is_numeric_dtype(data[feature]):
                raise ValueError(f"Feature '{feature}' must be numeric, got {data[feature].dtype}")
        
        # Check for missing values
        if data[self.original_features_].isnull().any().any():
            raise ValueError("Input data contains missing values in feature columns")
    
    def _split_data(
        self, 
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets with stratification.
        
        Args:
            data: Input DataFrame with features and target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = data[self.original_features_].copy()
        y = data[self.target_column_].copy()
        
        # Check if stratification is possible (need at least 2 samples per class)
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        num_classes = len(class_counts)
        
        # Need at least 2 classes for stratification
        if num_classes < 2:
            raise ValueError(
                f"Cannot perform stratified split: only {num_classes} unique class(es) found. "
                f"Need at least 2 different classes for stratification."
            )
        
        # Use stratification only if we have enough samples per class
        if min_samples >= 2 and len(data) * self.test_size >= len(class_counts):
            stratify_param = y
        else:
            stratify_param = None
            if min_samples < 2:
                raise ValueError(
                    f"Cannot perform stratified split: class '{class_counts.idxmin()}' "
                    f"has only {min_samples} sample(s). Need at least 2 samples per class."
                )
        
        # Stratified split to maintain crop distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test
    
    def _engineer_nutrient_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer nutrient ratio features (N:P, N:K, P:K).
        
        Nutrient ratios are critical in agriculture as they indicate nutrient balance.
        
        Args:
            X: DataFrame with N, P, K columns
            
        Returns:
            DataFrame with added ratio features
        """
        X = X.copy()
        
        # Add epsilon to avoid division by zero (larger value to prevent extreme ratios)
        epsilon = 0.1
        
        # N:P ratio (capped at reasonable maximum)
        X['N_to_P_ratio'] = np.minimum(X['N'] / (X['P'] + epsilon), 100)
        
        # N:K ratio (capped at reasonable maximum)
        X['N_to_K_ratio'] = np.minimum(X['N'] / (X['K'] + epsilon), 100)
        
        # P:K ratio (capped at reasonable maximum)
        X['P_to_K_ratio'] = np.minimum(X['P'] / (X['K'] + epsilon), 100)
        
        return X
    
    def _engineer_nutrient_totals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer total nutrient and balance features.
        
        Args:
            X: DataFrame with N, P, K columns
            
        Returns:
            DataFrame with added total and balance features
        """
        X = X.copy()
        
        # Total NPK (overall nutrient availability)
        X['total_NPK'] = X['N'] + X['P'] + X['K']
        
        # NPK balance index (standard deviation indicates balance)
        X['NPK_balance'] = X[['N', 'P', 'K']].std(axis=1)
        
        # Average NPK
        X['avg_NPK'] = X[['N', 'P', 'K']].mean(axis=1)
        
        return X
    
    def _engineer_environmental_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer environmental interaction features.
        
        Environmental factors interact in complex ways affecting crop growth.
        
        Args:
            X: DataFrame with temperature, humidity, rainfall columns
            
        Returns:
            DataFrame with added interaction features
        """
        X = X.copy()
        
        # Temperature × Humidity (heat stress indicator)
        X['temp_humidity_interaction'] = X['temperature'] * X['humidity'] / 100
        
        # Rainfall × Humidity (moisture availability)
        X['rainfall_humidity_interaction'] = X['rainfall'] * X['humidity'] / 100
        
        # Temperature × Rainfall (growing season indicator)
        X['temp_rainfall_interaction'] = X['temperature'] * np.log1p(X['rainfall'])
        
        return X
    
    def _engineer_ph_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer pH-nutrient interaction features.
        
        pH affects nutrient availability and uptake by plants.
        
        Args:
            X: DataFrame with pH and nutrient columns
            
        Returns:
            DataFrame with added pH interaction features
        """
        X = X.copy()
        
        # pH × N (nitrogen availability affected by pH)
        X['ph_N_interaction'] = X['ph'] * X['N']
        
        # pH × P (phosphorus availability highly pH-dependent)
        X['ph_P_interaction'] = X['ph'] * X['P']
        
        # pH × K (potassium availability affected by pH)
        X['ph_K_interaction'] = X['ph'] * X['K']
        
        return X
    
    def _engineer_composite_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer composite agricultural indicators.
        
        Args:
            X: DataFrame with environmental features
            
        Returns:
            DataFrame with added composite indicators
        """
        X = X.copy()
        
        # Moisture index (combined humidity and rainfall)
        X['moisture_index'] = (X['humidity'] / 100) * np.log1p(X['rainfall'])
        
        # Growing conditions index (normalized temperature and humidity)
        # Higher values indicate better growing conditions
        temp_norm = (X['temperature'] - X['temperature'].min()) / (X['temperature'].max() - X['temperature'].min() + 1e-10)
        humidity_norm = X['humidity'] / 100
        X['growing_conditions_index'] = (temp_norm + humidity_norm) / 2
        
        return X
    
    def _apply_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            X: DataFrame with original features
            
        Returns:
            DataFrame with engineered features
        """
        X = X.copy()
        
        # Apply log transform to rainfall if specified
        if self.apply_log_to_rainfall:
            X['rainfall_log'] = np.log1p(X['rainfall'])
        
        # Apply all feature engineering methods
        X = self._engineer_nutrient_ratios(X)
        X = self._engineer_nutrient_totals(X)
        X = self._engineer_environmental_interactions(X)
        X = self._engineer_ph_interactions(X)
        X = self._engineer_composite_indicators(X)
        
        return X
    
    def _validate_engineered_features(self, X: pd.DataFrame) -> None:
        """
        Validate engineered features for NaN, inf, and extreme values.
        
        Args:
            X: DataFrame with engineered features
            
        Raises:
            ValueError: If validation fails
        """
        # Check for NaN values
        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"Engineered features contain NaN values in columns: {nan_cols}")
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            inf_cols = X.columns[np.isinf(X.select_dtypes(include=[np.number])).any()].tolist()
            raise ValueError(f"Engineered features contain infinite values in columns: {inf_cols}")
        
        # Check for extreme values (beyond reasonable bounds)
        # Use a more lenient threshold since scaled values can be large
        for col in X.columns:
            if X[col].abs().max() > 1e8:
                raise ValueError(f"Feature '{col}' contains extreme values (max: {X[col].abs().max()})")
    
    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler.
        
        CRITICAL: Scaler is fit ONLY on training data to prevent data leakage.
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        # Initialize scaler
        if self.scaling_method == 'standard':
            self.scaler_ = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")
        
        # Fit scaler ONLY on training data
        X_train_scaled = pd.DataFrame(
            self.scaler_.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data using fitted scaler
        X_test_scaled = pd.DataFrame(
            self.scaler_.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled
    
    def _encode_labels(self, y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode categorical labels to numerical values.
        
        Args:
            y_train: Training labels
            y_test: Testing labels
            
        Returns:
            Tuple of (encoded_y_train, encoded_y_test)
        """
        self.label_encoder_ = LabelEncoder()
        
        # Fit encoder on training labels
        y_train_encoded = self.label_encoder_.fit_transform(y_train)
        
        # Transform test labels
        y_test_encoded = self.label_encoder_.transform(y_test)
        
        return y_train_encoded, y_test_encoded
    
    def fit_transform(
        self, 
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform data.
        
        This method:
        1. Validates input data
        2. Splits into train/test sets
        3. Engineers features
        4. Scales features (fit on train only)
        5. Encodes labels
        
        Args:
            data: Input DataFrame with features and target
            
        Returns:
            Tuple of (X_train_processed, X_test_processed, y_train_encoded, y_test_encoded)
        """
        # Validate input
        self._validate_input_data(data, require_target=True)
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(data)
        
        # Engineer features
        X_train_eng = self._apply_feature_engineering(X_train)
        X_test_eng = self._apply_feature_engineering(X_test)
        
        # Validate engineered features
        self._validate_engineered_features(X_train_eng)
        self._validate_engineered_features(X_test_eng)
        
        # Store feature names
        self.feature_names_ = X_train_eng.columns.tolist()
        
        # Scale features (fit on train only!)
        X_train_scaled, X_test_scaled = self._scale_features(X_train_eng, X_test_eng)
        
        # Encode labels
        y_train_encoded, y_test_encoded = self._encode_labels(y_train, y_test)
        
        # Mark as fitted
        self._is_fitted = True
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X: DataFrame with original features
            
        Returns:
            Transformed DataFrame
            
        Raises:
            ValueError: If preprocessor is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        # Validate input (no target required)
        self._validate_input_data(X, require_target=False)
        
        # Engineer features
        X_eng = self._apply_feature_engineering(X)
        
        # Validate engineered features
        self._validate_engineered_features(X_eng)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler_.transform(X_eng),
            columns=X_eng.columns,
            index=X_eng.index
        )
        
        return X_scaled
    
    def inverse_transform_features(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            X_scaled: Scaled features
            
        Returns:
            Features in original scale
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform.")
        
        X_original = pd.DataFrame(
            self.scaler_.inverse_transform(X_scaled),
            columns=X_scaled.columns,
            index=X_scaled.index
        )
        
        return X_original
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Inverse transform encoded labels back to original labels.
        
        Args:
            y_encoded: Encoded labels
            
        Returns:
            Original labels
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform.")
        
        return self.label_encoder_.inverse_transform(y_encoded)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features after engineering.
        
        Returns:
            List of feature names
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted first.")
        
        return self.feature_names_.copy()
    
    def save_pipeline(self, output_dir: str) -> Dict[str, str]:
        """
        Save fitted pipeline components to disk.
        
        Args:
            output_dir: Directory to save pipeline components
            
        Returns:
            Dictionary mapping component names to file paths
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before saving.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save scaler
        scaler_path = output_path / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler_, f)
        saved_files['scaler'] = str(scaler_path)
        
        # Save label encoder
        encoder_path = output_path / 'label_encoder.pkl'
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder_, f)
        saved_files['label_encoder'] = str(encoder_path)
        
        # Save feature names
        features_path = output_path / 'feature_names.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names_, f, indent=2)
        saved_files['feature_names'] = str(features_path)
        
        # Save preprocessor config
        config_path = output_path / 'preprocessor_config.json'
        config = {
            'test_size': self.test_size,
            'random_state': self.random_state,
            'scaling_method': self.scaling_method,
            'apply_log_to_rainfall': self.apply_log_to_rainfall,
            'original_features': self.original_features_,
            'target_column': self.target_column_
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        saved_files['config'] = str(config_path)
        
        return saved_files
    
    @classmethod
    def load_pipeline(cls, input_dir: str) -> 'CropPreprocessor':
        """
        Load fitted pipeline from disk.
        
        Args:
            input_dir: Directory containing saved pipeline components
            
        Returns:
            Loaded CropPreprocessor instance
        """
        input_path = Path(input_dir)
        
        # Load config
        config_path = input_path / 'preprocessor_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create preprocessor instance
        preprocessor = cls(
            test_size=config['test_size'],
            random_state=config['random_state'],
            scaling_method=config['scaling_method'],
            apply_log_to_rainfall=config['apply_log_to_rainfall']
        )
        
        # Load scaler
        scaler_path = input_path / 'scaler.pkl'
        with open(scaler_path, 'rb') as f:
            preprocessor.scaler_ = pickle.load(f)
        
        # Load label encoder
        encoder_path = input_path / 'label_encoder.pkl'
        with open(encoder_path, 'rb') as f:
            preprocessor.label_encoder_ = pickle.load(f)
        
        # Load feature names
        features_path = input_path / 'feature_names.json'
        with open(features_path, 'r') as f:
            preprocessor.feature_names_ = json.load(f)
        
        # Mark as fitted
        preprocessor._is_fitted = True
        
        return preprocessor


def preprocess_and_save(
    data_path: Optional[str] = None,
    output_dir: str = "data/processed",
    models_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, str]:
    """
    Convenience function to preprocess data and save all outputs.
    
    Args:
        data_path: Path to input CSV file (if None, uses default loader)
        output_dir: Directory to save preprocessed datasets
        models_dir: Directory to save pipeline components
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Dictionary with paths to saved files
    """
    # Load data
    if data_path is None:
        from src.data.loader import CropDataLoader
        loader = CropDataLoader()
        data = loader.load_data()
    else:
        data = pd.read_csv(data_path)
    
    # Initialize preprocessor
    preprocessor = CropPreprocessor(
        test_size=test_size,
        random_state=random_state
    )
    
    # Fit and transform
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(data)
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save preprocessed datasets
    print("Saving preprocessed datasets...")
    
    # Combine features and target for saving
    train_data = X_train.copy()
    train_data['label_encoded'] = y_train
    train_path = output_path / 'preprocessed_train.csv'
    train_data.to_csv(train_path, index=False)
    saved_files['train_data'] = str(train_path)
    
    test_data = X_test.copy()
    test_data['label_encoded'] = y_test
    test_path = output_path / 'preprocessed_test.csv'
    test_data.to_csv(test_path, index=False)
    saved_files['test_data'] = str(test_path)
    
    # Save pipeline
    print("Saving pipeline components...")
    pipeline_files = preprocessor.save_pipeline(models_dir)
    saved_files.update(pipeline_files)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Original features: {len(preprocessor.original_features_)}")
    print(f"Engineered features: {len(preprocessor.get_feature_names())}")
    print(f"Number of crops: {len(preprocessor.label_encoder_.classes_)}")
    print("\nSaved files:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")
    print("=" * 60)
    
    return saved_files


if __name__ == "__main__":
    # Run preprocessing when script is executed directly
    preprocess_and_save()
