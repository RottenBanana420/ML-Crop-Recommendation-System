"""
Comprehensive Test Suite for Crop Preprocessing Pipeline.

This test suite is designed to FAIL if:
- Data leakage occurs (scaler fit on test data, target used in features)
- Scaling is incorrect (mean != 0, std != 1 for training data)
- Engineered features have NaN or infinite values
- Train-test split proportions are wrong
- Pipeline cannot handle edge cases

CRITICAL: Tests should NEVER be modified to pass. Only fix the preprocessing code.

Author: Crop Recommendation System
Date: 2025-11-24
"""

import pytest
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Tuple

from src.features.preprocessing import CropPreprocessor, preprocess_and_save
from src.data.loader import CropDataLoader


# ==================== FIXTURES ====================

@pytest.fixture
def sample_data():
    """Load sample crop recommendation data."""
    loader = CropDataLoader()
    return loader.load_data()


@pytest.fixture
def small_sample_data(sample_data):
    """Create a small sample for faster testing."""
    # Sample 200 rows with stratification
    # Use a different approach to avoid FutureWarning
    sampled_dfs = []
    for label, group in sample_data.groupby('label'):
        sampled_dfs.append(group.sample(min(len(group), 20), random_state=42))
    return pd.concat(sampled_dfs, ignore_index=True)


@pytest.fixture
def fitted_preprocessor(small_sample_data):
    """Create a fitted preprocessor instance."""
    preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(small_sample_data)
    return preprocessor, X_train, X_test, y_train, y_test


# ==================== TEST DATA LEAKAGE ====================

class TestDataLeakage:
    """Tests to detect and prevent data leakage."""
    
    def test_no_data_leakage_in_scaling(self, small_sample_data):
        """
        CRITICAL: Verify scaler is fit ONLY on training data.
        
        This test ensures the scaler's mean and std are computed from
        training data only, preventing information leakage from test set.
        """
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(small_sample_data)
        
        # Get the scaler's learned parameters
        scaler_mean = preprocessor.scaler_.mean_
        scaler_std = preprocessor.scaler_.scale_
        
        # Manually compute what the mean/std should be if fit on train only
        # First, we need to recreate the train set and engineer features
        from sklearn.model_selection import train_test_split
        
        X = small_sample_data[preprocessor.original_features_].copy()
        y = small_sample_data[preprocessor.target_column_].copy()
        
        X_train_raw, X_test_raw, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Engineer features on train set
        X_train_eng = preprocessor._apply_feature_engineering(X_train_raw)
        
        # Compute expected mean and std from training data
        expected_mean = X_train_eng.mean().values
        expected_std = X_train_eng.std().values
        
        # Verify scaler was fit on training data only
        # Note: Small tolerance needed due to ddof differences (sklearn uses ddof=0)
        np.testing.assert_allclose(
            scaler_mean, expected_mean, rtol=1e-6,
            err_msg="Scaler mean does not match training data - possible data leakage!"
        )
        
        # Adjust std calculation to match sklearn (ddof=0)
        expected_std_sklearn = X_train_eng.std(ddof=0).values
        np.testing.assert_allclose(
            scaler_std, expected_std_sklearn, rtol=1e-6,
            err_msg="Scaler std does not match training data - possible data leakage!"
        )
    
    def test_no_target_leakage(self, small_sample_data):
        """
        Verify target variable is not used in feature engineering.
        
        Feature engineering should only use input features, never the target.
        """
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(small_sample_data)
        
        # Verify target column is not in feature names
        feature_names = preprocessor.get_feature_names()
        assert 'label' not in feature_names, "Target 'label' found in features - data leakage!"
        assert 'label_encoded' not in feature_names, "Encoded target found in features - data leakage!"
    
    def test_train_test_independence(self, small_sample_data):
        """
        Verify no overlap between train and test sets.
        
        Train and test sets must be completely independent.
        """
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(small_sample_data)
        
        # Check indices don't overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        overlap = train_indices.intersection(test_indices)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices between train and test!"
        
        # Verify total samples = train + test
        total_samples = len(small_sample_data)
        assert len(X_train) + len(X_test) == total_samples, "Sample count mismatch!"


# ==================== TEST SCALING ====================

class TestScaling:
    """Tests for feature scaling correctness."""
    
    def test_scaler_mean_std(self, fitted_preprocessor):
        """
        Verify scaled training data has mean ≈ 0 and std ≈ 1.
        
        StandardScaler should transform data to have zero mean and unit variance.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        # Check mean is close to 0 (within tolerance)
        train_means = X_train.mean()
        assert np.allclose(train_means, 0, atol=1e-10), \
            f"Training data mean not close to 0: {train_means.describe()}"
        
        # Check std is close to 1 (within tolerance)
        train_stds = X_train.std()
        assert np.allclose(train_stds, 1, atol=0.1), \
            f"Training data std not close to 1: {train_stds.describe()}"
    
    def test_scaler_consistency(self, fitted_preprocessor):
        """
        Ensure same scaler transforms train and test identically.
        
        The scaler fitted on train should be used for both train and test.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        # Verify scaler exists and is fitted
        assert preprocessor.scaler_ is not None, "Scaler not fitted!"
        assert hasattr(preprocessor.scaler_, 'mean_'), "Scaler not properly fitted!"
        
        # Test that scaler produces consistent results
        # Take a sample from test set and transform it twice
        sample = X_test.iloc[:5]
        
        # Inverse transform then transform again
        sample_original = preprocessor.inverse_transform_features(sample)
        sample_rescaled = pd.DataFrame(
            preprocessor.scaler_.transform(sample_original),
            columns=sample.columns
        )
        
        # Should get same result
        np.testing.assert_allclose(
            sample.values, sample_rescaled.values, rtol=1e-10,
            err_msg="Scaler not producing consistent results!"
        )
    
    def test_inverse_transform(self, fitted_preprocessor):
        """
        Verify scaling is reversible through inverse transform.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        # Take a sample
        sample_scaled = X_train.iloc[:10]
        
        # Inverse transform
        sample_original = preprocessor.inverse_transform_features(sample_scaled)
        
        # Transform again
        sample_rescaled = pd.DataFrame(
            preprocessor.scaler_.transform(sample_original),
            columns=sample_scaled.columns,
            index=sample_scaled.index
        )
        
        # Should match original scaled values
        np.testing.assert_allclose(
            sample_scaled.values, sample_rescaled.values, rtol=1e-10,
            err_msg="Inverse transform not reversible!"
        )


# ==================== TEST FEATURE ENGINEERING ====================

class TestFeatureEngineering:
    """Tests for engineered features validity."""
    
    def test_no_nan_values(self, fitted_preprocessor):
        """
        CRITICAL: Check all engineered features for NaN values.
        
        NaN values will break ML models and indicate engineering errors.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        # Check training data
        assert not X_train.isnull().any().any(), \
            f"NaN values found in training data: {X_train.columns[X_train.isnull().any()].tolist()}"
        
        # Check test data
        assert not X_test.isnull().any().any(), \
            f"NaN values found in test data: {X_test.columns[X_test.isnull().any()].tolist()}"
    
    def test_no_infinite_values(self, fitted_preprocessor):
        """
        CRITICAL: Check all engineered features for infinite values.
        
        Infinite values will break ML models and indicate engineering errors.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        # Check training data
        assert not np.isinf(X_train).any().any(), \
            f"Infinite values found in training data: {X_train.columns[np.isinf(X_train).any()].tolist()}"
        
        # Check test data
        assert not np.isinf(X_test).any().any(), \
            f"Infinite values found in test data: {X_test.columns[np.isinf(X_test).any()].tolist()}"
    
    def test_nutrient_ratios_valid(self, small_sample_data):
        """
        Verify nutrient ratios are positive and reasonable.
        
        Nutrient ratios should be positive (nutrients are non-negative).
        """
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(small_sample_data)
        
        # Before scaling, check raw engineered features
        X = small_sample_data[preprocessor.original_features_].copy()
        X_eng = preprocessor._apply_feature_engineering(X)
        
        # Check ratio features are positive
        ratio_features = ['N_to_P_ratio', 'N_to_K_ratio', 'P_to_K_ratio']
        for feature in ratio_features:
            assert (X_eng[feature] >= 0).all(), \
                f"Negative values found in {feature}!"
            
            # Check for unreasonably large ratios (might indicate division issues)
            assert (X_eng[feature] < 1000).all(), \
                f"Unreasonably large values in {feature}: max={X_eng[feature].max()}"
    
    def test_feature_count(self, fitted_preprocessor):
        """
        Ensure correct number of features after engineering.
        
        Should have original features + engineered features.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        feature_names = preprocessor.get_feature_names()
        
        # Expected features:
        # Original: 7 (N, P, K, temperature, humidity, ph, rainfall)
        # Engineered: rainfall_log, 3 ratios, 3 totals, 3 env interactions, 
        #             3 pH interactions, 2 composite = 14 new features
        # Total: 7 + 14 = 21 features
        
        expected_min_features = 20  # At least this many
        assert len(feature_names) >= expected_min_features, \
            f"Expected at least {expected_min_features} features, got {len(feature_names)}"
        
        # Verify train and test have same number of features
        assert X_train.shape[1] == X_test.shape[1], \
            "Train and test have different number of features!"


# ==================== TEST TRAIN-TEST SPLIT ====================

class TestTrainTestSplit:
    """Tests for train-test split correctness."""
    
    def test_split_proportions(self, small_sample_data):
        """
        CRITICAL: Verify train-test split ratio is correct.
        
        With test_size=0.2, test set should be ~20% of data.
        """
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(small_sample_data)
        
        total_samples = len(small_sample_data)
        test_samples = len(X_test)
        
        actual_test_ratio = test_samples / total_samples
        expected_test_ratio = 0.2
        
        # Allow 0.5% tolerance due to rounding with stratification
        assert abs(actual_test_ratio - expected_test_ratio) < 0.005, \
            f"Test ratio {actual_test_ratio:.3f} not close to expected {expected_test_ratio}"
    
    def test_stratification(self, small_sample_data):
        """
        Ensure balanced crop distribution in train and test sets.
        
        Stratification should maintain similar crop proportions in both sets.
        """
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(small_sample_data)
        
        # Get original label distribution
        original_dist = small_sample_data['label'].value_counts(normalize=True).sort_index()
        
        # Decode labels
        y_train_decoded = preprocessor.inverse_transform_labels(y_train)
        y_test_decoded = preprocessor.inverse_transform_labels(y_test)
        
        # Get train and test distributions
        train_dist = pd.Series(y_train_decoded).value_counts(normalize=True).sort_index()
        test_dist = pd.Series(y_test_decoded).value_counts(normalize=True).sort_index()
        
        # Check distributions are similar (within 10% relative difference)
        for crop in original_dist.index:
            if crop in train_dist.index and crop in test_dist.index:
                rel_diff = abs(train_dist[crop] - test_dist[crop]) / original_dist[crop]
                assert rel_diff < 0.15, \
                    f"Crop '{crop}' distribution differs too much: train={train_dist[crop]:.3f}, test={test_dist[crop]:.3f}"
    
    def test_no_data_loss(self, small_sample_data):
        """
        Verify total samples = train + test (no data lost in split).
        """
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(small_sample_data)
        
        total_original = len(small_sample_data)
        total_split = len(X_train) + len(X_test)
        
        assert total_original == total_split, \
            f"Data loss detected: original={total_original}, after split={total_split}"


# ==================== TEST EDGE CASES ====================

class TestEdgeCases:
    """Tests for edge cases and boundary values."""
    
    def test_zero_values_handling(self):
        """
        Test preprocessing with zero nutrient values.
        
        Division by zero should be handled gracefully.
        """
        # Create data with some zero values and multiple crops
        data = pd.DataFrame({
            'N': [0, 10, 20, 0, 10, 20],
            'P': [10, 0, 20, 10, 0, 20],
            'K': [10, 20, 0, 10, 20, 0],
            'temperature': [25, 26, 27, 25, 26, 27],
            'humidity': [80, 81, 82, 80, 81, 82],
            'ph': [6.5, 6.6, 6.7, 6.5, 6.6, 6.7],
            'rainfall': [100, 110, 120, 100, 110, 120],
            'label': ['rice', 'maize', 'chickpea', 'rice', 'maize', 'chickpea']
        })
        
        preprocessor = CropPreprocessor(test_size=0.33, random_state=42)
        
        # Should not raise error
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(data)
        
        # Check no NaN or inf values
        assert not X_train.isnull().any().any(), "NaN values from zero handling!"
        assert not np.isinf(X_train).any().any(), "Inf values from zero handling!"
    
    def test_extreme_values_handling(self):
        """
        Test with extreme but valid values.
        """
        # Need at least 2 samples per class for stratification
        data = pd.DataFrame({
            'N': [100, 0, 50, 100, 0, 50],
            'P': [80, 0, 40, 80, 0, 40],
            'K': [80, 0, 40, 80, 0, 40],
            'temperature': [45, 10, 25, 45, 10, 25],  # Extreme temperatures
            'humidity': [95, 20, 60, 95, 20, 60],     # Extreme humidity
            'ph': [9.0, 4.0, 6.5, 9.0, 4.0, 6.5],     # Extreme pH
            'rainfall': [500, 10, 100, 500, 10, 100], # Extreme rainfall
            'label': ['rice', 'maize', 'chickpea', 'rice', 'maize', 'chickpea']
        })
        
        preprocessor = CropPreprocessor(test_size=0.33, random_state=42)
        
        # Should not raise error
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(data)
        
        # Verify no invalid values
        assert not X_train.isnull().any().any()
        assert not np.isinf(X_train).any().any()
    
    def test_single_crop_handling(self):
        """
        Test with data containing only one crop type.
        
        Should fail gracefully as stratification requires multiple classes.
        """
        data = pd.DataFrame({
            'N': [90, 85, 80],
            'P': [40, 45, 50],
            'K': [40, 45, 50],
            'temperature': [25, 26, 27],
            'humidity': [80, 81, 82],
            'ph': [6.5, 6.6, 6.7],
            'rainfall': [200, 210, 220],
            'label': ['rice', 'rice', 'rice']  # Only one crop
        })
        
        preprocessor = CropPreprocessor(test_size=0.33, random_state=42)
        
        # Should raise ValueError due to insufficient samples per class
        with pytest.raises(ValueError, match="Cannot perform stratified split"):
            preprocessor.fit_transform(data)
    
    def test_missing_features_error(self):
        """
        Verify error on missing required features.
        """
        # Data missing 'rainfall' feature
        data = pd.DataFrame({
            'N': [90, 85],
            'P': [40, 45],
            'K': [40, 45],
            'temperature': [25, 26],
            'humidity': [80, 81],
            'ph': [6.5, 6.6],
            'label': ['rice', 'maize']
        })
        
        preprocessor = CropPreprocessor(test_size=0.5, random_state=42)
        
        with pytest.raises(ValueError, match="Missing required features"):
            preprocessor.fit_transform(data)
    
    def test_invalid_data_types_error(self):
        """
        Verify error on invalid data types.
        """
        data = pd.DataFrame({
            'N': ['high', 'low', 'medium'],  # String instead of numeric
            'P': [40, 45, 50],
            'K': [40, 45, 50],
            'temperature': [25, 26, 27],
            'humidity': [80, 81, 82],
            'ph': [6.5, 6.6, 6.7],
            'rainfall': [200, 210, 220],
            'label': ['rice', 'maize', 'rice']
        })
        
        preprocessor = CropPreprocessor(test_size=0.33, random_state=42)
        
        with pytest.raises(ValueError, match="must be numeric"):
            preprocessor.fit_transform(data)
    
    def test_empty_dataframe_error(self):
        """
        Verify error on empty DataFrame.
        """
        data = pd.DataFrame()
        
        preprocessor = CropPreprocessor(test_size=0.2, random_state=42)
        
        with pytest.raises(ValueError, match="empty"):
            preprocessor.fit_transform(data)


# ==================== TEST PIPELINE REUSABILITY ====================

class TestPipelineReusability:
    """Tests for pipeline save/load and reusability."""
    
    def test_save_load_pipeline(self, fitted_preprocessor, tmp_path):
        """
        Verify pipeline can be saved and loaded.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        # Save pipeline
        saved_files = preprocessor.save_pipeline(str(tmp_path))
        
        # Verify files were created
        assert 'scaler' in saved_files
        assert 'label_encoder' in saved_files
        assert 'feature_names' in saved_files
        assert 'config' in saved_files
        
        for file_path in saved_files.values():
            assert Path(file_path).exists(), f"File not created: {file_path}"
        
        # Load pipeline
        loaded_preprocessor = CropPreprocessor.load_pipeline(str(tmp_path))
        
        # Verify loaded preprocessor is fitted
        assert loaded_preprocessor._is_fitted
        assert loaded_preprocessor.scaler_ is not None
        assert loaded_preprocessor.label_encoder_ is not None
    
    def test_transform_new_data(self, fitted_preprocessor, small_sample_data):
        """
        Test pipeline on completely new data.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        # Create new sample data (different from train/test)
        new_data = small_sample_data[preprocessor.original_features_].sample(5, random_state=99)
        
        # Transform new data
        new_transformed = preprocessor.transform(new_data)
        
        # Verify output shape
        assert new_transformed.shape[0] == 5
        assert new_transformed.shape[1] == X_train.shape[1]
        
        # Verify no NaN or inf
        assert not new_transformed.isnull().any().any()
        assert not np.isinf(new_transformed).any().any()
    
    def test_feature_names_consistency(self, fitted_preprocessor, tmp_path):
        """
        Ensure feature names match after save/load.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        # Get original feature names
        original_features = preprocessor.get_feature_names()
        
        # Save and load
        preprocessor.save_pipeline(str(tmp_path))
        loaded_preprocessor = CropPreprocessor.load_pipeline(str(tmp_path))
        
        # Get loaded feature names
        loaded_features = loaded_preprocessor.get_feature_names()
        
        # Should match exactly
        assert original_features == loaded_features, \
            "Feature names changed after save/load!"
    
    def test_label_encoding_consistency(self, fitted_preprocessor, tmp_path):
        """
        Verify label encoding is consistent after save/load.
        """
        preprocessor, X_train, X_test, y_train, y_test = fitted_preprocessor
        
        # Get original classes
        original_classes = preprocessor.label_encoder_.classes_.tolist()
        
        # Save and load
        preprocessor.save_pipeline(str(tmp_path))
        loaded_preprocessor = CropPreprocessor.load_pipeline(str(tmp_path))
        
        # Get loaded classes
        loaded_classes = loaded_preprocessor.label_encoder_.classes_.tolist()
        
        # Should match exactly
        assert original_classes == loaded_classes, \
            "Label encoder classes changed after save/load!"


# ==================== TEST INTEGRATION ====================

class TestIntegration:
    """Integration tests for end-to-end preprocessing."""
    
    def test_full_preprocessing_pipeline(self, sample_data, tmp_path):
        """
        Test complete preprocessing pipeline from start to finish.
        """
        # Run full preprocessing
        saved_files = preprocess_and_save(
            data_path=None,
            output_dir=str(tmp_path / 'processed'),
            models_dir=str(tmp_path / 'models'),
            test_size=0.2,
            random_state=42
        )
        
        # Verify all files were created
        assert 'train_data' in saved_files
        assert 'test_data' in saved_files
        assert 'scaler' in saved_files
        assert 'label_encoder' in saved_files
        
        # Load and verify preprocessed data
        train_data = pd.read_csv(saved_files['train_data'])
        test_data = pd.read_csv(saved_files['test_data'])
        
        # Verify no NaN values
        assert not train_data.isnull().any().any()
        assert not test_data.isnull().any().any()
        
        # Verify label column exists
        assert 'label_encoded' in train_data.columns
        assert 'label_encoded' in test_data.columns
        
        # Verify can load pipeline
        loaded_preprocessor = CropPreprocessor.load_pipeline(str(tmp_path / 'models'))
        assert loaded_preprocessor._is_fitted


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
