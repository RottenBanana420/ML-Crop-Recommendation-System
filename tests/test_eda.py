"""
Comprehensive Test Suite for Exploratory Data Analysis Module.

These tests are designed to FAIL if:
- Any expected visualization is missing
- Statistical calculations are incorrect
- Data quality checks are skipped
- Important insights are not captured

CRITICAL: If any test fails, modify the CODEBASE (eda.py), NOT the tests.

Author: Crop Recommendation System
Date: 2025-11-23
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.data.loader import CropDataLoader
from src.analysis.eda import CropEDA, run_comprehensive_eda


@pytest.fixture
def sample_data():
    """Load the crop recommendation dataset for testing."""
    loader = CropDataLoader()
    return loader.load_data()


@pytest.fixture
def eda_instance(sample_data, tmp_path):
    """Create an EDA instance with temporary output directory."""
    return CropEDA(sample_data, output_dir=str(tmp_path))


@pytest.fixture
def eda_with_outputs(sample_data, tmp_path):
    """Create an EDA instance and run all analyses."""
    eda = CropEDA(sample_data, output_dir=str(tmp_path))
    eda.run_all_analyses()
    return eda


class TestVisualizationGeneration:
    """Tests to verify all visualizations are generated."""
    
    def test_all_visualizations_generated(self, eda_with_outputs):
        """Test that all expected visualization files are created."""
        viz_dir = eda_with_outputs.viz_dir
        
        expected_files = [
            'distribution_plots.png',
            'box_plots.png',
            'correlation_heatmap.png',
            'crop_distribution.png',
            'crop_wise_features.png',
            'pair_plot.png',
            'scatter_plots.png'
        ]
        
        for filename in expected_files:
            file_path = viz_dir / filename
            assert file_path.exists(), f"Visualization '{filename}' was not generated"
            assert file_path.stat().st_size > 0, f"Visualization '{filename}' is empty"
    
    def test_distribution_plots_exist(self, eda_with_outputs):
        """Test that distribution plots file exists and is valid."""
        file_path = eda_with_outputs.viz_dir / 'distribution_plots.png'
        assert file_path.exists(), "Distribution plots file not found"
        assert file_path.stat().st_size > 10000, "Distribution plots file is too small"
    
    def test_box_plots_exist(self, eda_with_outputs):
        """Test that box plots file exists and is valid."""
        file_path = eda_with_outputs.viz_dir / 'box_plots.png'
        assert file_path.exists(), "Box plots file not found"
        assert file_path.stat().st_size > 10000, "Box plots file is too small"
    
    def test_correlation_heatmap_exists(self, eda_with_outputs):
        """Test that correlation heatmap exists and is valid."""
        file_path = eda_with_outputs.viz_dir / 'correlation_heatmap.png'
        assert file_path.exists(), "Correlation heatmap not found"
        assert file_path.stat().st_size > 10000, "Correlation heatmap file is too small"
    
    def test_crop_distribution_plot_exists(self, eda_with_outputs):
        """Test that crop distribution plot exists and is valid."""
        file_path = eda_with_outputs.viz_dir / 'crop_distribution.png'
        assert file_path.exists(), "Crop distribution plot not found"
        assert file_path.stat().st_size > 10000, "Crop distribution plot file is too small"
    
    def test_crop_wise_plots_exist(self, eda_with_outputs):
        """Test that crop-wise feature plots exist and are valid."""
        file_path = eda_with_outputs.viz_dir / 'crop_wise_features.png'
        assert file_path.exists(), "Crop-wise feature plots not found"
        assert file_path.stat().st_size > 10000, "Crop-wise feature plots file is too small"
    
    def test_pair_plot_exists(self, eda_with_outputs):
        """Test that pair plot exists and is valid."""
        file_path = eda_with_outputs.viz_dir / 'pair_plot.png'
        assert file_path.exists(), "Pair plot not found"
        assert file_path.stat().st_size > 10000, "Pair plot file is too small"
    
    def test_scatter_plots_exist(self, eda_with_outputs):
        """Test that scatter plots exist and are valid."""
        file_path = eda_with_outputs.viz_dir / 'scatter_plots.png'
        assert file_path.exists(), "Scatter plots not found"
        assert file_path.stat().st_size > 10000, "Scatter plots file is too small"


class TestStatisticalCalculations:
    """Tests to verify statistical calculations are correct."""
    
    def test_descriptive_statistics_complete(self, eda_instance):
        """Test that descriptive statistics are calculated for all features."""
        stats_df = eda_instance.generate_descriptive_statistics()
        
        expected_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Check all features are present
        for feature in expected_features:
            assert feature in stats_df.columns, f"Feature '{feature}' missing from statistics"
        
        # Check all expected statistics are present
        expected_stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']
        for stat in expected_stats:
            assert stat in stats_df.index, f"Statistic '{stat}' missing from results"
    
    def test_descriptive_statistics_accuracy(self, eda_instance, sample_data):
        """Test that descriptive statistics match pandas calculations."""
        stats_df = eda_instance.generate_descriptive_statistics()
        
        # Verify mean calculations
        for feature in eda_instance.numerical_features:
            calculated_mean = stats_df.loc['mean', feature]
            expected_mean = sample_data[feature].mean()
            assert abs(calculated_mean - expected_mean) < 0.01, \
                f"Mean calculation incorrect for '{feature}'"
    
    def test_crop_wise_statistics_all_crops(self, eda_instance, sample_data):
        """Test that crop-wise statistics include all crops."""
        crop_stats = eda_instance.calculate_crop_wise_statistics()
        
        unique_crops = sample_data['label'].unique()
        
        # Check each feature has statistics for all crops
        for feature, stats_df in crop_stats.items():
            for crop in unique_crops:
                assert crop in stats_df.index, \
                    f"Crop '{crop}' missing from statistics for feature '{feature}'"
    
    def test_crop_wise_statistics_accuracy(self, eda_instance, sample_data):
        """Test that crop-wise statistics are calculated correctly."""
        crop_stats = eda_instance.calculate_crop_wise_statistics()
        
        # Verify mean calculation for a specific crop and feature
        feature = 'N'
        crop = sample_data['label'].iloc[0]
        
        calculated_mean = crop_stats[feature].loc[crop, 'mean']
        expected_mean = sample_data[sample_data['label'] == crop][feature].mean()
        
        assert abs(calculated_mean - expected_mean) < 0.01, \
            f"Crop-wise mean calculation incorrect for '{crop}' and '{feature}'"
    
    def test_correlation_matrix_shape(self, eda_instance):
        """Test that correlation matrix has correct dimensions."""
        corr_matrix = eda_instance.calculate_correlation_matrix()
        
        n_features = len(eda_instance.numerical_features)
        
        assert corr_matrix.shape == (n_features, n_features), \
            f"Correlation matrix shape incorrect. Expected ({n_features}, {n_features}), got {corr_matrix.shape}"
    
    def test_correlation_matrix_values(self, eda_instance):
        """Test that correlation values are in valid range [-1, 1]."""
        corr_matrix = eda_instance.calculate_correlation_matrix()
        
        # Check all values are in [-1, 1]
        assert (corr_matrix >= -1).all().all(), "Correlation values below -1 found"
        assert (corr_matrix <= 1).all().all(), "Correlation values above 1 found"
        
        # Check diagonal is all 1s (self-correlation)
        diagonal = np.diag(corr_matrix)
        assert np.allclose(diagonal, 1.0), "Diagonal values are not 1.0"
    
    def test_correlation_matrix_symmetry(self, eda_instance):
        """Test that correlation matrix is symmetric."""
        corr_matrix = eda_instance.calculate_correlation_matrix()
        
        assert np.allclose(corr_matrix, corr_matrix.T), \
            "Correlation matrix is not symmetric"
    
    def test_distribution_analysis_complete(self, eda_instance):
        """Test that distribution analysis covers all features."""
        dist_info = eda_instance.analyze_feature_distributions()
        
        for feature in eda_instance.numerical_features:
            assert feature in dist_info, f"Feature '{feature}' missing from distribution analysis"
            assert 'skewness' in dist_info[feature], f"Skewness missing for '{feature}'"
            assert 'kurtosis' in dist_info[feature], f"Kurtosis missing for '{feature}'"
            assert 'is_normal' in dist_info[feature], f"Normality test missing for '{feature}'"


class TestDataQualityChecks:
    """Tests to verify data quality checks are performed."""
    
    def test_data_quality_check_performed(self, eda_instance):
        """Test that data quality check is executed."""
        quality_report = eda_instance.check_data_quality()
        
        required_keys = ['total_rows', 'total_columns', 'missing_values', 
                        'missing_percentage', 'duplicate_rows', 'data_types']
        
        for key in required_keys:
            assert key in quality_report, f"Key '{key}' missing from quality report"
    
    def test_outlier_detection_iqr_performed(self, eda_instance):
        """Test that IQR outlier detection is executed for all features."""
        outliers_info = eda_instance.detect_outliers_iqr()
        
        for feature in eda_instance.numerical_features:
            assert feature in outliers_info, f"IQR outlier detection missing for '{feature}'"
            assert 'count' in outliers_info[feature], f"Outlier count missing for '{feature}'"
            assert 'percentage' in outliers_info[feature], f"Outlier percentage missing for '{feature}'"
            assert 'lower_bound' in outliers_info[feature], f"Lower bound missing for '{feature}'"
            assert 'upper_bound' in outliers_info[feature], f"Upper bound missing for '{feature}'"
    
    def test_outlier_detection_zscore_performed(self, eda_instance):
        """Test that Z-score outlier detection is executed for all features."""
        outliers_info = eda_instance.detect_outliers_zscore()
        
        for feature in eda_instance.numerical_features:
            assert feature in outliers_info, f"Z-score outlier detection missing for '{feature}'"
            assert 'count' in outliers_info[feature], f"Outlier count missing for '{feature}'"
            assert 'percentage' in outliers_info[feature], f"Outlier percentage missing for '{feature}'"
            assert 'threshold' in outliers_info[feature], f"Threshold missing for '{feature}'"
    
    def test_outlier_bounds_valid(self, eda_instance):
        """Test that outlier bounds are calculated correctly."""
        outliers_info = eda_instance.detect_outliers_iqr()
        
        for feature in eda_instance.numerical_features:
            lower = outliers_info[feature]['lower_bound']
            upper = outliers_info[feature]['upper_bound']
            
            assert lower < upper, \
                f"Invalid bounds for '{feature}': lower ({lower}) >= upper ({upper})"


class TestCorrelationAnalysis:
    """Tests to verify correlation analysis is complete."""
    
    def test_strong_correlations_identified(self, eda_instance):
        """Test that strong correlations are identified."""
        strong_corrs = eda_instance.identify_strong_correlations(threshold=0.3)
        
        # Should be a list of tuples
        assert isinstance(strong_corrs, list), "Strong correlations must be a list"
        
        # Each item should be a tuple with 3 elements
        for item in strong_corrs:
            assert len(item) == 3, "Each correlation must be a tuple of (feat1, feat2, corr_value)"
            assert abs(item[2]) >= 0.3, f"Correlation value {item[2]} below threshold"
    
    def test_feature_crop_relationships_analyzed(self, eda_instance):
        """Test that feature-crop relationships are analyzed."""
        relationships = eda_instance.analyze_feature_crop_relationships()
        
        for feature in eda_instance.numerical_features:
            assert feature in relationships, f"Feature '{feature}' missing from relationship analysis"
            assert 'f_statistic' in relationships[feature], f"F-statistic missing for '{feature}'"
            assert 'p_value' in relationships[feature], f"P-value missing for '{feature}'"
            assert 'significant' in relationships[feature], f"Significance flag missing for '{feature}'"
    
    def test_correlation_insights_stored(self, eda_with_outputs):
        """Test that correlation insights are stored."""
        insights = eda_with_outputs.insights
        
        assert 'correlations' in insights, "Correlations missing from insights"
        assert 'matrix' in insights['correlations'], "Correlation matrix missing from insights"
        assert 'strong_correlations' in insights['correlations'], \
            "Strong correlations missing from insights"


class TestInsightsDocumentation:
    """Tests to verify insights are properly documented."""
    
    def test_insights_report_generated(self, eda_with_outputs):
        """Test that insights report JSON file is created."""
        insights_file = eda_with_outputs.insights_dir / 'eda_insights.json'
        
        assert insights_file.exists(), "Insights report file not found"
        assert insights_file.stat().st_size > 0, "Insights report file is empty"
    
    def test_insights_report_structure(self, eda_with_outputs):
        """Test that insights report has all required sections."""
        insights = eda_with_outputs.insights
        
        required_sections = [
            'data_quality',
            'statistics',
            'correlations',
            'distributions',
            'crop_patterns',
            'outliers',
            'metadata',
            'key_findings'
        ]
        
        for section in required_sections:
            assert section in insights, f"Section '{section}' missing from insights report"
    
    def test_insights_report_loadable(self, eda_with_outputs):
        """Test that insights report can be loaded as valid JSON."""
        insights_file = eda_with_outputs.insights_dir / 'eda_insights.json'
        
        with open(insights_file, 'r') as f:
            loaded_insights = json.load(f)
        
        assert isinstance(loaded_insights, dict), "Insights must be a dictionary"
        assert len(loaded_insights) > 0, "Insights dictionary is empty"
    
    def test_insights_contain_key_findings(self, eda_with_outputs):
        """Test that key findings are documented."""
        insights = eda_with_outputs.insights
        
        assert 'key_findings' in insights, "Key findings missing from insights"
        
        key_findings = insights['key_findings']
        expected_keys = ['data_quality', 'distributions', 'correlations', 'outliers']
        
        for key in expected_keys:
            assert key in key_findings, f"Key finding '{key}' missing"
    
    def test_insights_contain_crop_patterns(self, eda_with_outputs):
        """Test that crop-specific patterns are captured."""
        insights = eda_with_outputs.insights
        
        assert 'crop_patterns' in insights, "Crop patterns missing from insights"
        
        crop_patterns = insights['crop_patterns']
        assert 'distribution' in crop_patterns, "Crop distribution missing"
        assert 'statistics' in crop_patterns, "Crop-wise statistics missing"
        assert 'feature_relationships' in crop_patterns, "Feature relationships missing"
    
    def test_metadata_complete(self, eda_with_outputs):
        """Test that metadata section is complete."""
        metadata = eda_with_outputs.insights['metadata']
        
        required_keys = ['total_samples', 'total_features', 'total_crops', 'crop_list']
        
        for key in required_keys:
            assert key in metadata, f"Metadata key '{key}' missing"
        
        assert metadata['total_samples'] > 0, "Total samples must be positive"
        assert metadata['total_features'] > 0, "Total features must be positive"
        assert metadata['total_crops'] > 0, "Total crops must be positive"
        assert len(metadata['crop_list']) > 0, "Crop list must not be empty"


class TestFunctionExecution:
    """Tests to verify all analysis functions execute correctly."""
    
    def test_run_all_analyses_executes(self, eda_instance):
        """Test that run_all_analyses executes without errors."""
        outputs = eda_instance.run_all_analyses()
        
        assert isinstance(outputs, dict), "run_all_analyses must return a dictionary"
        assert len(outputs) > 0, "Outputs dictionary is empty"
    
    def test_all_analysis_functions_callable(self, eda_instance):
        """Test that all analysis functions are accessible and callable."""
        functions = [
            'check_data_quality',
            'detect_outliers_iqr',
            'detect_outliers_zscore',
            'generate_descriptive_statistics',
            'calculate_crop_wise_statistics',
            'analyze_feature_distributions',
            'calculate_correlation_matrix',
            'identify_strong_correlations',
            'analyze_feature_crop_relationships',
            'create_distribution_plots',
            'create_box_plots',
            'create_correlation_heatmap',
            'create_crop_distribution_plot',
            'create_crop_wise_feature_plots',
            'create_pair_plot',
            'create_scatter_plots',
            'generate_insights_report'
        ]
        
        for func_name in functions:
            assert hasattr(eda_instance, func_name), f"Function '{func_name}' not found"
            assert callable(getattr(eda_instance, func_name)), \
                f"'{func_name}' is not callable"
    
    def test_run_comprehensive_eda_function(self, tmp_path):
        """Test that the convenience function run_comprehensive_eda works."""
        outputs = run_comprehensive_eda(output_dir=str(tmp_path))
        
        assert isinstance(outputs, dict), "run_comprehensive_eda must return a dictionary"
        assert len(outputs) > 0, "Outputs dictionary is empty"
        
        # Verify key outputs exist
        expected_keys = [
            'distribution_plots',
            'box_plots',
            'correlation_heatmap',
            'crop_distribution',
            'crop_wise_features',
            'pair_plot',
            'scatter_plots',
            'insights_report'
        ]
        
        for key in expected_keys:
            assert key in outputs, f"Output '{key}' missing from results"


class TestVisualizationContent:
    """Tests to verify visualization content is meaningful."""
    
    def test_distribution_plots_created_for_all_features(self, eda_instance):
        """Test that distribution plots are created for all numerical features."""
        output_path = eda_instance.create_distribution_plots()
        
        assert Path(output_path).exists(), "Distribution plots file not created"
    
    def test_box_plots_created_for_all_features(self, eda_instance):
        """Test that box plots are created for all numerical features."""
        output_path = eda_instance.create_box_plots()
        
        assert Path(output_path).exists(), "Box plots file not created"
    
    def test_correlation_heatmap_created(self, eda_instance):
        """Test that correlation heatmap is created."""
        output_path = eda_instance.create_correlation_heatmap()
        
        assert Path(output_path).exists(), "Correlation heatmap not created"
    
    def test_crop_distribution_created(self, eda_instance):
        """Test that crop distribution plot is created."""
        output_path = eda_instance.create_crop_distribution_plot()
        
        assert Path(output_path).exists(), "Crop distribution plot not created"
    
    def test_crop_wise_features_created(self, eda_instance):
        """Test that crop-wise feature plots are created."""
        output_path = eda_instance.create_crop_wise_feature_plots()
        
        assert Path(output_path).exists(), "Crop-wise feature plots not created"
    
    def test_pair_plot_created(self, eda_instance):
        """Test that pair plot is created."""
        output_path = eda_instance.create_pair_plot()
        
        assert Path(output_path).exists(), "Pair plot not created"
    
    def test_scatter_plots_created(self, eda_instance):
        """Test that scatter plots are created."""
        output_path = eda_instance.create_scatter_plots()
        
        assert Path(output_path).exists(), "Scatter plots not created"


class TestDataIntegrity:
    """Tests to ensure EDA doesn't modify original data."""
    
    def test_original_data_not_modified(self, sample_data, tmp_path):
        """Test that EDA doesn't modify the original dataset."""
        original_data = sample_data.copy()
        
        eda = CropEDA(sample_data, output_dir=str(tmp_path))
        eda.run_all_analyses()
        
        # Check data hasn't changed
        pd.testing.assert_frame_equal(sample_data, original_data)
    
    def test_eda_uses_data_copy(self, sample_data, tmp_path):
        """Test that EDA works with a copy of the data."""
        eda = CropEDA(sample_data, output_dir=str(tmp_path))
        
        # Modify EDA's data
        eda.data.iloc[0, 0] = -9999
        
        # Original should be unchanged
        assert sample_data.iloc[0, 0] != -9999, "Original data was modified"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_strong_correlations_handled(self, eda_instance):
        """Test that empty strong correlations are handled gracefully."""
        # Use very high threshold to get no correlations
        strong_corrs = eda_instance.identify_strong_correlations(threshold=1.5)
        
        assert isinstance(strong_corrs, list), "Must return a list even if empty"
        assert len(strong_corrs) == 0, "Should find no correlations with threshold > 1"
    
    def test_insights_generated_even_with_no_correlations(self, eda_instance):
        """Test that insights are generated even with no strong correlations."""
        # Clear any existing correlations
        eda_instance.insights['correlations'] = {'strong_correlations': []}
        
        insights = eda_instance.generate_insights_report()
        
        assert 'key_findings' in insights, "Key findings must be generated"
