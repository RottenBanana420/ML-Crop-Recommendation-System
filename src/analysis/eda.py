"""
Comprehensive Exploratory Data Analysis (EDA) Module for Crop Recommendation System.

This module provides functions for analyzing the crop recommendation dataset following
agricultural data analysis best practices. It includes data quality checks, statistical
analysis, correlation analysis, visualization generation, and insights documentation.

Author: Crop Recommendation System
Date: 2025-11-23
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CropEDA:
    """
    Comprehensive Exploratory Data Analysis for Crop Recommendation Dataset.
    
    This class provides methods for data quality checks, statistical analysis,
    correlation analysis, visualization generation, and insights documentation.
    """
    
    def __init__(self, data: pd.DataFrame, output_dir: str = "data"):
        """
        Initialize the EDA analyzer.
        
        Args:
            data: DataFrame containing the crop recommendation dataset
            output_dir: Base directory for outputs (visualizations and insights)
        """
        self.data = data.copy()
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.insights_dir = self.output_dir / "insights"
        
        # Create output directories
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature definitions
        self.numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.target_column = 'label'
        
        # Storage for analysis results
        self.insights = {
            'data_quality': {},
            'statistics': {},
            'correlations': {},
            'distributions': {},
            'crop_patterns': {},
            'outliers': {}
        }
    
    # ==================== DATA QUALITY CHECKS ====================
    
    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks.
        
        Returns:
            Dictionary containing data quality metrics
        """
        quality_report = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicate_rows': int(self.data.duplicated().sum()),
            'data_types': self.data.dtypes.astype(str).to_dict()
        }
        
        self.insights['data_quality'] = quality_report
        return quality_report
    
    def detect_outliers_iqr(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Returns:
            Dictionary with outlier information for each numerical feature
        """
        outliers_info = {}
        
        for feature in self.numerical_features:
            Q1 = self.data[feature].quantile(0.25)
            Q3 = self.data[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[
                (self.data[feature] < lower_bound) | 
                (self.data[feature] > upper_bound)
            ]
            
            outliers_info[feature] = {
                'count': len(outliers),
                'percentage': round(len(outliers) / len(self.data) * 100, 2),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_values': outliers[feature].tolist()[:10]  # First 10 for brevity
            }
        
        self.insights['outliers']['iqr_method'] = outliers_info
        return outliers_info
    
    def detect_outliers_zscore(self, threshold: float = 3.0) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using Z-score method.
        
        Args:
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Dictionary with outlier information for each numerical feature
        """
        outliers_info = {}
        
        for feature in self.numerical_features:
            z_scores = np.abs(stats.zscore(self.data[feature]))
            outliers = self.data[z_scores > threshold]
            
            outliers_info[feature] = {
                'count': len(outliers),
                'percentage': round(len(outliers) / len(self.data) * 100, 2),
                'threshold': threshold,
                'max_zscore': float(z_scores.max()),
                'outlier_values': outliers[feature].tolist()[:10]  # First 10 for brevity
            }
        
        self.insights['outliers']['zscore_method'] = outliers_info
        return outliers_info
    
    # ==================== STATISTICAL ANALYSIS ====================
    
    def generate_descriptive_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive descriptive statistics for all numerical features.
        
        Returns:
            DataFrame with descriptive statistics
        """
        stats_df = self.data[self.numerical_features].describe()
        
        # Add additional statistics
        stats_df.loc['skewness'] = self.data[self.numerical_features].skew()
        stats_df.loc['kurtosis'] = self.data[self.numerical_features].kurtosis()
        
        # Store in insights
        self.insights['statistics']['descriptive'] = stats_df.to_dict()
        
        return stats_df
    
    def calculate_crop_wise_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate statistics grouped by crop type for each feature.
        
        Returns:
            Dictionary mapping features to crop-wise statistics DataFrames
        """
        crop_stats = {}
        
        for feature in self.numerical_features:
            stats_df = self.data.groupby(self.target_column)[feature].agg([
                'count', 'mean', 'std', 'min', 'max', 
                ('q25', lambda x: x.quantile(0.25)),
                ('median', 'median'),
                ('q75', lambda x: x.quantile(0.75))
            ]).round(2)
            
            crop_stats[feature] = stats_df
        
        # Store in insights (convert to dict for JSON serialization)
        self.insights['crop_patterns']['statistics'] = {
            feature: df.to_dict() for feature, df in crop_stats.items()
        }
        
        return crop_stats
    
    def analyze_feature_distributions(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze distribution characteristics (skewness, kurtosis) for each feature.
        
        Returns:
            Dictionary with distribution metrics for each feature
        """
        distribution_info = {}
        
        for feature in self.numerical_features:
            distribution_info[feature] = {
                'skewness': float(self.data[feature].skew()),
                'kurtosis': float(self.data[feature].kurtosis()),
                'is_normal': bool(stats.normaltest(self.data[feature])[1] > 0.05)
            }
        
        self.insights['distributions'] = distribution_info
        return distribution_info
    
    # ==================== CORRELATION ANALYSIS ====================
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for all numerical features.
        
        Returns:
            Correlation matrix DataFrame
        """
        corr_matrix = self.data[self.numerical_features].corr()
        self.insights['correlations']['matrix'] = corr_matrix.to_dict()
        return corr_matrix
    
    def identify_strong_correlations(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Identify strong correlations between features.
        
        Args:
            threshold: Correlation threshold (default: 0.5)
            
        Returns:
            List of tuples (feature1, feature2, correlation_value)
        """
        corr_matrix = self.data[self.numerical_features].corr()
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corrs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr_value)
                    ))
        
        # Sort by absolute correlation value
        strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        self.insights['correlations']['strong_correlations'] = [
            {'feature1': f1, 'feature2': f2, 'correlation': corr}
            for f1, f2, corr in strong_corrs
        ]
        
        return strong_corrs
    
    def analyze_feature_crop_relationships(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze how features relate to different crop types using ANOVA.
        
        Returns:
            Dictionary with F-statistic and p-value for each feature
        """
        relationships = {}
        
        for feature in self.numerical_features:
            # Group feature values by crop
            groups = [group[feature].values for name, group in self.data.groupby(self.target_column)]
            
            # Perform one-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            relationships[feature] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        
        self.insights['crop_patterns']['feature_relationships'] = relationships
        return relationships
    
    # ==================== VISUALIZATION GENERATION ====================
    
    def create_distribution_plots(self) -> str:
        """
        Create histograms with KDE for all numerical features.
        
        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.numerical_features):
            ax = axes[idx]
            
            # Histogram with KDE
            ax.hist(self.data[feature], bins=30, alpha=0.6, color='skyblue', 
                   edgecolor='black', density=True, label='Histogram')
            
            # KDE curve
            self.data[feature].plot(kind='kde', ax=ax, color='darkblue', 
                                   linewidth=2, label='KDE')
            
            ax.set_xlabel(feature, fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots
        for idx in range(len(self.numerical_features), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_path = self.viz_dir / 'distribution_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_box_plots(self) -> str:
        """
        Create box plots for outlier detection and distribution comparison.
        
        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.numerical_features):
            ax = axes[idx]
            
            # Box plot
            bp = ax.boxplot(self.data[feature], vert=True, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='blue', linewidth=1.5),
                           capprops=dict(color='blue', linewidth=1.5))
            
            ax.set_ylabel(feature, fontsize=11, fontweight='bold')
            ax.set_title(f'Box Plot: {feature}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove extra subplots
        for idx in range(len(self.numerical_features), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_path = self.viz_dir / 'box_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_correlation_heatmap(self) -> str:
        """
        Create annotated correlation heatmap.
        
        Returns:
            Path to saved visualization
        """
        corr_matrix = self.data[self.numerical_features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Heatmap of Features', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        output_path = self.viz_dir / 'correlation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_crop_distribution_plot(self) -> str:
        """
        Create count plot showing crop label distribution.
        
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=(12, 6))
        
        crop_counts = self.data[self.target_column].value_counts().sort_values(ascending=False)
        
        ax = sns.barplot(x=crop_counts.index, y=crop_counts.values, palette='viridis')
        ax.set_xlabel('Crop Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Crop Labels', fontsize=14, fontweight='bold', pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(crop_counts.values):
            ax.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.viz_dir / 'crop_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store distribution in insights
        self.insights['crop_patterns']['distribution'] = crop_counts.to_dict()
        
        return str(output_path)
    
    def create_crop_wise_feature_plots(self) -> str:
        """
        Create violin plots showing feature distributions per crop.
        
        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.numerical_features):
            ax = axes[idx]
            
            # Violin plot
            sns.violinplot(data=self.data, y=self.target_column, x=feature, 
                          ax=ax, palette='Set2', orient='h')
            
            ax.set_xlabel(feature, fontsize=11, fontweight='bold')
            ax.set_ylabel('Crop Type', fontsize=11, fontweight='bold')
            ax.set_title(f'{feature} Distribution by Crop', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        # Remove extra subplots
        for idx in range(len(self.numerical_features), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_path = self.viz_dir / 'crop_wise_features.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_pair_plot(self) -> str:
        """
        Create pair plot for key feature relationships.
        
        Returns:
            Path to saved visualization
        """
        # Select a subset of features for clarity
        key_features = ['N', 'P', 'K', 'ph', self.target_column]
        
        # Sample data if too large (for performance)
        sample_data = self.data[key_features]
        if len(sample_data) > 1000:
            sample_data = sample_data.sample(n=1000, random_state=42)
        
        pair_plot = sns.pairplot(sample_data, hue=self.target_column, 
                                palette='tab10', diag_kind='kde', 
                                plot_kws={'alpha': 0.6, 's': 30},
                                height=2.5)
        
        pair_plot.fig.suptitle('Pair Plot: Key Features', y=1.02, 
                              fontsize=14, fontweight='bold')
        
        output_path = self.viz_dir / 'pair_plot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_scatter_plots(self) -> str:
        """
        Create scatter plots for strongly correlated features.
        
        Returns:
            Path to saved visualization
        """
        strong_corrs = self.identify_strong_correlations(threshold=0.3)
        
        if not strong_corrs:
            # Create a placeholder if no strong correlations
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No strong correlations found (threshold: 0.3)', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            # Create scatter plots for top correlations
            n_plots = min(6, len(strong_corrs))
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, (feat1, feat2, corr) in enumerate(strong_corrs[:n_plots]):
                ax = axes[idx]
                
                ax.scatter(self.data[feat1], self.data[feat2], 
                          alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
                
                # Add regression line
                z = np.polyfit(self.data[feat1], self.data[feat2], 1)
                p = np.poly1d(z)
                ax.plot(self.data[feat1].sort_values(), 
                       p(self.data[feat1].sort_values()), 
                       "r--", linewidth=2, label=f'r = {corr:.2f}')
                
                ax.set_xlabel(feat1, fontsize=11, fontweight='bold')
                ax.set_ylabel(feat2, fontsize=11, fontweight='bold')
                ax.set_title(f'{feat1} vs {feat2}', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Remove extra subplots
            for idx in range(n_plots, len(axes)):
                fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_path = self.viz_dir / 'scatter_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    # ==================== INSIGHTS GENERATION ====================
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights report from all analyses.
        
        Returns:
            Dictionary containing all insights
        """
        # Add metadata
        self.insights['metadata'] = {
            'total_samples': len(self.data),
            'total_features': len(self.numerical_features),
            'total_crops': int(self.data[self.target_column].nunique()),
            'crop_list': sorted(self.data[self.target_column].unique().tolist())
        }
        
        # Add key findings summary
        self.insights['key_findings'] = self._generate_key_findings()
        
        # Save to JSON
        output_path = self.insights_dir / 'eda_insights.json'
        with open(output_path, 'w') as f:
            json.dump(self.insights, f, indent=2)
        
        return self.insights
    
    def _generate_key_findings(self) -> Dict[str, Any]:
        """
        Generate key findings summary from analysis results.
        
        Returns:
            Dictionary with key findings
        """
        findings = {}
        
        # Data quality findings
        if self.insights.get('data_quality'):
            dq = self.insights['data_quality']
            findings['data_quality'] = {
                'has_missing_values': any(v > 0 for v in dq['missing_values'].values()),
                'has_duplicates': dq['duplicate_rows'] > 0,
                'quality_score': 'Excellent' if dq['duplicate_rows'] == 0 and 
                                not any(v > 0 for v in dq['missing_values'].values()) else 'Good'
            }
        
        # Distribution findings
        if self.insights.get('distributions'):
            skewed_features = [
                feat for feat, info in self.insights['distributions'].items()
                if abs(info['skewness']) > 1
            ]
            findings['distributions'] = {
                'highly_skewed_features': skewed_features,
                'normal_features': [
                    feat for feat, info in self.insights['distributions'].items()
                    if info['is_normal']
                ]
            }
        
        # Correlation findings
        if self.insights.get('correlations', {}).get('strong_correlations'):
            findings['correlations'] = {
                'count': len(self.insights['correlations']['strong_correlations']),
                'strongest': self.insights['correlations']['strong_correlations'][0] 
                           if self.insights['correlations']['strong_correlations'] else None
            }
        
        # Outlier findings
        if self.insights.get('outliers', {}).get('iqr_method'):
            outlier_counts = {
                feat: info['count'] 
                for feat, info in self.insights['outliers']['iqr_method'].items()
            }
            findings['outliers'] = {
                'features_with_outliers': [feat for feat, count in outlier_counts.items() if count > 0],
                'total_outliers_by_feature': outlier_counts
            }
        
        return findings
    
    # ==================== MAIN EXECUTION ====================
    
    def run_all_analyses(self) -> Dict[str, str]:
        """
        Run all EDA analyses and generate visualizations.
        
        Returns:
            Dictionary mapping analysis types to output paths
        """
        print("Starting Comprehensive EDA...")
        print("=" * 60)
        
        outputs = {}
        
        # Data Quality Checks
        print("\n1. Performing Data Quality Checks...")
        self.check_data_quality()
        self.detect_outliers_iqr()
        self.detect_outliers_zscore()
        print("   ✓ Data quality checks complete")
        
        # Statistical Analysis
        print("\n2. Generating Statistical Summaries...")
        self.generate_descriptive_statistics()
        self.calculate_crop_wise_statistics()
        self.analyze_feature_distributions()
        print("   ✓ Statistical analysis complete")
        
        # Correlation Analysis
        print("\n3. Performing Correlation Analysis...")
        self.calculate_correlation_matrix()
        self.identify_strong_correlations()
        self.analyze_feature_crop_relationships()
        print("   ✓ Correlation analysis complete")
        
        # Visualizations
        print("\n4. Generating Visualizations...")
        outputs['distribution_plots'] = self.create_distribution_plots()
        print("   ✓ Distribution plots created")
        
        outputs['box_plots'] = self.create_box_plots()
        print("   ✓ Box plots created")
        
        outputs['correlation_heatmap'] = self.create_correlation_heatmap()
        print("   ✓ Correlation heatmap created")
        
        outputs['crop_distribution'] = self.create_crop_distribution_plot()
        print("   ✓ Crop distribution plot created")
        
        outputs['crop_wise_features'] = self.create_crop_wise_feature_plots()
        print("   ✓ Crop-wise feature plots created")
        
        outputs['pair_plot'] = self.create_pair_plot()
        print("   ✓ Pair plot created")
        
        outputs['scatter_plots'] = self.create_scatter_plots()
        print("   ✓ Scatter plots created")
        
        # Generate Insights Report
        print("\n5. Generating Insights Report...")
        self.generate_insights_report()
        outputs['insights_report'] = str(self.insights_dir / 'eda_insights.json')
        print("   ✓ Insights report generated")
        
        print("\n" + "=" * 60)
        print("EDA Complete! All outputs saved to:")
        print(f"  - Visualizations: {self.viz_dir}")
        print(f"  - Insights: {self.insights_dir}")
        print("=" * 60)
        
        return outputs


# ==================== CONVENIENCE FUNCTION ====================

def run_comprehensive_eda(data_path: Optional[str] = None, output_dir: str = "data") -> Dict[str, str]:
    """
    Run comprehensive EDA on the crop recommendation dataset.
    
    Args:
        data_path: Path to the CSV file. If None, uses default path.
        output_dir: Base directory for outputs
        
    Returns:
        Dictionary mapping analysis types to output file paths
    """
    # Load data
    if data_path is None:
        from src.data.loader import CropDataLoader
        loader = CropDataLoader()
        df = loader.load_data()
    else:
        df = pd.read_csv(data_path)
    
    # Run EDA
    eda = CropEDA(df, output_dir=output_dir)
    outputs = eda.run_all_analyses()
    
    return outputs


if __name__ == "__main__":
    # Run EDA when script is executed directly
    run_comprehensive_eda()
