# Crop Recommendation System

A comprehensive machine learning project for recommending optimal crops based on soil and environmental conditions using advanced ensemble learning techniques.

## Overview

This project analyzes soil nutrient levels (N, P, K) and environmental factors (temperature, humidity, pH, rainfall) to recommend the most suitable crop from 22 different crop types. The system implements and compares **Random Forest** and **XGBoost** classifiers, achieving **99.3% test accuracy** with the production-selected Random Forest model. The project includes systematic model comparison with multi-criteria decision analysis to select the optimal model for deployment.

## Dataset

The dataset contains 2,200 samples with the following features:
- **N**: Nitrogen content in soil (0-140)
- **P**: Phosphorus content in soil (5-145)
- **K**: Potassium content in soil (5-205)
- **temperature**: Temperature in Celsius (8.8-43.7°C)
- **humidity**: Relative humidity in percentage (14.3-99.9%)
- **ph**: pH value of the soil (3.5-9.9)
- **rainfall**: Rainfall in mm (20.2-298.6mm)

**Target Crops (22 types)**: apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

## Project Structure

```
ML-Crop-Recommendation-System/
├── data/
│   ├── raw/                    # Original dataset (Crop_recommendation.csv)
│   ├── processed/              # Processed train/test datasets
│   ├── insights/               # EDA insights (JSON)
│   └── visualizations/         # EDA and model performance plots
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # Data loading and validation
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── eda.py              # Exploratory Data Analysis
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessing.py    # Feature engineering & preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── random_forest_model.py           # Random Forest classifier
│   │   ├── xgboost_model.py                 # XGBoost classifier
│   │   ├── model_comparison.py              # Model comparison framework
│   │   ├── comparison_visualizations.py     # Comparison visualizations
│   │   ├── comparison_report.py             # HTML report generation
│   │   ├── visualize_model_performance.py   # Model visualization
│   │   └── xgboost_visualizations.py        # XGBoost-specific visualizations
│   └── utils/                  # Utility functions
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Pytest fixtures
│   ├── test_data_loader.py              # Data loader tests
│   ├── test_eda.py                      # EDA tests
│   ├── test_preprocessing.py            # Preprocessing tests
│   ├── test_random_forest_model.py      # Random Forest tests
│   ├── test_xgboost_model.py            # XGBoost tests
│   └── test_model_comparison.py         # Model comparison tests
├── models/                     # Saved model files and metadata
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── production_model.pkl             # Selected production model
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── model_metadata.json
│   ├── xgboost_metadata.json
│   ├── production_model_metadata.json
│   ├── feature_importance.json
│   ├── xgboost_feature_importance.json
│   ├── full_metrics.json
│   ├── xgboost_full_metrics.json
│   ├── model_comparison.json            # Detailed comparison metrics
│   ├── model_comparison.md              # Human-readable comparison report
│   ├── comparison_report.html           # Interactive HTML report
│   └── comparison_visualizations/       # Comparison charts
│       ├── radar_chart.png
│       ├── per_class_heatmap.png
│       ├── confusion_matrices.png
│       ├── inference_time_distribution.png
│       ├── feature_importance_correlation.png
│       └── performance_efficiency_scatter.png
├── notebooks/                  # Jupyter notebooks
├── logs/                       # Log files
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

## Key Features

### 🔍 Comprehensive Data Analysis
- **Exploratory Data Analysis (EDA)**: Complete statistical analysis with 7+ visualizations
- **Data Quality Checks**: Automated validation for missing values, duplicates, and outliers
- **Statistical Insights**: Distribution analysis, correlation matrices, and ANOVA tests
- **Crop Pattern Analysis**: Detailed feature distributions for each of 22 crop types

### 🛠️ Advanced Feature Engineering
- **Nutrient Ratios**: N:P, N:K, P:K ratios for nutrient balance analysis
- **Environmental Interactions**: Temperature × humidity, rainfall interactions
- **pH Interactions**: pH-nutrient availability modeling
- **Composite Indicators**: Growing condition index and moisture stress index
- **Total Features**: 22 engineered features from 7 original features

### 🤖 Machine Learning Models

#### Random Forest Classifier
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold cross-validation
- **Performance**: 99.3% test accuracy, 99.2% cross-validation score
- **Optimized Parameters**:
  - n_estimators: 200
  - max_depth: 40
  - min_samples_split: 10
  - criterion: entropy
- **Feature Importance**: Detailed analysis of feature contributions

#### XGBoost Classifier
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold cross-validation
- **Performance**: 99.1% test accuracy, 99.0% cross-validation score
- **Optimized Parameters**:
  - n_estimators: 200
  - max_depth: 6
  - learning_rate: 0.1
  - subsample: 0.8
- **Feature Importance**: Gain-based feature importance analysis

### 🔬 Model Comparison Framework
- **Multi-Criteria Decision Analysis**: Systematic comparison across accuracy, speed, memory, and stability
- **Statistical Significance Testing**: McNemar's test for accuracy differences
- **Production Readiness Criteria**:
  - Minimum 97.5% test accuracy
  - All per-class F1-scores ≥ 95%
  - Inference latency < 100ms per 1000 samples
  - Model size < 50MB
- **Comprehensive Metrics**:
  - Per-class precision, recall, F1-score
  - Matthews Correlation Coefficient (MCC)
  - Cohen's Kappa
  - Inference latency percentiles (P50, P95, P99)
  - Memory footprint analysis
- **Automated Model Selection**: Intelligent selection based on balanced performance characteristics
- **Interactive HTML Report**: Detailed comparison with visualizations

### 📊 Visualizations

#### Data Analysis Visualizations
- Distribution plots for all features
- Box plots for outlier detection
- Correlation heatmap
- Crop distribution analysis
- Crop-wise feature violin plots
- Pair plots for feature relationships

#### Model Performance Visualizations
- Confusion matrices for both models
- ROC curves and precision-recall curves
- Feature importance charts

#### Model Comparison Visualizations
- **Radar Chart**: Multi-metric performance comparison
- **Per-Class Heatmap**: F1 scores across all 22 crop types
- **Confusion Matrix Comparison**: Side-by-side confusion matrices
- **Inference Time Distribution**: Box plots of prediction latencies
- **Feature Importance Correlation**: Correlation between model feature rankings
- **Performance-Efficiency Scatter**: Accuracy vs speed trade-off analysis

### ✅ Robust Testing
- **200+ comprehensive tests** across all modules
- Test-driven development approach
- High code coverage across all modules
- Edge case handling and validation
- Model comparison validation tests
- Production readiness criteria tests

### 💾 Model Persistence
- Trained model saved with joblib
- Scaler and label encoder persistence
- Comprehensive metadata (hyperparameters, metrics, feature names)
- Feature importance rankings

## Setup Instructions

### Prerequisites

- Python 3.12 or higher
- pyenv and pyenv-virtualenv (recommended)

### Installation

1. **Clone the repository**
   ```bash
   cd ML-Crop-Recommendation-System
   ```

2. **Create and activate virtual environment**
   
   Using pyenv-virtualenv (recommended):
   ```bash
   pyenv virtualenv 3.13.7 crop-recommendation-env
   pyenv local crop-recommendation-env
   ```
   
   Or using venv:
   ```bash
   python -m venv crop-recommendation-env
   source crop-recommendation-env/bin/activate  # On Windows: crop-recommendation-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Loading Data

```python
from src.data.loader import CropDataLoader

# Initialize the loader
loader = CropDataLoader()

# Load and validate data
df = loader.load_data()

# Get dataset information
info = loader.get_dataset_info()
print(f"Dataset shape: {info['shape']}")
print(f"Number of crops: {len(info['unique_labels'])}")
```

**Data Validation**: The `CropDataLoader` automatically validates:
- ✅ All required columns are present
- ✅ No missing values
- ✅ No duplicate rows
- ✅ Correct data types
- ✅ Value ranges (e.g., humidity 0-100%, pH 0-14)
- ✅ Valid crop labels

### 2. Exploratory Data Analysis (EDA)

```python
from src.analysis.eda import run_comprehensive_eda

# Run complete EDA pipeline
results = run_comprehensive_eda(
    data_path='data/raw/Crop_recommendation.csv',
    output_dir='data'
)

# EDA generates:
# - Distribution plots for all features
# - Box plots for outlier detection
# - Correlation heatmap
# - Crop distribution analysis
# - Crop-wise feature analysis
# - Pair plots
# - Statistical insights (JSON)
```

**Key Findings from EDA**:
- Strong correlation (0.74) between Phosphorus and Potassium
- All features show significant relationships with crop types (ANOVA p < 0.001)
- Balanced dataset: 100 samples per crop
- No missing values or duplicates

### 3. Data Preprocessing & Feature Engineering

```python
from src.features.preprocessing import CropPreprocessor

# Initialize preprocessor
preprocessor = CropPreprocessor(
    test_size=0.2,
    random_state=42,
    scaling_method='standard'
)

# Fit and transform data
X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

# Engineered features include:
# - Nutrient ratios (N:P, N:K, P:K)
# - Total nutrients and nutrient balance
# - Environmental interactions (temp × humidity, etc.)
# - pH-nutrient interactions
# - Composite agricultural indicators
```

**Preprocessing Pipeline**:
1. Train-test split (80/20) with stratification
2. Feature engineering (15 new features)
3. StandardScaler normalization (fit on train only)
4. Label encoding for crop names

### 4. Model Training & Prediction

#### Training Random Forest Model

```python
from src.models.random_forest_model import CropRecommendationModel

# Initialize model
model = CropRecommendationModel(random_state=42)

# Load preprocessed data
X_train, X_test, y_train, y_test = model.load_data('data/processed')

# Tune hyperparameters (optional)
best_params = model.tune_hyperparameters(X_train, y_train)

# Train model
model.train(X_train, y_train, use_best_params=True)

# Evaluate model
metrics = model.evaluate(X_train, y_train, X_test, y_test)
print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")

# Save model
model.save_model('models/')
```

#### Training XGBoost Model

```python
from src.models.xgboost_model import XGBoostCropModel

# Initialize model
xgb_model = XGBoostCropModel(random_state=42)

# Load preprocessed data
X_train, X_test, y_train, y_test = xgb_model.load_data('data/processed')

# Tune hyperparameters (optional)
best_params = xgb_model.tune_hyperparameters(X_train, y_train)

# Train model
xgb_model.train(X_train, y_train, use_best_params=True)

# Evaluate model
metrics = xgb_model.evaluate(X_train, y_train, X_test, y_test)
print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")

# Save model
xgb_model.save_model('models/', prefix='xgboost')
```

#### Comparing Models

```python
from src.models.model_comparison import ModelComparison

# Initialize comparison framework
comparison = ModelComparison(
    models_dir='models/',
    output_dir='models/'
)

# Load both models
comparison.load_models()

# Run comprehensive comparison
results = comparison.compare_models(X_test, y_test)

# Generate visualizations
comparison.create_comparison_visualizations()

# Generate HTML report
comparison.generate_html_report()

# Get selected production model
selected_model = comparison.select_production_model()
print(f"Selected Model: {selected_model}")

# Access comparison results
print(f"Accuracy Difference: {results['accuracy_difference']:.4f}")
print(f"Statistical Significance: p={results['mcnemar_p_value']:.4f}")
print(f"Production Ready: {results['production_ready']}")
```

### 5. Making Predictions

```python
import joblib
import numpy as np

# Load production model (automatically selected best model)
model = joblib.load('models/production_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Prepare new data (with engineered features)
new_data = np.array([[...]])  # Shape: (1, 22) - includes engineered features
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
crop_name = label_encoder.inverse_transform(prediction)
print(f"Recommended crop: {crop_name[0]}")

# Get prediction probabilities
probabilities = model.predict_proba(new_data_scaled)
confidence = probabilities.max()
print(f"Confidence: {confidence:.2%}")
```

## Running Tests

The project includes comprehensive tests for all components with high code coverage.

### Run all tests
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run specific test modules
```bash
# Data loader tests
pytest tests/test_data_loader.py -v

# EDA tests
pytest tests/test_eda.py -v

# Preprocessing tests
pytest tests/test_preprocessing.py -v

# Random Forest model tests
pytest tests/test_random_forest_model.py -v

# XGBoost model tests
pytest tests/test_xgboost_model.py -v

# Model comparison tests
pytest tests/test_model_comparison.py -v
```

### Test Coverage Summary
- **Total Tests**: 200+ comprehensive tests
- **Test Modules**: 6 (data loader, EDA, preprocessing, Random Forest, XGBoost, model comparison)
- **Coverage**: High coverage across all modules
- **Test Categories**:
  - Data loading and validation (54 tests)
  - EDA completeness and accuracy (40+ tests)
  - Preprocessing and feature engineering (30+ tests)
  - Random Forest training and evaluation (20+ tests)
  - XGBoost training and evaluation (25+ tests)
  - Model comparison and selection (30+ tests)

## Dependencies

- **pandas** (2.3.3): Data manipulation
- **numpy** (2.3.5): Numerical operations
- **scikit-learn** (1.7.2): Machine learning algorithms
- **xgboost** (3.1.2): Gradient boosting
- **seaborn** (0.13.2): Statistical visualization
- **matplotlib** (3.10.7): Plotting
- **flask** (3.1.2): Web framework (for future API)
- **joblib** (1.5.2): Model serialization
- **pytest** (9.0.1): Testing framework
- **pytest-cov** (7.0.0): Coverage reporting

## Development

### Project Workflow

This project follows a systematic machine learning pipeline:

1. **Data Loading & Validation** (`src/data/loader.py`)
   - Load raw data with comprehensive validation
   - Ensure data quality and integrity

2. **Exploratory Data Analysis** (`src/analysis/eda.py`)
   - Understand data distributions and patterns
   - Identify correlations and relationships
   - Generate insights and visualizations

3. **Feature Engineering** (`src/features/preprocessing.py`)
   - Engineer domain-specific features
   - Apply scaling and encoding
   - Split data with stratification

4. **Model Training** (`src/models/`)
   - **Random Forest**: Hyperparameter tuning with cross-validation
   - **XGBoost**: Hyperparameter tuning with cross-validation
   - Train optimized models
   - Evaluate performance metrics

5. **Model Comparison** (`src/models/model_comparison.py`)
   - Systematic comparison across multiple criteria
   - Statistical significance testing
   - Production readiness validation
   - Automated model selection

6. **Model Persistence** (`models/`)
   - Save trained models and preprocessors
   - Store metadata and feature importance
   - Save production-selected model
   - Enable reproducible predictions

### Adding New Features

1. Modify feature engineering in `src/features/preprocessing.py`
2. Add corresponding tests in `tests/test_preprocessing.py`
3. Re-run preprocessing and model training
4. Validate improvements in model performance

### Training New Models

1. Create model class in `src/models/`
2. Implement training, evaluation, and persistence methods
3. Add comprehensive tests in `tests/`
4. Compare performance with existing models

## Testing Philosophy

This project follows a **test-first** approach:
- Tests are designed to **FAIL** if implementation has issues
- Tests are **comprehensive** and catch edge cases
- If tests fail, we fix the **codebase**, not the tests
- This ensures robust, production-ready code
- All new features must include corresponding tests

## Model Performance

### Production Model: Random Forest ✓

**Selected based on**: Multi-criteria decision analysis with balanced consideration of accuracy, speed, size, and stability.

#### Performance Metrics
- **Test Accuracy**: 99.32%
- **Training Accuracy**: 100.00%
- **Cross-Validation Score**: 99.20% (±0.42%)
- **Precision (Weighted)**: 99.35%
- **Recall (Weighted)**: 99.32%
- **F1-Score (Weighted)**: 99.32%
- **Matthews Correlation Coefficient**: 99.29%
- **Cohen's Kappa**: 99.29%

#### Efficiency Metrics
- **Training Time**: 0.82s
- **Prediction Time**: 8.56ms (1000 samples)
- **P50 Latency**: 8.49ms
- **P95 Latency**: 9.04ms
- **P99 Latency**: 9.20ms
- **Model Size**: 4.76 MB

#### Top Feature Importance
1. Humidity (11.35%)
2. Potassium - K (10.53%)
3. Rainfall (log) (7.43%)
4. Moisture Index (6.96%)
5. Rainfall × Humidity Interaction (6.90%)

### XGBoost Model (Alternative)

#### Performance Metrics
- **Test Accuracy**: 99.09%
- **Training Accuracy**: 100.00%
- **Cross-Validation Score**: 99.03% (±0.53%)
- **Precision (Weighted)**: 99.15%
- **Recall (Weighted)**: 99.09%
- **F1-Score (Weighted)**: 99.09%
- **Matthews Correlation Coefficient**: 99.05%
- **Cohen's Kappa**: 99.05%

#### Efficiency Metrics
- **Training Time**: 4.33s
- **Prediction Time**: 3.18ms (1000 samples)
- **P50 Latency**: 3.10ms
- **P95 Latency**: 3.60ms
- **P99 Latency**: 4.07ms
- **Model Size**: 5.07 MB

#### Top Feature Importance
1. Rainfall (log) (12.10%)
2. Potassium - K (7.74%)
3. N:K Ratio (7.58%)
4. Rainfall (6.92%)
5. Moisture Index (6.67%)

### Model Comparison Summary

| Metric | Random Forest | XGBoost | Winner |
|--------|---------------|---------|--------|
| **Test Accuracy** | 99.32% | 99.09% | Random Forest |
| **Training Speed** | 0.82s | 4.33s | Random Forest (5.3x faster) |
| **Prediction Speed** | 8.56ms | 3.18ms | XGBoost (2.7x faster) |
| **Model Size** | 4.76 MB | 5.07 MB | Random Forest |
| **Cross-Val Stability** | ±0.42% | ±0.53% | Random Forest |
| **Statistical Significance** | - | p=1.00 | Not significant |

**Key Findings**:
- Random Forest has slightly higher accuracy (0.23% difference)
- XGBoost is faster at inference but slower at training
- Accuracy difference is not statistically significant (McNemar's test)
- Both models meet all production readiness criteria
- Random Forest selected for balanced performance characteristics

### Production Readiness Criteria ✓

All criteria met for production deployment:
- ✅ Test Accuracy ≥ 97.5%: **99.32%**
- ✅ All Per-Class F1-Scores ≥ 95%: **Minimum 97.44%**
- ✅ Inference Latency < 100ms/1000 samples: **8.56ms**
- ✅ Model Size < 50MB: **4.76 MB**

## Future Enhancements

### Completed ✅
- [x] Data loading and validation
- [x] Exploratory Data Analysis
- [x] Feature engineering pipeline
- [x] Random Forest model implementation
- [x] XGBoost model implementation
- [x] Hyperparameter tuning for both models
- [x] Model evaluation and visualization
- [x] Systematic model comparison framework
- [x] Production model selection with multi-criteria analysis
- [x] Comprehensive testing suite (200+ tests)
- [x] Model persistence and metadata storage
- [x] Interactive HTML comparison reports

### In Progress 🔄
- [ ] REST API for predictions (Flask/FastAPI)
- [ ] Web interface for user-friendly predictions

### Planned 🚀
- [ ] Additional ML models (LightGBM, Neural Networks)
- [ ] Model ensemble and stacking
- [ ] A/B testing framework for model deployment
- [ ] Real-time model performance monitoring
- [ ] Automated model retraining pipeline
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Mobile application integration
- [ ] Explainable AI (SHAP/LIME) for prediction interpretability
- [ ] Multi-language support for global deployment
- [ ] Integration with IoT sensors for real-time soil data

## Environment

- **Python**: 3.13.7
- **Virtual Environment**: crop-recommendation-env (pyenv-virtualenv)
- **Platform**: macOS (compatible with Linux and Windows)
- **Key Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn

## License

See LICENSE file for details.

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/ -v`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Contribution Guidelines
- Follow the existing code style and structure
- Write comprehensive tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR
- Keep commits focused and atomic

## Acknowledgments

- Dataset source: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Built with scikit-learn, pandas, and other amazing open-source libraries
