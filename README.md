# Crop Recommendation System

A comprehensive machine learning project for recommending optimal crops based on soil and environmental conditions using Random Forest classification.

## Overview

This project analyzes soil nutrient levels (N, P, K) and environmental factors (temperature, humidity, pH, rainfall) to recommend the most suitable crop from 22 different crop types. The system achieves **99.3% test accuracy** using a Random Forest classifier with optimized hyperparameters.

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
│   │   └── visualize_model_performance.py   # Model visualization
│   └── utils/                  # Utility functions
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Pytest fixtures
│   ├── test_data_loader.py              # Data loader tests
│   ├── test_eda.py                      # EDA tests
│   ├── test_preprocessing.py            # Preprocessing tests
│   └── test_random_forest_model.py      # Model tests
├── models/                     # Saved model files and metadata
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── model_metadata.json
│   ├── feature_importance.json
│   └── full_metrics.json
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

### 🤖 Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold cross-validation
- **Performance**: 99.3% test accuracy, 99.2% cross-validation score
- **Optimized Parameters**:
  - n_estimators: 200
  - max_depth: 40
  - min_samples_split: 10
  - criterion: entropy
- **Feature Importance**: Detailed analysis of feature contributions

### 📊 Visualizations
- Distribution plots for all features
- Box plots for outlier detection
- Correlation heatmap
- Crop distribution analysis
- Crop-wise feature violin plots
- Pair plots for feature relationships
- Model performance metrics (confusion matrix, ROC curves, etc.)

### ✅ Robust Testing
- 140+ comprehensive tests
- Test-driven development approach
- High code coverage across all modules
- Edge case handling and validation

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

### 5. Making Predictions

```python
import joblib
import numpy as np

# Load trained model
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Prepare new data (with engineered features)
new_data = np.array([[...]])  # Shape: (1, 22) - includes engineered features
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
crop_name = label_encoder.inverse_transform(prediction)
print(f"Recommended crop: {crop_name[0]}")
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

# Model tests
pytest tests/test_random_forest_model.py -v
```

### Test Coverage Summary
- **Total Tests**: 140+ comprehensive tests
- **Test Modules**: 4 (data loader, EDA, preprocessing, model)
- **Coverage**: High coverage across all modules
- **Test Categories**:
  - Data loading and validation (54 tests)
  - EDA completeness and accuracy (40+ tests)
  - Preprocessing and feature engineering (30+ tests)
  - Model training, evaluation, and persistence (20+ tests)

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

4. **Model Training** (`src/models/random_forest_model.py`)
   - Hyperparameter tuning with cross-validation
   - Train optimized Random Forest model
   - Evaluate performance metrics

5. **Model Persistence** (`models/`)
   - Save trained model and preprocessors
   - Store metadata and feature importance
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

### Current Results (Random Forest)
- **Test Accuracy**: 99.32%
- **Training Accuracy**: 100.00%
- **Cross-Validation Score**: 99.20% (±0.42%)
- **Precision (Macro)**: 99.35%
- **Recall (Macro)**: 99.32%
- **F1-Score (Macro)**: 99.32%

### Top Feature Importance
1. Rainfall
2. Potassium (K)
3. Phosphorus (P)
4. Humidity
5. Temperature

*Note: Feature importance includes both original and engineered features*

## Future Enhancements

### Completed ✅
- [x] Data loading and validation
- [x] Exploratory Data Analysis
- [x] Feature engineering pipeline
- [x] Random Forest model implementation
- [x] Hyperparameter tuning
- [x] Model evaluation and visualization
- [x] Comprehensive testing suite
- [x] Model persistence

### Planned 🚀
- [ ] Additional ML models (XGBoost, LightGBM, Neural Networks)
- [ ] Model ensemble and stacking
- [ ] REST API for predictions (Flask/FastAPI)
- [ ] Web interface for user-friendly predictions
- [ ] Docker containerization
- [ ] Model deployment (cloud platforms)
- [ ] Real-time prediction service
- [ ] Model monitoring and retraining pipeline
- [ ] Mobile application integration

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
