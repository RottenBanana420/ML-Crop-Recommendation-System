# Crop Recommendation System

A machine learning project for recommending optimal crops based on soil and environmental conditions.

## Overview

This project uses soil nutrient levels (N, P, K) and environmental factors (temperature, humidity, pH, rainfall) to recommend the most suitable crop from 22 different crop types.

## Dataset

The dataset contains 2,202 samples with the following features:
- **N**: Nitrogen content in soil
- **P**: Phosphorus content in soil
- **K**: Potassium content in soil
- **temperature**: Temperature in Celsius
- **humidity**: Relative humidity in percentage
- **ph**: pH value of the soil
- **rainfall**: Rainfall in mm

**Target Crops (22 types)**: apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

## Project Structure

```
ML-Crop-Recommendation-System/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Processed datasets
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # Data loading and validation
│   ├── features/               # Feature engineering (future)
│   ├── models/                 # ML models (future)
│   └── utils/                  # Utility functions
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   └── test_data_loader.py     # Comprehensive tests
├── notebooks/                  # Jupyter notebooks
├── models/                     # Saved model files
├── logs/                       # Log files
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

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

### Loading Data

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

# Get feature names
features = loader.get_feature_names()
print(f"Features: {features}")

# Get class distribution
distribution = loader.get_class_distribution()
print(f"Class distribution: {distribution}")
```

### Data Validation

The `CropDataLoader` automatically validates:
- ✅ All required columns are present
- ✅ No missing values
- ✅ No duplicate rows
- ✅ Correct data types
- ✅ Value ranges (e.g., humidity 0-100%, pH 0-14)
- ✅ Valid crop labels

## Running Tests

The project includes comprehensive tests to ensure data integrity.

### Run all tests
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run specific test class
```bash
pytest tests/test_data_loader.py::TestDataLoading -v
```

### Current Test Coverage
- **54 tests** covering all aspects of data loading
- **77% code coverage** on data loader module
- Tests validate: data loading, columns, data types, missing values, duplicates, value ranges, labels, and edge cases

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

### Adding New Features

1. Create feature engineering functions in `src/features/`
2. Add corresponding tests in `tests/`
3. Run tests to ensure everything works

### Training Models

1. Create model classes in `src/models/`
2. Train models using the loaded data
3. Save models to `models/` directory

### Creating Notebooks

1. Add exploratory notebooks to `notebooks/`
2. Use the data loader for consistent data access

## Environment

- **Python**: 3.13.7
- **Virtual Environment**: crop-recommendation-env (pyenv-virtualenv)
- **Platform**: macOS (compatible with Linux and Windows)

## Testing Philosophy

This project follows a **test-first** approach:
- Tests are designed to **FAIL** if data has any integrity issues
- Tests are **aggressive** and catch edge cases
- If tests fail, we fix the **codebase**, not the tests
- This ensures robust, production-ready code

## License

See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Future Enhancements

- [ ] Feature engineering pipeline
- [ ] Multiple ML model implementations
- [ ] Model comparison and selection
- [ ] Hyperparameter tuning
- [ ] REST API for predictions
- [ ] Web interface
- [ ] Model deployment
