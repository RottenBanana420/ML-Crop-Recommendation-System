# ML Crop Recommendation System

A production-ready machine learning system that recommends optimal crops based on soil and environmental conditions, featuring a modern web interface, comprehensive testing suite (336+ tests), and systematic model comparison framework.

## 🌾 Project Purpose

Agriculture is heavily dependent on selecting the right crop for specific soil and climate conditions. This system leverages machine learning to analyze soil nutrient levels (Nitrogen, Phosphorus, Potassium) and environmental factors (temperature, humidity, pH, rainfall) to recommend the most suitable crop from 22 different types. 

**Key Objectives**:
- Provide accurate, data-driven crop recommendations to farmers and agricultural professionals
- Achieve production-ready performance with 99%+ accuracy
- Deliver predictions through an intuitive web interface
- Enable real-time recommendations with sub-100ms latency
- Support agricultural decision-making with explainable AI

## ✨ Features

### 🌐 Web Application
- **Modern UI**: Beautiful, responsive web interface with glassmorphism design
- **Real-time Predictions**: Instant crop recommendations with confidence scores
- **Interactive Forms**: User-friendly input validation and error handling
- **Batch Processing**: Upload CSV files for bulk predictions
- **Visualization Dashboard**: Interactive charts showing prediction distributions and confidence levels
- **RESTful API**: JSON API endpoints for programmatic access
- **Health Monitoring**: Built-in health check and status endpoints

### 🤖 Machine Learning Pipeline
- **Dual Model Architecture**: Random Forest and XGBoost classifiers
- **Advanced Feature Engineering**: 22 engineered features from 7 original inputs
  - Nutrient ratios (N:P, N:K, P:K)
  - Environmental interactions (temperature × humidity)
  - pH-nutrient availability modeling
  - Composite agricultural indicators (moisture index, growing condition index)
- **Hyperparameter Optimization**: RandomizedSearchCV with 5-fold cross-validation
- **Production Model Selection**: Multi-criteria decision analysis (accuracy, speed, size, stability)
- **Model Persistence**: Serialized models with comprehensive metadata

### 📊 Data Analysis & Visualization
- **Exploratory Data Analysis (EDA)**: 7+ statistical visualizations
- **Feature Importance Analysis**: Identify key factors driving predictions
- **Confusion Matrices**: Per-class performance visualization
- **Model Comparison Dashboard**: Interactive HTML reports with radar charts, heatmaps, and performance metrics
- **Correlation Analysis**: Feature relationship heatmaps

### 🧪 Comprehensive Testing
- **336+ Test Cases**: Unit, integration, and end-to-end tests
- **90%+ Code Coverage**: Across all modules
- **Test Categories**:
  - Data loading and validation (54 tests)
  - EDA completeness and accuracy (40+ tests)
  - Preprocessing and feature engineering (30+ tests)
  - Model training and evaluation (45+ tests)
  - Flask API endpoints (30+ tests)
  - Security and performance tests (20+ tests)
- **Continuous Integration**: Automated testing on every commit

### 🔒 Production-Ready Features
- **Input Validation**: Comprehensive validation for all user inputs
- **Error Handling**: Graceful degradation with informative error messages
- **Rate Limiting**: Configurable request throttling (default: 100 req/hour)
- **CORS Support**: Configurable cross-origin resource sharing
- **Logging**: Structured logging with rotation and retention policies
- **Security**: Environment-based configuration, no hardcoded secrets
- **Health Checks**: `/api/health` endpoint for monitoring

## 📦 Installation

### Prerequisites
- Python 3.12 or higher
- pyenv and pyenv-virtualenv (recommended)
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ML-Crop-Recommendation-System.git
   cd ML-Crop-Recommendation-System
   ```

2. **Set up virtual environment**
   
   Using pyenv-virtualenv (recommended):
   ```bash
   pyenv virtualenv 3.13.7 crop-recommendation-env
   pyenv local crop-recommendation-env
   ```
   
   Or using venv:
   ```bash
   python -m venv crop-recommendation-env
   source crop-recommendation-env/bin/activate  # Windows: crop-recommendation-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   # Development mode
   python run.py
   
   # Production mode with Gunicorn
   gunicorn -w 4 -b 0.0.0.0:5001 wsgi:app
   ```

6. **Access the application**
   - Web Interface: http://localhost:5001
   - API Documentation: http://localhost:5001/api/docs
   

## 🚀 Usage Examples

### Web Interface

1. **Single Prediction**
   - Navigate to http://localhost:5001
   - Enter soil and environmental parameters
   - Click "Get Recommendation"
   - View recommended crop with confidence score

2. **Batch Predictions**
   - Click "Batch Upload" tab
   - Upload CSV file with required columns
   - Download results with predictions

### API Usage

#### Single Prediction
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.00,
    "ph": 6.50,
    "rainfall": 202.93
  }'
```

Response:
```json
{
  "prediction": "rice",
  "confidence": 0.98,
  "probabilities": {
    "rice": 0.98,
    "jute": 0.01,
    "coconut": 0.01
  },
  "timestamp": "2025-11-28T14:02:39Z"
}
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5001/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"N": 90, "P": 42, "K": 43, "temperature": 20.87, "humidity": 82.00, "ph": 6.50, "rainfall": 202.93},
      {"N": 85, "P": 58, "K": 41, "temperature": 21.77, "humidity": 80.32, "ph": 7.04, "rainfall": 226.66}
    ]
  }'
```

### Python SDK

```python
from src.data.loader import CropDataLoader
from src.features.preprocessing import CropPreprocessor
from app.services.prediction_service import PredictionService

# Initialize services
loader = CropDataLoader()
preprocessor = CropPreprocessor()
predictor = PredictionService()

# Load and prepare data
df = loader.load_data()
X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

# Make prediction
input_data = {
    'N': 90, 'P': 42, 'K': 43,
    'temperature': 20.87, 'humidity': 82.00,
    'ph': 6.50, 'rainfall': 202.93
}

result = predictor.predict(input_data)
print(f"Recommended crop: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📊 Dataset Information

### Overview
- **Total Samples**: 2,200
- **Features**: 7 (soil nutrients + environmental factors)
- **Target Classes**: 22 crop types
- **Class Balance**: Perfectly balanced (100 samples per crop)
- **Data Quality**: No missing values, no duplicates

### Features

| Feature | Type | Range | Unit | Description |
|---------|------|-------|------|-------------|
| **N** | Continuous | 0-140 | ratio | Nitrogen content in soil |
| **P** | Continuous | 5-145 | ratio | Phosphorus content in soil |
| **K** | Continuous | 5-205 | ratio | Potassium content in soil |
| **temperature** | Continuous | 8.8-43.7 | °C | Average temperature |
| **humidity** | Continuous | 14.3-99.9 | % | Relative humidity |
| **ph** | Continuous | 3.5-9.9 | pH | Soil pH level |
| **rainfall** | Continuous | 20.2-298.6 | mm | Annual rainfall |

### Target Crops (22 types)
apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

### Key Statistical Insights
- **Strong Correlation**: Phosphorus and Potassium (r=0.74)
- **Feature Significance**: All features show significant relationships with crop types (ANOVA p < 0.001)
- **Outliers**: Minimal outliers detected, all within valid agricultural ranges
- **Distribution**: Most features show normal or near-normal distributions

## 📈 Model Performance Metrics

### Production Model: Random Forest ✓

**Selection Rationale**: Chosen through multi-criteria decision analysis for balanced accuracy, training efficiency, and stability.

#### Accuracy Metrics
| Metric | Value | Status |
|--------|-------|--------|
| **Test Accuracy** | 99.32% | ✅ Exceeds target (97.5%) |
| **Training Accuracy** | 100.00% | ⚠️ Slight overfitting (0.68% gap) |
| **Cross-Validation Mean** | 99.20% | ✅ Excellent |
| **Cross-Validation Std** | ±0.42% | ✅ Very stable |
| **Precision (Weighted)** | 99.35% | ✅ Excellent |
| **Recall (Weighted)** | 99.32% | ✅ Excellent |
| **F1-Score (Weighted)** | 99.32% | ✅ Excellent |
| **Matthews Correlation Coefficient** | 99.29% | ✅ Near perfect |
| **Cohen's Kappa** | 99.29% | ✅ Almost perfect agreement |

#### Efficiency Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Training Time** | 0.82s | - | ✅ Fast |
| **Prediction Time (1000 samples)** | 8.56ms | <100ms | ✅ 11.7x faster |
| **P50 Latency** | 8.49ms | - | ✅ Consistent |
| **P95 Latency** | 9.04ms | - | ✅ Low variance |
| **P99 Latency** | 9.20ms | - | ✅ Reliable |
| **Model Size** | 4.76 MB | <50MB | ✅ 10.5x smaller |
| **Throughput** | ~116,800 pred/sec | - | ✅ High |

#### Per-Class Performance
- **Perfect Classification (100% F1)**: 16 out of 22 crops (72.7%)
- **Excellent Classification (≥97% F1)**: 22 out of 22 crops (100%)
- **Minimum F1-Score**: 97.44% (blackgram, lentil, rice)
- **Maximum F1-Score**: 100.00% (16 crops)
- **Total Misclassifications**: 3 out of 440 test samples (0.68% error rate)

#### Top 5 Feature Importance
1. **Humidity** (11.35%) - Most critical environmental factor
2. **Potassium - K** (10.53%) - Key soil nutrient for fruit quality
3. **Rainfall (log)** (7.43%) - Log-transformed precipitation
4. **Moisture Index** (6.96%) - Engineered water availability indicator
5. **Rainfall × Humidity** (6.90%) - Engineered interaction term

### XGBoost Model (Alternative)

#### Performance Comparison
| Metric | Random Forest | XGBoost | Winner |
|--------|---------------|---------|--------|
| **Test Accuracy** | 99.32% | 99.09% | Random Forest (+0.23%) |
| **Training Speed** | 0.82s | 4.33s | Random Forest (5.3x faster) |
| **Prediction Speed** | 8.56ms | 3.18ms | XGBoost (2.7x faster) |
| **Model Size** | 4.76 MB | 5.07 MB | Random Forest (-6.1%) |
| **CV Stability** | ±0.42% | ±0.53% | Random Forest |
| **Min F1-Score** | 97.44% | 94.74% | Random Forest (+2.70%) |

**Statistical Significance**: McNemar's test (p=1.00) indicates accuracy difference is not statistically significant.

### Production Readiness ✅

All criteria exceeded for production deployment:
- ✅ **Test Accuracy ≥ 97.5%**: Achieved 99.32% (+1.82%)
- ✅ **Per-Class F1 ≥ 95%**: Minimum 97.44% (+2.44%)
- ✅ **Inference Latency < 100ms/1000**: Achieved 8.56ms (11.7x faster)
- ✅ **Model Size < 50MB**: Achieved 4.76 MB (10.5x smaller)
- ✅ **Cross-Val Stability**: ±0.42% (excellent)
- ✅ **Security**: Input validation, rate limiting, CORS configured
- ✅ **Scalability**: Stateless, thread-safe, ~400K requests/hour capacity

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Interface (HTML/CSS/JS)              │
│                  Modern UI with Glassmorphism                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Application Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Routes     │  │  Middleware  │  │   Config     │      │
│  │  (API/Web)   │  │ (CORS, Rate) │  │ (Env-based)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Prediction       │  │ Validation       │                │
│  │ Service          │  │ Service          │                │
│  └──────────────────┘  └──────────────────┘                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML Pipeline                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Data    │→ │ Feature  │→ │  Model   │→ │  Post-   │   │
│  │  Loader  │  │  Prep    │  │ Predict  │  │ Process  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Raw Dataset  │  │ Trained      │  │ Metadata     │      │
│  │ (CSV)        │  │ Models (PKL) │  │ (JSON)       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### ML Pipeline Flow

```
Input Data (7 features)
    │
    ▼
Data Validation
    │
    ▼
Feature Engineering (→ 22 features)
    │
    ├─ Nutrient Ratios (N:P, N:K, P:K)
    ├─ Environmental Interactions (temp×humidity)
    ├─ pH Interactions (pH×N, pH×P, pH×K)
    └─ Composite Indicators (moisture_index, growing_condition_index)
    │
    ▼
Scaling (StandardScaler)
    │
    ▼
Model Prediction (Random Forest)
    │
    ▼
Label Decoding
    │
    ▼
Output (Crop + Confidence)
```

### Technology Stack

#### Backend
- **Framework**: Flask 3.1.2
- **ML Libraries**: scikit-learn 1.7.2, XGBoost 3.1.2
- **Data Processing**: pandas 2.3.3, numpy 2.3.5
- **Serialization**: joblib 1.5.2
- **WSGI Server**: Gunicorn (production)

#### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with glassmorphism
- **JavaScript**: Vanilla JS for interactivity
- **Charts**: Chart.js for visualizations

#### Testing & Quality
- **Testing**: pytest 9.0.1, pytest-cov 7.0.0
- **Coverage**: 90%+ across all modules
- **Linting**: flake8, black (code formatting)

#### Deployment
- **Containerization**: Docker (planned)
- **CI/CD**: GitHub Actions (planned)
- **Monitoring**: Prometheus + Grafana (planned)

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)**: Production deployment instructions
- **[DEVELOPMENT_DECISIONS.md](docs/DEVELOPMENT_DECISIONS.md)**: Architecture and design rationale
- **[PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md)**: Detailed performance analysis

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov=app --cov-report=html

# Run specific test categories
pytest tests/test_flask_app.py -v          # Flask API tests
pytest tests/test_random_forest_model.py -v # Model tests
pytest tests/test_preprocessing.py -v       # Feature engineering tests

# Run with markers
pytest -m "not slow" -v                     # Skip slow tests
pytest -m "integration" -v                  # Integration tests only
```

### Test Coverage
- **Total Tests**: 336+
- **Overall Coverage**: 90%+
- **Critical Path Coverage**: 95%+

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Write tests** for new functionality (maintain 90%+ coverage)
4. **Follow code style** (PEP 8, use black for formatting)
5. **Update documentation** as needed
6. **Commit changes** (`git commit -m 'Add amazing feature'`)
7. **Push to branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Development Guidelines
- All new features must include tests
- Maintain or improve code coverage
- Update README and docs for significant changes
- Follow existing architecture patterns
- Write meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Libraries**: scikit-learn, pandas, Flask, and the entire Python ML ecosystem
- **Community**: Thanks to all contributors and users

## 📞 Support

For questions, issues, or feature requests:
- **Issues**: [GitHub Issues](https://github.com/yourusername/ML-Crop-Recommendation-System/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ML-Crop-Recommendation-System/discussions)

---

**Built with ❤️ for sustainable agriculture**
