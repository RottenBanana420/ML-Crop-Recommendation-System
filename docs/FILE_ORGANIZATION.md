# File Organization Summary

**Date**: November 28, 2025  
**Project**: ML Crop Recommendation System

## Overview
This document describes the file organization structure of the ML Crop Recommendation System and the cleanup actions taken to maintain a clean, professional project structure.

## Directory Structure

```
ML-Crop-Recommendation-System/
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── .python-version           # Python version specification
├── LICENSE                   # Project license
├── README.md                 # Main project documentation
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
├── run.py                    # Flask application entry point
├── wsgi.py                   # WSGI entry point for production
│
├── app/                      # Flask web application
│   ├── __init__.py           # Flask app factory
│   ├── config.py             # Application configuration
│   ├── routes/               # API routes
│   ├── services/             # Business logic services
│   ├── static/               # Static assets (CSS, JS, images)
│   └── templates/            # HTML templates
│
├── src/                      # Core ML pipeline source code
│   ├── __init__.py
│   ├── analysis/             # Exploratory data analysis
│   ├── data/                 # Data loading and validation
│   ├── features/             # Feature engineering
│   ├── models/               # Model training and evaluation
│   └── utils/                # Utility functions
│
├── tests/                    # Comprehensive test suite
│   ├── test_*.py             # Unit and integration tests
│   └── conftest.py           # Pytest fixtures and configuration
│
├── data/                     # Data storage
│   └── raw/                  # Raw dataset (Crop_recommendation.csv)
│
├── models/                   # Trained models and metadata
│   ├── *.pkl                 # Trained model files (gitignored)
│   ├── *_metadata.json       # Model metadata (tracked)
│   └── *_metrics.json        # Model performance metrics (tracked)
│
├── reports/                  # Generated reports and coverage
│   ├── .coverage             # Coverage data file (gitignored)
│   ├── coverage.json         # JSON coverage report (gitignored)
│   ├── htmlcov/              # HTML coverage report (gitignored)
│   └── README.md             # Reports directory documentation
│
├── docs/                     # Project documentation
│   ├── README.md             # Documentation index
│   ├── DEPLOYMENT_GUIDE.md   # Deployment instructions
│   ├── DEVELOPMENT_DECISIONS.md  # Architecture and design decisions
│   ├── PERFORMANCE_REPORT.md # Model performance analysis
│   ├── FILE_ORGANIZATION.md  # This file
│   ├── design/               # UI/UX design documents
│   └── planning/             # Project planning documents
│
└── logs/                     # Application logs (gitignored)
    └── *.log                 # Log files
```

## Organization Actions Taken

### 1. Moved Coverage Reports to `reports/` Directory
**Problem**: Coverage files were scattered in the root directory, cluttering the project structure.

**Actions**:
- Moved `coverage.json` from root to `reports/`
- Moved `.coverage` from root to `reports/`
- Moved `htmlcov/` from root to `reports/`

**Benefit**: Keeps the root directory clean and all generated reports in one location.

### 2. Updated `.gitignore`
**Actions**:
- Added `reports/coverage.json` to gitignore rules
- Ensured all coverage-related files in `reports/` are properly ignored

**Benefit**: Prevents generated coverage reports from being tracked in version control.

### 3. Updated `pytest.ini`
**Actions**:
- Changed `--cov-report=html` to `--cov-report=html:reports/htmlcov`
- Added `--cov-report=json:reports/coverage.json`

**Benefit**: Future test runs will automatically output coverage reports to the correct location.

### 4. Verified Directory Structure
**Actions**:
- Confirmed all source code is properly organized in `src/` and `app/`
- Verified all tests are in `tests/` directory
- Confirmed documentation is in `docs/` directory
- Verified data is in `data/` directory
- Confirmed models are in `models/` directory

## Files Excluded from Version Control

### Generated Files (Can be Regenerated)
- `reports/.coverage` - Coverage data file
- `reports/coverage.json` - JSON coverage report
- `reports/htmlcov/` - HTML coverage report
- `models/*.pkl` - Trained model binaries
- `logs/*.log` - Application logs
- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - Pytest cache

### Sensitive Files (Never Commit)
- `.env` - Environment variables with secrets
- `*.key`, `*.pem` - Private keys
- `secrets/` - Secret files

### Tracked Files (Important to Keep)
- `models/*_metadata.json` - Model metadata
- `models/*_metrics.json` - Model performance metrics
- `data/raw/*.csv` - Raw dataset (if small and non-sensitive)
- All source code in `src/`, `app/`, `tests/`
- All documentation in `docs/`
- Configuration files: `pytest.ini`, `requirements.txt`, `.env.example`

## Best Practices Followed

1. **Separation of Concerns**: Code is organized by function (data, models, analysis, features)
2. **Clean Root Directory**: Only essential configuration files in root
3. **Centralized Reports**: All generated reports in `reports/` directory
4. **Comprehensive Documentation**: All docs in `docs/` directory
5. **Test Organization**: All tests in `tests/` directory
6. **Proper Gitignore**: Generated and sensitive files excluded from version control

## Maintenance Guidelines

### Adding New Files
- **Source code**: Place in appropriate `src/` subdirectory
- **Tests**: Place in `tests/` directory with `test_` prefix
- **Documentation**: Place in `docs/` directory
- **Models**: Save to `models/` directory (binaries will be gitignored)
- **Data**: Place in `data/raw/` or `data/processed/`

### Running Tests
```bash
pytest
```
Coverage reports will automatically be generated in `reports/` directory.

### Viewing Coverage
```bash
open reports/htmlcov/index.html  # macOS
xdg-open reports/htmlcov/index.html  # Linux
```

## Summary

The project now has a clean, professional structure with:
- ✅ Clean root directory (only essential config files)
- ✅ All coverage reports in `reports/` directory
- ✅ Proper gitignore rules for generated files
- ✅ Organized source code structure
- ✅ Comprehensive documentation
- ✅ Clear separation of concerns

This organization makes the project easier to navigate, maintain, and collaborate on.
