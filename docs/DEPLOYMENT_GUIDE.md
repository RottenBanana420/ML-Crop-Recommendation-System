# Deployment Guide

**Project**: ML Crop Recommendation System  
**Version**: 1.0  
**Last Updated**: November 28, 2025

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Deployment Options](#deployment-options)
4. [Production Configuration](#production-configuration)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Scaling Strategies](#scaling-strategies)
7. [Security Best Practices](#security-best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB
- **Disk Space**: 2 GB (includes models, dependencies, logs)
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+

#### Recommended for Production
- **CPU**: 4+ cores, 2.5+ GHz
- **RAM**: 8+ GB
- **Disk Space**: 10+ GB (for logs, model versions, backups)
- **OS**: Linux (Ubuntu 22.04 LTS)

### Software Requirements

- **Python**: 3.10 or higher (tested with 3.13.7)
- **pip**: Latest version
- **virtualenv** or **pyenv-virtualenv**: For environment isolation
- **Git**: For version control
- **Gunicorn**: WSGI HTTP server (production)
- **Nginx** (optional): Reverse proxy for production

### Network Requirements

- **Inbound**: Port 5000 (development) or 8000 (production with Gunicorn)
- **Outbound**: Internet access for pip package installation (initial setup only)

---

## Environment Configuration

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ML-Crop-Recommendation-System.git
cd ML-Crop-Recommendation-System
```

### 2. Create Virtual Environment

#### Using pyenv-virtualenv (Recommended)
```bash
# Install pyenv and pyenv-virtualenv if not already installed
# macOS
brew install pyenv pyenv-virtualenv

# Linux
curl https://pyenv.run | bash

# Install Python 3.13.7
pyenv install 3.13.7

# Create virtual environment
pyenv virtualenv 3.13.7 crop-recommendation-env

# Activate (automatic with pyenv local)
pyenv local crop-recommendation-env
```

#### Using venv
```bash
python3 -m venv crop-recommendation-env
source crop-recommendation-env/bin/activate  # Linux/macOS
# OR
crop-recommendation-env\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install production dependencies
pip install -r requirements.txt

# Verify installation
python -c "import flask, sklearn, xgboost; print('All dependencies installed successfully')"
```

### 4. Environment Variables

Create a `.env` file in the project root:

```bash
# Flask Configuration
FLASK_APP=run.py
FLASK_ENV=production  # Options: development, production, testing
SECRET_KEY=your-secret-key-here-change-in-production

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4  # Number of Gunicorn workers (2-4 × CPU cores)

# Logging
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=logs/app.log

# Rate Limiting
RATELIMIT_ENABLED=true
RATELIMIT_DEFAULT=100 per hour
RATELIMIT_STORAGE_URL=memory://

# CORS
CORS_ORIGINS=*  # Change to specific domains in production

# Model Configuration
MODEL_PATH=models/production_model.pkl
SCALER_PATH=models/scaler.pkl
LABEL_ENCODER_PATH=models/label_encoder.pkl
```

**Security Note**: Generate a strong SECRET_KEY:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 5. Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Check model files exist
ls -lh models/*.pkl

# Test Flask app
python run.py
# Visit http://localhost:5000 in browser
```

---

## Deployment Options

### Option 1: Local Development

**Use Case**: Development, testing, debugging

```bash
# Activate virtual environment
source crop-recommendation-env/bin/activate

# Run Flask development server
python run.py

# Or with Flask CLI
export FLASK_APP=run.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

**Access**: http://localhost:5000

**Note**: Development server is NOT suitable for production (single-threaded, no security hardening).

---

### Option 2: Production with Gunicorn

**Use Case**: Production deployment on single server

#### 2.1 Install Gunicorn
```bash
pip install gunicorn
```

#### 2.2 Create Gunicorn Configuration

Create `gunicorn_config.py`:

```python
# Gunicorn configuration file
import multiprocessing
import os

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = 'logs/gunicorn_access.log'
errorlog = 'logs/gunicorn_error.log'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'crop_recommendation_api'

# Server mechanics
daemon = False
pidfile = 'logs/gunicorn.pid'
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if using HTTPS)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'
```

#### 2.3 Start Gunicorn

```bash
# Create logs directory
mkdir -p logs

# Start Gunicorn
gunicorn -c gunicorn_config.py wsgi:app

# Or with command-line options
gunicorn --bind 0.0.0.0:8000 \
         --workers 4 \
         --timeout 30 \
         --access-logfile logs/access.log \
         --error-logfile logs/error.log \
         wsgi:app
```

#### 2.4 Run as Background Service

Create systemd service file `/etc/systemd/system/crop-recommendation.service`:

```ini
[Unit]
Description=Crop Recommendation API
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/path/to/ML-Crop-Recommendation-System
Environment="PATH=/path/to/crop-recommendation-env/bin"
ExecStart=/path/to/crop-recommendation-env/bin/gunicorn -c gunicorn_config.py wsgi:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable crop-recommendation
sudo systemctl start crop-recommendation
sudo systemctl status crop-recommendation
```

---

### Option 3: Docker Deployment

**Use Case**: Containerized deployment, cloud platforms, Kubernetes

#### 3.1 Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.13.7-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/predict/api/health')"

# Run Gunicorn
CMD ["gunicorn", "-c", "gunicorn_config.py", "wsgi:app"]
```

#### 3.2 Create docker-compose.yml

```yaml
version: '3.8'

services:
  crop-api:
    build: .
    container_name: crop-recommendation-api
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
      - WORKERS=4
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models:ro  # Read-only models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/predict/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

#### 3.3 Build and Run

```bash
# Build image
docker build -t crop-recommendation:latest .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

### Option 4: Cloud Deployment

#### AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p python-3.13 crop-recommendation-api

# Create environment
eb create crop-recommendation-prod

# Deploy
eb deploy

# Open in browser
eb open
```

#### Google Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/crop-recommendation

# Deploy to Cloud Run
gcloud run deploy crop-recommendation \
  --image gcr.io/PROJECT_ID/crop-recommendation \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure App Service

```bash
# Create resource group
az group create --name crop-recommendation-rg --location eastus

# Create App Service plan
az appservice plan create --name crop-recommendation-plan \
  --resource-group crop-recommendation-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group crop-recommendation-rg \
  --plan crop-recommendation-plan --name crop-recommendation-api \
  --runtime "PYTHON|3.13"

# Deploy
az webapp up --name crop-recommendation-api
```

---

## Production Configuration

### Nginx Reverse Proxy

Create `/etc/nginx/sites-available/crop-recommendation`:

```nginx
upstream crop_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL certificates
    ssl_certificate /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Logging
    access_log /var/log/nginx/crop-api-access.log;
    error_log /var/log/nginx/crop-api-error.log;

    # Client body size (for large requests)
    client_max_body_size 10M;

    # Timeouts
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;

    location / {
        proxy_pass http://crop_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/ML-Crop-Recommendation-System/app/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/crop-recommendation /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Monitoring and Maintenance

### Health Check Endpoint

The application provides a health check endpoint:

```bash
curl http://localhost:8000/predict/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-28T13:37:08Z"
}
```

### Logging

Logs are written to:
- **Application logs**: `logs/app.log`
- **Gunicorn access logs**: `logs/gunicorn_access.log`
- **Gunicorn error logs**: `logs/gunicorn_error.log`

Monitor logs in real-time:
```bash
tail -f logs/app.log
```

### Performance Monitoring

#### Prometheus + Grafana (Recommended)

Install prometheus-flask-exporter:
```bash
pip install prometheus-flask-exporter
```

Add to Flask app:
```python
from prometheus_flask_exporter import PrometheusMetrics
metrics = PrometheusMetrics(app)
```

Access metrics: `http://localhost:8000/metrics`

### Model Drift Detection

Monitor prediction distribution over time:
```python
# Log predictions for analysis
import json
from datetime import datetime

def log_prediction(input_data, prediction, confidence):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
        "confidence": confidence
    }
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

Analyze monthly:
```bash
# Check prediction distribution
python scripts/analyze_predictions.py --month 2025-11
```

### Automated Retraining

Set up cron job for monthly retraining:
```bash
# Edit crontab
crontab -e

# Add monthly retraining (1st day of month at 2 AM)
0 2 1 * * /path/to/crop-recommendation-env/bin/python /path/to/scripts/retrain_model.py
```

---

## Scaling Strategies

### Horizontal Scaling

#### Load Balancer Configuration

Use Nginx or HAProxy to distribute load:

```nginx
upstream crop_api_cluster {
    least_conn;  # Load balancing method
    server 192.168.1.10:8000 weight=1;
    server 192.168.1.11:8000 weight=1;
    server 192.168.1.12:8000 weight=1;
}

server {
    listen 80;
    location / {
        proxy_pass http://crop_api_cluster;
    }
}
```

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crop-recommendation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crop-recommendation
  template:
    metadata:
      labels:
        app: crop-recommendation
    spec:
      containers:
      - name: api
        image: crop-recommendation:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: crop-recommendation-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: crop-recommendation
```

### Vertical Scaling

Increase Gunicorn workers based on CPU cores:
```python
workers = (2 × CPU_cores) + 1
```

### Caching

Implement Redis caching for frequent predictions:

```python
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_prediction(input_data):
    cache_key = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    return None

def cache_prediction(input_data, prediction, ttl=3600):
    cache_key = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
    redis_client.setex(cache_key, ttl, json.dumps(prediction))
```

---

## Security Best Practices

### 1. Environment Variables

Never commit `.env` files or secrets to version control:
```bash
# Add to .gitignore
echo ".env" >> .gitignore
```

### 2. HTTPS/TLS

Always use HTTPS in production:
- Use Let's Encrypt for free SSL certificates
- Configure strong TLS settings (TLS 1.2+)
- Enable HSTS headers

### 3. Rate Limiting

Already configured in Flask app:
```python
RATELIMIT_DEFAULT = "100 per hour"
```

Adjust based on usage patterns.

### 4. Input Validation

All inputs are validated in the API:
- Type checking
- Range validation
- Sanitization

### 5. CORS Configuration

Restrict CORS to specific domains in production:
```python
CORS_ORIGINS = ["https://your-frontend.com"]
```

### 6. Security Headers

Add security headers in Nginx:
```nginx
add_header X-Frame-Options "SAMEORIGIN";
add_header X-Content-Type-Options "nosniff";
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
```

---

## Troubleshooting

### Common Issues

#### 1. Model File Not Found
```
Error: FileNotFoundError: models/production_model.pkl
```
**Solution**: Ensure model files are in the correct location:
```bash
ls -lh models/*.pkl
# Should show: production_model.pkl, scaler.pkl, label_encoder.pkl
```

#### 2. Port Already in Use
```
Error: Address already in use
```
**Solution**: Kill process using the port:
```bash
lsof -ti:8000 | xargs kill -9
```

#### 3. Permission Denied
```
Error: Permission denied: 'logs/app.log'
```
**Solution**: Fix permissions:
```bash
sudo chown -R $USER:$USER logs/
chmod 755 logs/
```

#### 4. Out of Memory
```
Error: MemoryError
```
**Solution**: Reduce Gunicorn workers or increase server RAM:
```python
workers = 2  # Reduce from 4
```

#### 5. Slow Predictions
**Diagnosis**:
```bash
# Check model loading time
python -c "import time; start=time.time(); import joblib; joblib.load('models/production_model.pkl'); print(f'Load time: {time.time()-start:.2f}s')"
```

**Solution**: Preload models in Gunicorn:
```python
# gunicorn_config.py
def on_starting(server):
    import joblib
    server.model = joblib.load('models/production_model.pkl')
```

---

## Deployment Checklist

Before deploying to production:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Environment variables configured
- [ ] SECRET_KEY is strong and unique
- [ ] HTTPS/TLS certificates installed
- [ ] CORS origins restricted
- [ ] Rate limiting enabled
- [ ] Logging configured
- [ ] Health check endpoint working
- [ ] Monitoring set up (Prometheus/Grafana)
- [ ] Backup strategy in place
- [ ] Load testing completed
- [ ] Security headers configured
- [ ] Documentation updated
- [ ] Rollback plan prepared

---

**For Support**: See [GitHub Issues](https://github.com/yourusername/ML-Crop-Recommendation-System/issues)

**Last Updated**: November 28, 2025
