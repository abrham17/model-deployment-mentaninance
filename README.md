# MNIST Model Serving with TensorFlow Serving

A production-ready machine learning system for serving MNIST digit classification models with automated retraining, monitoring, and deployment capabilities.

## 🏗️ Architecture Overview

\`\`\`
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client Apps   │───▶│  TF Serving API  │───▶│  MNIST Model    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Prometheus     │
                       │   Monitoring     │
                       └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Weekly Retrain   │
                       │ Pipeline (97%+)  │
                       └──────────────────┘
\`\`\`

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.8+
- TensorFlow 2.x

### 1. Train Initial Model
\`\`\`bash
python train_model.py
\`\`\`

### 2. Build and Start Services
\`\`\`bash
bash scripts/build.sh
bash scripts/run.sh
\`\`\`

### 3. Test the API
\`\`\`bash
bash scripts/test_api.sh
\`\`\`

## 📡 API Documentation

### Base URLs
- **REST API**: `http://localhost:8501`
- **gRPC API**: `localhost:8500`
- **Prometheus Metrics**: `http://localhost:9090`

### Endpoints

#### 1. Model Metadata
Get information about the served model.

**Request:**
\`\`\`bash
GET /v1/models/mnist_model/metadata
\`\`\`

**Response:**
\`\`\`json
{
  "model_spec": {
    "name": "mnist_model",
    "signature_name": "",
    "version": "1"
  },
  "metadata": {
    "signature_def": {
      "signature_def": {
        "serving_default": {
          "inputs": {
            "input_1": {
              "dtype": "DT_FLOAT",
              "tensor_shape": {
                "dim": [
                  {"size": "-1", "name": ""},
                  {"size": "28", "name": ""},
                  {"size": "28", "name": ""},
                  {"size": "1", "name": ""}
                ]
              }
            }
          },
          "outputs": {
            "dense_1": {
              "dtype": "DT_FLOAT",
              "tensor_shape": {
                "dim": [
                  {"size": "-1", "name": ""},
                  {"size": "10", "name": ""}
                ]
              }
            }
          }
        }
      }
    }
  }
}
\`\`\`

#### 2. Model Prediction
Classify MNIST digit images.

**Request:**
\`\`\`bash
POST /v1/models/mnist_model:predict
Content-Type: application/json
\`\`\`

**Request Body:**
\`\`\`json
{
  "instances": [
    [[[0.0], [0.0], ..., [1.0]], 
     [[0.0], [0.1], ..., [0.9]], 
     ...
     [[0.0], [0.0], ..., [0.0]]]
  ]
}
\`\`\`

**Response:**
\`\`\`json
{
  "predictions": [
    [0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.989, 0.001]
  ]
}
\`\`\`

#### 3. Health Check
Check if the model server is healthy.

**Request:**
\`\`\`bash
GET /v1/models/mnist_model
\`\`\`

**Response:**
\`\`\`json
{
  "model_version_status": [
    {
      "version": "1",
      "state": "AVAILABLE",
      "status": {
        "error_code": "OK",
        "error_message": ""
      }
    }
  ]
}
\`\`\`

## 🔧 Sample Requests

### Python Client Example
\`\`\`python
import requests
import numpy as np
import json

# Create sample 28x28 image
image = np.random.rand(28, 28, 1).tolist()

# Prepare request
url = "http://localhost:8501/v1/models/mnist_model:predict"
data = {"instances": [image]}

# Make prediction
response = requests.post(url, json=data)
predictions = response.json()["predictions"][0]

# Get predicted digit
predicted_digit = np.argmax(predictions)
confidence = predictions[predicted_digit]

print(f"Predicted digit: {predicted_digit}")
print(f"Confidence: {confidence:.4f}")
\`\`\`

### cURL Example
\`\`\`bash
# Create test data
echo '{
  "instances": [
    [[[0.0], [0.0]], [[0.0], [0.0]]]
  ]
}' > test_data.json

# Make prediction
curl -X POST \
  http://localhost:8501/v1/models/mnist_model:predict \
  -H 'Content-Type: application/json' \
  -d @test_data.json
\`\`\`

### JavaScript/Node.js Example
\`\`\`javascript
const axios = require('axios');

async function predictDigit(imageData) {
  try {
    const response = await axios.post(
      'http://localhost:8501/v1/models/mnist_model:predict',
      { instances: [imageData] },
      { headers: { 'Content-Type': 'application/json' } }
    );
    
    const predictions = response.data.predictions[0];
    const predictedDigit = predictions.indexOf(Math.max(...predictions));
    
    return {
      digit: predictedDigit,
      confidence: predictions[predictedDigit],
      allProbabilities: predictions
    };
  } catch (error) {
    console.error('Prediction failed:', error.message);
    throw error;
  }
}
\`\`\`

## 📊 Monitoring & Metrics

### Prometheus Metrics
Access metrics at: `http://localhost:9090`

#### Key Metrics:
- `tensorflow_serving_request_count` - Total requests processed
- `tensorflow_serving_request_latency` - Request processing time
- `tensorflow_serving_model_warmup_latency` - Model loading time
- `tensorflow_serving_batch_size` - Batch processing metrics

### Custom Application Metrics
\`\`\`python
# Example: Adding custom metrics to your application
from prometheus_client import Counter, Histogram, Gauge

# Request counters
prediction_requests = Counter('mnist_predictions_total', 'Total predictions made')
failed_requests = Counter('mnist_prediction_failures_total', 'Failed predictions')

# Latency histogram
prediction_latency = Histogram('mnist_prediction_duration_seconds', 'Prediction latency')

# Model accuracy gauge
model_accuracy = Gauge('mnist_model_accuracy', 'Current model accuracy')
\`\`\`

### Grafana Dashboard Setup
1. Connect Grafana to Prometheus: `http://prometheus:9090`
2. Import dashboard configuration from `monitoring/grafana-dashboard.json`
3. Key panels:
   - Request rate and latency
   - Model accuracy over time
   - Error rates
   - System resource usage

## 🔄 Retraining Pipeline

### Automated Weekly Retraining
The system automatically retrains the model weekly with an accuracy gate:

- **Trigger**: Every Monday at 2 AM UTC
- **Accuracy Threshold**: 97%
- **Promotion**: New model promoted only if accuracy ≥ 97%
- **Rollback**: Previous version maintained if new model fails

### Manual Retraining
\`\`\`bash
# Run retraining pipeline
bash scripts/retrain.sh

# Run with custom parameters
python retrain_pipeline.py --epochs 15 --config config.json
\`\`\`

### Configuration
Edit `config.json` to customize:
\`\`\`json
{
  "accuracy_threshold": 0.97,
  "training_config": {
    "epochs": 10,
    "batch_size": 32
  },
  "notification_config": {
    "enabled": true
  }
}
\`\`\`

## 🔐 Authentication & Security

### API Authentication (Optional)
For production deployments, implement authentication:

\`\`\`python
# Example: JWT token validation
import jwt
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return {'error': 'No token provided'}, 401
        
        try:
            jwt.decode(token, 'your-secret-key', algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}, 401
            
        return f(*args, **kwargs)
    return decorated_function
\`\`\`

### Environment Variables
Set these environment variables for secure operation:

\`\`\`bash
# Email notifications
export SENDER_EMAIL="alerts@company.com"
export SENDER_PASSWORD="app-specific-password"
export RECIPIENT_EMAIL="admin@company.com"

# Optional: API authentication
export JWT_SECRET_KEY="your-secret-key"
export API_KEY="your-api-key"
\`\`\`

## 🔄 Model Promotion & Rollback

### Promotion Process
1. New model trained and evaluated
2. Accuracy check against 97% threshold
3. Backup current model version
4. Promote new model if threshold met
5. Update serving containers
6. Send notification

### Manual Rollback
\`\`\`python
# Rollback to previous version
from retrain_pipeline import ModelRetrainer

retrainer = ModelRetrainer()
retrainer.rollback_model(target_version=1)
\`\`\`

### Version Management
\`\`\`bash
# List all model versions
ls models/mnist_model/

# Check model metadata
cat models/mnist_model/metadata_v2.json

# View backup models
ls models/backups/
\`\`\`

## 🚨 Troubleshooting

### Common Issues

#### 1. Model Not Loading
\`\`\`bash
# Check model directory structure
ls -la models/mnist_model/1/

# Verify SavedModel format
saved_model_cli show --dir models/mnist_model/1 --all
\`\`\`

#### 2. Docker Build Fails
\`\`\`bash
# Check Docker daemon
docker info

# Rebuild with verbose output
docker build -t mnist-serving:latest . --no-cache
\`\`\`

#### 3. Low Accuracy During Retraining
- Check training data quality
- Verify preprocessing steps
- Adjust hyperparameters in `config.json`
- Review training logs in `retrain.log`

#### 4. API Connection Issues
\`\`\`bash
# Test connectivity
curl -f http://localhost:8501/v1/models/mnist_model

# Check container logs
docker-compose logs mnist-serving
\`\`\`

### Logs and Debugging
\`\`\`bash
# View application logs
tail -f retrain.log

# Docker container logs
docker-compose logs -f mnist-serving

# Prometheus metrics
curl http://localhost:8501/monitoring/prometheus/metrics
\`\`\`

## 📈 Performance Optimization

### Batch Processing
\`\`\`python
# Process multiple images at once
data = {"instances": [image1, image2, image3, ...]}
response = requests.post(url, json=data)
\`\`\`

### Model Optimization
- Use TensorFlow Lite for mobile deployment
- Implement model quantization for faster inference
- Consider TensorRT for GPU acceleration

### Scaling
\`\`\`yaml
# docker-compose.yml - Scale serving instances
services:
  mnist-serving:
    deploy:
      replicas: 3
    ports:
      - "8501-8503:8501"
\`\`\`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check troubleshooting section
- Review logs for error details

---

**Last Updated**: $(date)
**Version**: 1.0.0
# model-deployment-mentaninance
