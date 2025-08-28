# TensorFlow Serving ML Pipeline with Automated Retraining

A complete machine learning pipeline featuring TensorFlow model training, serving via TF Serving + Docker, and automated weekly retraining with accuracy gates.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)

### Launch the Complete Pipeline
\`\`\`bash
# Start all services (trainer, TF Serving, frontend, monitoring)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
\`\`\`

### Access Points
- **Frontend Web UI**: http://localhost:5002
- **TF Serving REST API**: http://localhost:8501
- **TF Serving gRPC**: localhost:8500
- **Prometheus Metrics**: http://localhost:8501/monitoring/prometheus/metrics

##  Model Training & Deployment

### Automated Retraining Pipeline
The system includes an automated retraining pipeline with quality gates:

\`\`\`bash
# Manual training trigger
python retrain_pipeline.py

# Or use the convenience script
./scripts/manual_train.sh
\`\`\`

### Accuracy Gate System
- **Threshold**: 97% accuracy (configurable in `config.json`)
- **Promotion**: Models meeting the threshold are automatically deployed
- **Rollback**: Failing models are rejected, previous version remains active
- **Notifications**: All decisions logged to `notifications.log`

### Model Versioning
- Models stored in `models/{version}/` directories
- TF Serving automatically serves the highest version number
- Each model includes `metrics.json` with performance data

##  API Documentation

### Prediction Endpoint

**POST /predict**

Make predictions using the deployed model.

#### JSON Request Format
\`\`\`bash
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.5, 2.3], [0.8, -1.2]]}'
\`\`\`

#### Form Request Format
\`\`\`bash
curl -X POST http://localhost:5002/predict \
  -d "x1=1.5&x2=2.3"
\`\`\`

#### Response Format
\`\`\`json
{
  "predictions": [[0.8234567], [0.1234567]]
}
\`\`\`

### Direct TensorFlow Serving API

**POST http://localhost:8501/v1/models/simple_classifier:predict**

\`\`\`bash
curl -X POST http://localhost:8501/v1/models/simple_classifier:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.5, 2.3]]}'
\`\`\`

## ⚙ Configuration

### Model Configuration (`config.json`)
\`\`\`json
{
  "model_name": "simple_classifier",
  "gate": {
    "min_accuracy_percent": 97.0
  },
  "train": {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.01
  }
}
\`\`\`

### Environment Variables
- `TF_HOST`: TensorFlow Serving host (default: http://localhost:8501)
- `MODEL_NAME`: Model name for serving (default: simple_classifier)
- `TF_CPP_MIN_LOG_LEVEL`: TensorFlow logging level

##  Monitoring & Metrics

### Prometheus Integration
- TF Serving exposes metrics on `/monitoring/prometheus/metrics`
- Automatic model performance tracking
- Request/response metrics and latencies

### Model Metrics
Each trained model includes comprehensive metrics:
- Test accuracy percentage
- Training loss
- Timestamp and version info
- Dataset characteristics

##  Weekly Retraining Setup

### Automated Scheduling (Cron)
Add to your crontab for weekly retraining:
\`\`\`bash
# Run every Sunday at 2 AM
0 2 * * 0 cd /path/to/project && python retrain_pipeline.py
\`\`\`

### GitHub Actions (CI/CD)
Automated workflows for testing and deployment are configured in `.github/workflows/`.

## 🛠️ Development

### Local Development Setup
\`\`\`bash
# Install dependencies
pip install -r requirements.frontend.txt

# Run training locally
python train_tf.py

# Start Flask app locally
python app.py
\`\`\`

### Manual Operations
\`\`\`bash
# Train a new model version
./scripts/manual_train.sh

# Start all services
./scripts/run_all.sh

# View training logs
cat last_train_log.txt

# Check notifications
cat notifications.log
\`\`\`

##  Troubleshooting

### Common Issues
1. **Port conflicts**: Ensure ports 5002, 8500, 8501 are available
2. **Model loading**: Check `docker-compose logs tfserving` for TF Serving issues
3. **Training failures**: Review `last_train_log.txt` for training errors

### Health Checks
\`\`\`bash
# Check TF Serving health
curl http://localhost:8501/v1/models/simple_classifier

# Test prediction endpoint
curl -X POST http://localhost:5002/predict -d "x1=1&x2=1"
\`\`\`

##  Project Structure
\`\`\`
├── models/                 # Versioned TensorFlow SavedModels
├── scripts/               # Training and deployment scripts
├── monitoring/            # Prometheus configuration
├── templates/             # Flask HTML templates
├── .github/workflows/     # CI/CD automation
├── docker-compose.yml     # Service orchestration
├── retrain_pipeline.py    # Automated retraining with gates
├── train_tf.py           # Model training script
├── app.py                # Flask prediction API
└── config.json           # Model and training configuration
\`\`\`

## 📄 License
MIT License - see LICENSE file for details.
