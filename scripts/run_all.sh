#!/bin/bash
set -e

# Stop old TF Serving if running
echo "Stopping any existing tfserving container..."
docker rm -f tfserving 2>/dev/null || true

# Start TensorFlow Serving
echo "Starting TensorFlow Serving..."
docker run -d --rm \
  --name tfserving \
  -p 8501:8501 \
  -v "$(pwd)/exported_model:/models/" \
  -e MODEL_NAME=simple_classifier \
  tensorflow/serving:2.14.0

# Build frontend
echo "Building frontend..."
docker build -t model-serving-frontend .

# Stop old frontend if running
echo "Stopping any existing frontend container..."
docker rm -f model-frontend 2>/dev/null || true

# Run frontend
echo "Starting frontend..."
docker run -d --rm \
  --name model-frontend \
  -p 5002:5002 \
  model-serving-frontend

echo "✅ All services are up!"
echo "Frontend running at: http://localhost:5002"
echo "TF Serving running at: http://localhost:8501/v1/models/simple_classifier"
