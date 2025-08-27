#!/bin/bash

echo "Starting MNIST TensorFlow Serving..."

# Stop existing container if running
docker-compose down

# Start the services
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Services started successfully!"
    echo ""
    echo "🚀 TensorFlow Serving is running at:"
    echo "   REST API: http://localhost:8501"
    echo "   gRPC API: localhost:8500"
    echo "   Prometheus: http://localhost:9090"
    echo ""
    echo "📊 Health check: curl http://localhost:8501/v1/models/mnist_model"
    echo "📝 Test prediction: bash scripts/test_api.sh"
else
    echo "❌ Failed to start services!"
    exit 1
fi
