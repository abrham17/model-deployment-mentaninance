#!/bin/bash
set -e

echo "🚀 Starting ML Pipeline with Docker Compose..."

# Check if models exist, if not train first
if [ ! -d "models/1" ] && [ ! -d "models/2" ]; then
    echo "📚 No models found. Training initial model..."
    python train_tf.py
fi

# Stop any existing services
echo "🛑 Stopping existing services..."
docker-compose down --remove-orphans 2>/dev/null || true

# Clean up any old containers
docker rm -f tfserving model-frontend 2>/dev/null || true

# Build and start all services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service status
echo "📊 Service Status:"
docker-compose ps

# Test endpoints
echo "🧪 Testing endpoints..."
if curl -s http://localhost:8501/v1/models/simple_classifier > /dev/null; then
    echo "✅ TensorFlow Serving is healthy"
else
    echo "❌ TensorFlow Serving is not responding"
fi

if curl -s http://localhost:5002/health > /dev/null; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend is not responding"
fi

echo ""
echo "🎉 All services are up!"
echo "Frontend: http://localhost:5002"
echo "TF Serving: http://localhost:8501/v1/models/simple_classifier"
echo "Prometheus: http://localhost:9090"
echo ""
echo "To stop services: docker-compose down"
echo "To view logs: docker-compose logs -f"
