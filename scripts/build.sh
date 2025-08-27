#!/bin/bash

echo "Building MNIST TensorFlow Serving container..."

# Check if model exists
if [ ! -d "models/mnist_model" ]; then
    echo "Error: Model not found at models/mnist_model"
    echo "Please run 'python train_model.py' first to train the model"
    exit 1
fi

# Build the Docker image
docker build -t mnist-serving:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo "Run 'bash scripts/run.sh' to start the serving container"
else
    echo "❌ Docker build failed!"
    exit 1
fi
