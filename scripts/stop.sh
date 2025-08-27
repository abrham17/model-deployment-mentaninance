#!/bin/bash

echo "Stopping MNIST TensorFlow Serving..."

docker-compose down

if [ $? -eq 0 ]; then
    echo "✅ Services stopped successfully!"
else
    echo "❌ Failed to stop services!"
    exit 1
fi
