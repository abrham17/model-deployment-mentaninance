#!/bin/bash

echo "Testing MNIST TensorFlow Serving API..."

# Wait for service to be ready
echo "Waiting for service to be ready..."
sleep 10

# Test model metadata
echo "📋 Getting model metadata:"
curl -s http://localhost:8501/v1/models/mnist_model/metadata | jq '.'

echo -e "\n🧪 Testing prediction with sample data:"

# Create test data (28x28 image with some pattern)
python3 -c "
import json
import numpy as np

# Create a simple test image (digit-like pattern)
test_image = np.zeros((28, 28))
test_image[10:18, 10:18] = 1.0  # Simple square pattern
test_data = test_image.reshape(1, 28, 28, 1).tolist()

# Create prediction request
request = {
    'instances': test_data
}

print(json.dumps(request))
" > test_request.json

# Make prediction request
curl -X POST \
  http://localhost:8501/v1/models/mnist_model:predict \
  -H 'Content-Type: application/json' \
  -d @test_request.json | jq '.'

# Clean up
rm -f test_request.json

echo -e "\n✅ API test completed!"
