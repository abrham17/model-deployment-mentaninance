import requests
import numpy as np
import time
import json

def test_api_health():
    """Test API health endpoint"""
    response = requests.get("http://localhost:8501/v1/models/mnist_model")
    assert response.status_code == 200
    data = response.json()
    assert "model_version_status" in data
    print("✅ Health check passed")

def test_prediction_endpoint():
    """Test prediction endpoint with sample data"""
    # Create sample 28x28 image
    test_image = np.random.rand(28, 28, 1).tolist()
    
    url = "http://localhost:8501/v1/models/mnist_model:predict"
    data = {"instances": [test_image]}
    
    response = requests.post(url, json=data)
    assert response.status_code == 200
    
    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 1
    assert len(result["predictions"][0]) == 10
    
    # Check probabilities sum to ~1
    probs = result["predictions"][0]
    assert abs(sum(probs) - 1.0) < 0.01
    
    print("✅ Prediction endpoint test passed")

def test_batch_prediction():
    """Test batch prediction"""
    # Create batch of 5 images
    batch_images = [np.random.rand(28, 28, 1).tolist() for _ in range(5)]
    
    url = "http://localhost:8501/v1/models/mnist_model:predict"
    data = {"instances": batch_images}
    
    response = requests.post(url, json=data)
    assert response.status_code == 200
    
    result = response.json()
    assert len(result["predictions"]) == 5
    
    print("✅ Batch prediction test passed")

def test_performance():
    """Test API performance"""
    test_image = np.random.rand(28, 28, 1).tolist()
    url = "http://localhost:8501/v1/models/mnist_model:predict"
    data = {"instances": [test_image]}
    
    # Warmup
    requests.post(url, json=data)
    
    # Measure latency
    start_time = time.time()
    for _ in range(10):
        response = requests.post(url, json=data)
        assert response.status_code == 200
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / 10
    print(f"✅ Average latency: {avg_latency:.3f}s")
    
    # Assert reasonable performance
    assert avg_latency < 1.0, f"Latency too high: {avg_latency:.3f}s"

if __name__ == "__main__":
    print("Running integration tests...")
    
    try:
        test_api_health()
        test_prediction_endpoint()
        test_batch_prediction()
        test_performance()
        print("\n🎉 All integration tests passed!")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        exit(1)
