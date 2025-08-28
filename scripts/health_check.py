#!/usr/bin/env python3
"""
Health check script for the ML pipeline
Validates all components are working correctly
"""

import requests
import json
import sys
from pathlib import Path

def check_tf_serving():
    """Check TensorFlow Serving health"""
    try:
        response = requests.get("http://localhost:8501/v1/models/simple_classifier", timeout=5)
        if response.status_code == 200:
            print(" TensorFlow Serving is healthy")
            return True
        else:
            print(f"TensorFlow Serving unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"TensorFlow Serving connection failed: {e}")
        return False

def check_frontend():
    """Check Flask frontend health"""
    try:
        response = requests.get("http://localhost:5002/", timeout=5)
        if response.status_code == 200:
            print(" Frontend is healthy")
            return True
        else:
            print(f"Frontend unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"Frontend connection failed: {e}")
        return False

def check_prediction_api():
    """Test prediction endpoint"""
    try:
        data = {"instances": [[1.0, 1.0]]}
        response = requests.post("http://localhost:5002/predict", 
                               json=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f" Prediction API working: {result}")
            return True
        else:
            print(f" Prediction API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f" Prediction API error: {e}")
        return False

def check_models():
    """Check model versions and metrics"""
    try:
        model_base = Path("models")
        if not model_base.exists():
            print(" Models directory not found")
            return False
            
        versions = sorted([int(p.name) for p in model_base.iterdir() 
                          if p.is_dir() and p.name.isdigit()])
        
        if not versions:
            print(" No model versions found")
            return False
            
        latest = versions[-1]
        metrics_path = model_base / str(latest) / "metrics.json"
        
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            accuracy = metrics.get("test_accuracy_percent", 0)
            print(f"Latest model v{latest} accuracy: {accuracy}%")
            
            if accuracy < 95.0:
                print(" Warning: Model accuracy below 95%")
                
            return True
        else:
            print(f"No metrics found for model v{latest}")
            return False
            
    except Exception as e:
        print(f"Model check failed: {e}")
        return False

def check_prometheus_metrics():
    """Check Prometheus metrics endpoint"""
    try:
        response = requests.get("http://localhost:8501/monitoring/prometheus/metrics", timeout=5)
        if response.status_code == 200:
            print("Prometheus metrics available")
            return True
        else:
            print(f"Prometheus metrics unavailable: {response.status_code}")
            return False
    except Exception as e:
        print(f"Prometheus metrics error: {e}")
        return False

def main():
    """Run all health checks"""
    print("Running ML Pipeline Health Checks...\n")

    checks = [
        ("TensorFlow Serving", check_tf_serving),
        ("Frontend API", check_frontend),
        ("Prediction Endpoint", check_prediction_api),
        ("Model Versions", check_models),
        ("Prometheus Metrics", check_prometheus_metrics),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        if check_func():
            passed += 1
    
    print(f"\n Health Check Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print(" All systems healthy!")
        sys.exit(0)
    else:
        print(" Some systems need attention")
        sys.exit(1)

if __name__ == "__main__":
    main()
