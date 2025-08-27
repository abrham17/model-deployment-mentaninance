#!/bin/bash

echo "🔄 Starting MNIST Model Retraining Pipeline..."

# Set environment variables for email notifications (optional)
# export SENDER_EMAIL="your-email@gmail.com"
# export SENDER_PASSWORD="your-app-password"
# export RECIPIENT_EMAIL="admin@company.com"
# export SMTP_SERVER="smtp.gmail.com"
# export SMTP_PORT="587"

# Create necessary directories
mkdir -p models/mnist_model
mkdir -p models/backups
mkdir -p logs

# Run the retraining pipeline
python retrain_pipeline.py --epochs 10 --config config.json

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Retraining pipeline completed successfully!"
    
    # Restart Docker containers if model was promoted
    if [ -f "models/mnist_model/$(ls models/mnist_model | grep -E '^[0-9]+$' | sort -n | tail -1)" ]; then
        echo "🔄 Restarting TensorFlow Serving containers..."
        docker-compose restart mnist-serving
    fi
else
    echo "❌ Retraining pipeline failed!"
    exit 1
fi
