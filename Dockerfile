FROM tensorflow/serving:latest

# Copy the model to the container
COPY models/mnist_model /models/mnist_model

# Set environment variables for better performance
ENV MODEL_NAME=mnist_model
ENV MODEL_BASE_PATH=/models/mnist_model

# Expose ports: 8500 for gRPC, 8501 for REST
EXPOSE 8500 8501

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/v1/models/${MODEL_NAME} || exit 1

# Run TF Serving with optimized configuration
ENTRYPOINT ["tensorflow_model_server", \
  "--port=8500", \
  "--rest_api_port=8501", \
  "--model_name=mnist_model", \
  "--model_base_path=/models/mnist_model", \
  "--monitoring_config_file=/models/monitoring.config", \
  "--allow_version_labels_for_unavailable_models=true"]
