import json
import numpy as np
import requests
import tensorflow as tf

# Load test data
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., tf.newaxis]

# Pick one test image
sample = x_test[0].tolist()

# Send to REST API
data = json.dumps({"instances": [sample]})
headers = {"content-type": "application/json"}
url = "http://localhost:8501/v1/models/mnist_model:predict"

response = requests.post(url, data=data, headers=headers)
print(response.json())
