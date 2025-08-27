import tensorflow as tf
import os
import datetime

# Load and preprocess MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Build a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# --- Save with Serving Signature for TensorFlow Serving ---
version = 1  # Increment for new versions
export_path = f'./models/mnist_model/{version}'
os.makedirs(export_path, exist_ok=True)

# Define explicit serving function
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32)])
def serving_fn(input_data):
    return {"predictions": model(input_data)}

# Save the model with explicit signature
tf.saved_model.save(model, export_path, signatures={"serving_default": serving_fn})

print(f"Model successfully saved for TensorFlow Serving at: {export_path}")
