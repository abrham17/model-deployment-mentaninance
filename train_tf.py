import os, json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime

HERE = Path(__file__).parent
with open(HERE / "config.json") as f:
    cfg = json.load(f)

rng = cfg.get("random_state", 42)
ds = cfg["dataset"]
X, y = make_classification(
    n_samples=ds.get("n_samples", 1000),
    n_features=ds.get("n_features", 2),
    n_informative=ds.get("n_features", 2),
    n_redundant=0,
    n_repeated=0,
    class_sep=ds.get("class_sep", 2.0),
    flip_y=ds.get("flip_y", 0.01),
    random_state=rng,
)
X = X.astype("float32")
y = y.astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)

# Simple logistic regression: Dense(1, sigmoid) with no hidden layers
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
opt = tf.keras.optimizers.SGD(learning_rate=cfg["train"].get("learning_rate", 0.01))
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    validation_split=cfg["train"].get("validation_split", 0.2),
    epochs=cfg["train"].get("epochs", 20),
    batch_size=cfg["train"].get("batch_size", 32),
    verbose=0
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
acc_pct = float(acc * 100.0)

# Versioning: next numeric version under models/<version>
model_base = HERE / cfg["model_base_path"]
model_base.mkdir(parents=True, exist_ok=True)
# Determine next version id
existing = [int(p.name) for p in model_base.iterdir() if p.is_dir() and p.name.isdigit()]
next_ver = (max(existing) + 1) if existing else int(cfg.get("current_version", 1))
version_path = model_base / str(next_ver)
version_path.mkdir(parents=True, exist_ok=True)

# Save as TensorFlow SavedModel
tf.saved_model.save(model, str(version_path))

metrics = {
    "version": next_ver,
    "test_accuracy_percent": acc_pct,
    "test_loss": float(loss),
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "n_features": int(X.shape[1]),
    "n_samples": int(X.shape[0])
}
with open(version_path / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics))
