import os, sys, json, requests

TF_HOST = os.environ.get("TF_HOST", "http://localhost:8501")
MODEL = os.environ.get("MODEL_NAME", "simple_classifier")
vA = os.environ.get("VERSION_A")  # current
vB = os.environ.get("VERSION_B")  # candidate

if not (vA and vB):
    print("Set VERSION_A and VERSION_B env vars to compare.")
    sys.exit(1)

urlA = f"{TF_HOST}/v1/models/{MODEL}/versions/{vA}:predict"
urlB = f"{TF_HOST}/v1/models/{MODEL}/versions/{vB}:predict"

payload = {"instances": [[0.1, -0.2], [1.0, 1.0], [-1.5, 0.3]]}
rA = requests.post(urlA, json=payload).json()
rB = requests.post(urlB, json=payload).json()

print("A:", json.dumps(rA, indent=2))
print("B:", json.dumps(rB, indent=2))

# Simple drift check: average absolute diff between predictions
import numpy as np
a = np.array(rA.get("predictions", []), dtype=float).flatten()
b = np.array(rB.get("predictions", []), dtype=float).flatten()
diff = float(np.mean(np.abs(a - b)))
print(json.dumps({"avg_abs_diff": diff}))
