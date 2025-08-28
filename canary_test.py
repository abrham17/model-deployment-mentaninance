import os, sys, json, requests
import numpy as np
TF_HOST =  "http://localhost:8501"
MODEL = "simple_classifier"
vA = 12
vB = 11
#http://localhost:8501/v1/models/simple_classifier/versions/12:predict
if not (vA and vB):
    print("Set VERSION_A and VERSION_B env vars to compare.")
    sys.exit(1)
urlA = f"{TF_HOST}/v1/models/{MODEL}/versions/{vA}:predict"
urlB = f"{TF_HOST}/v1/models/{MODEL}/versions/{vB}:predict"

payload = {"instances": [[1.0, 1.0]]}
rA = requests.post(urlA, json=payload['instances']).json()
rB = requests.post(urlB, json=payload['instances']).json()

print("A:", json.dumps(rA, indent=2))
print("B:", json.dumps(rB, indent=2))

# Simple drift check: average absolute diff between predictions
a = np.array(rA.get("predictions", []), dtype=float).flatten()
b = np.array(rB.get("predictions", []), dtype=float).flatten()
print(a , b)
diff = float(np.mean(np.abs(a - b)))
print(json.dumps({"avg_abs_diff": diff}))
