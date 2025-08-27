from flask import Flask, request, render_template, jsonify
import requests, os, json

TF_HOST = os.environ.get("TF_HOST", "http://localhost:8501")
MODEL_NAME = os.environ.get("MODEL_NAME", "simple_classifier")
PRED_URL = f"{TF_HOST}/v1/models/{MODEL_NAME}:predict"

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        body = request.get_json()
        instances = body.get("instances")
        if instances is None:
            return jsonify({"error": "Provide JSON with 'instances': [[x1,x2], ...]"}), 400
    else:
        try:
            x1 = float(request.form.get("x1"))
            x2 = float(request.form.get("x2"))
            instances = [[x1, x2]]
        except Exception:
            return render_template("index.html", error="Enter valid numbers for x1 and x2"), 400
    resp = requests.post(PRED_URL, json={"instances": instances}, timeout=5)
    if not resp.ok:
        return jsonify({"error": "TF Serving error", "details": resp.text}), 500
    data = resp.json()
    return (render_template("index.html", result=data, x1=instances[0][0], x2=instances[0][1])
            if not request.is_json else jsonify(data))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
