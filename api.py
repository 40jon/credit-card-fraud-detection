from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os


app = Flask(__name__)
CORS(app)  # allow browser requests from your frontend

# Paths to your existing model files in the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "fraud_scaler.pkl")

# Load the model and scaler once when the server starts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
        "features": [v1, v2, v3, ..., vN]
    }
    The order must match the columns used during training.
    """
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' list"}), 400

    # Convert to 2D array for scikit-learn
    features = np.array(data["features"]).reshape(1, -1)

    # Apply same scaling as during training
    features_scaled = scaler.transform(features)

    # Predict
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0].tolist()

    return jsonify({
        "prediction": int(pred),  # 1 = fraud, 0 = not fraud
        "fraud_probability": proba[1]  # probability of class 1
    })


@app.route("/", methods=["GET"])
def health():
    return "Fraud detection API is running."


if __name__ == "__main__":
    # Run on http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)