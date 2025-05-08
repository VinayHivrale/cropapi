from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model from the 'model' folder
model_path = "crop_model.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded successfully from '{model_path}'")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

@app.route("/")
def home():
    return "🌾 Crop Recommendation API is running!"

@app.route("/crop-predict", methods=["POST"])
def predict_crop():
    data = request.get_json()
    print("[DEBUG] Incoming request data:", data)

    try:
        # Extract and convert inputs
        features = [
            float(data["nitrogen"]),
            float(data["phosphorous"]),
            float(data["potassium"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"]),
        ]
        print(f"[DEBUG] Parsed features: {features}")

        if model is None:
            raise ValueError("Model not loaded")

        prediction = model.predict([features])[0]
        print(f"[DEBUG] Prediction result: {prediction}")

        return jsonify({"success": True, "prediction": prediction})

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Invalid input. Please provide all required numeric fields."
        })

if __name__ == "__main__":
    # Allow access from devices on the same network
    app.run(debug=True, host="0.0.0.0", port=5000)
