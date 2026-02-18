from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# โหลด pipeline
pipeline = joblib.load("obesity_model/pipeline.pkl")

FEATURE_ORDER = [
    "Sex",
    "Age",
    "Height",
    "Overweight_Obese_Family",
    "Calculation_of_Calorie_Intake",
    "Frequency_of_Consuming_Vegetables",
    "Number_of_Main_Meals_Daily",
    "Food_Intake_Between_Meals",
    "Smoking",
    "Liquid_Intake_Daily",
    "Physical_Excercise",
    "Schedule_Dedicated_to_Technology",
    "Type_of_Transportation_Used"
]


@app.route("/", methods=["GET"])
def home():
    return "Obesity Prediction API Ready"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        features = [data[f] for f in FEATURE_ORDER]
        X = np.array(features).reshape(1, -1)

        prediction = pipeline.predict(X)[0]

        confidence = None
        if hasattr(pipeline, "predict_proba"):
            confidence = float(pipeline.predict_proba(X).max())

        return jsonify({
            "prediction": str(prediction),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2569, debug=True)
