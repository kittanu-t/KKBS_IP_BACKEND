from flask import Flask, jsonify, request
import joblib
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# 1. ตั้งค่า Path และโหลด Model
# ตรวจสอบให้แน่ใจว่าชื่อไฟล์ตรงกับที่คุณ Save ไว้
model_path = 'knn_minimal_model/obesity_model_minimal.joblib'

# นิยามลำดับ Class ให้ตรงกับ LabelEncoder (จาก Dataset Obesity)
class_names = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]

try:
    model = joblib.load(model_path)
    print(f"Successfully loaded minimal model from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# รายชื่อ Feature 6 ตัวที่โมเดลต้องการ
REQUIRED_FEATURES = ['Age', 'Height', 'Weight', 'SMOKE', 'CALC']

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Obesity Prediction API (6-Feature Version) is running!",
        "required_features": REQUIRED_FEATURES
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # 2. ตรวจสอบว่าส่งมาครบ 6 ตัวไหม
        missing = [f for f in REQUIRED_FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # 3. เตรียมข้อมูล (รับมาแค่ไหน ใช้แค่นั้น)
        # กรองเอาเฉพาะ 6 ตัวที่เราต้องการ เผื่อ User ส่งเกินมา
        input_data = {k: data[k] for k in REQUIRED_FEATURES}
        input_df = pd.DataFrame([input_data])

        # 4. ทำการ Prediction
        prediction_idx = model.predict(input_df)[0]
        prediction_label = class_names[int(prediction_idx)]

        # คำนวณความมั่นใจ (Confidence)
        try:
            prob = model.predict_proba(input_df).max()
        except:
            prob = None

        return jsonify({
            "status": "success",
            "prediction": prediction_label,
            "class_index": int(prediction_idx),
            "confidence": float(prob) if prob is not None else "N/A",
            "received_features": list(input_data.keys())
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# --- Error Handlers ---
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found", "code": 404}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal Server Error", "code": 500}), 500

if __name__ == '__main__':
    # รันที่ port 2569 ตามที่คุณต้องการ
    app.run(host='0.0.0.0', port=2569, debug=True)