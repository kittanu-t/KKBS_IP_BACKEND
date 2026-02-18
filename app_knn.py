from flask import Flask, jsonify, request
import joblib
import pandas as pd
from flask_cors import CORS
import os

# 1. โหลด Model Pipeline ที่บันทึกไว้
# ปรับ path ให้ตรงกับโฟลเดอร์ของคุณ
model_path = os.path.join('knn_model', 'knn_obesity_pipeline.joblib')

try:
    model = joblib.load(model_path)
    print(f"Successfully loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)

# ตั้งค่า CORS เพื่อให้ Frontend เรียกใช้งานได้
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route("/", methods=["GET"])
def mainRoute():
    return "Obesity Prediction Flask API (KNN Pipeline) is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    try:
        # รับข้อมูล JSON จาก Request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # แปลงข้อมูลเป็น DataFrame (เนื่องจาก Pipeline รับ Input เป็น DataFrame)
        # ข้อมูลควรมี Key ตรงกับชื่อ Column ตอน Train (เช่น Gender, Age, Height, Weight, ...)
        input_df = pd.DataFrame([data])

        # ทำการ Prediction ด้วย Pipeline (ซึ่งจะทำ Scaling/Encoding ให้อัตโนมัติ)
        prediction = model.predict(input_df)[0]
        
        # (Optional) หากต้องการค่าความน่าเชื่อถือ (Probability)
        try:
            prob = model.predict_proba(input_df).max()
        except:
            prob = None

        return jsonify({
            "status": "success",
            "prediction": str(prediction),
            "confidence": float(prob) if prob is not None else "N/A"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route("/health", methods=["GET"])
def check_health():
    return jsonify({
        "code": 200,
        "model_loaded": model is not None
    })

# --- Error Handlers (คงไว้ตามตัวอย่างเดิม) ---

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "ส่งข้อมูลมาไม่ถูก Pattern", "code": 400}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "ไม่รู้จัก Route ที่เรียกใช้ครับ", "code": 404}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method ไม่ถูกต้องครับ", "code": 405}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal Server Error", "code": 500}), 500

if __name__ == '__main__':
    # รันที่ port 2569 ตามที่คุณต้องการ
    app.run(host='0.0.0.0', port=2569, debug=True)