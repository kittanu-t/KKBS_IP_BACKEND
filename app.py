import os
import math
import base64
import logging
import re
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

OBESITY_MODEL_PATH = 'knn_minimal_model/obesity_model_minimal.joblib'

CLASS_NAMES = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]
REQUIRED_FEATURES = ['Age', 'Height', 'Weight', 'SMOKE', 'CALC']

# --- Obesity Model ---
def load_obesity_model():
    if os.path.exists(OBESITY_MODEL_PATH):
        try:
            model = joblib.load(OBESITY_MODEL_PATH)
            logger.info(f"Model loaded successfully from {OBESITY_MODEL_PATH}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return None

obesity_model = load_obesity_model()

# -------- Check Status --------
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "obesity_model_loaded": obesity_model is not None,
        "required_features": REQUIRED_FEATURES
    })

#-------- Obesity Prediction Endpoint --------
@app.route("/predict", methods=["POST"])
def predict():
    if not obesity_model:
        return jsonify({"error": "Obesity model is not available"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400

    try:
        input_df = pd.DataFrame([{f: data[f] for f in REQUIRED_FEATURES}])
        
        prediction_idx = obesity_model.predict(input_df)[0]
        label = CLASS_NAMES[int(prediction_idx)]
        
        confidence = "N/A"
        if hasattr(obesity_model, "predict_proba"):
            confidence = float(np.max(obesity_model.predict_proba(input_df)))

        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400





# --- Helper Functions ---
def decode_base64_to_cv2(base64_string):
    # ตัด metadata prefix ออก (data:image/png;base64,...)
    img_data = re.sub('^data:image/.+;base64,', '', base64_string)
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    # ถ้ามี Alpha channel ให้แปลงเป็นพื้นหลังขาว
    if img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        rgb_channels = img[:, :, :3]
        white_bg = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
        img = rgb_channels * alpha_factor + white_bg * (1 - alpha_factor)
        img = img.astype(np.uint8)
    
    return img

def get_angle(x1, y1, x2, y2):
    # คำนวณมุมในหน่วยองศา (0 คือเลข 3, วนตามเข็ม)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle % 360

# -------- Analyze Clock Endpoint --------
@app.route("/analyze-clock", methods=["POST"])
def analyze_clock():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"status": "fail", "message": "No image data"}), 400

    try:
        # 1. Decode Image
        img_str = re.sub('^data:image/.+;base64,', '', data['image'])
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # จัดการ Alpha Channel (แปลงเป็นพื้นหลังขาว)
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            rgb = img[:, :, :3]
            white_bg = np.ones_like(rgb) * 255
            mask = alpha[:, :, np.newaxis] / 255.0
            img = (rgb * mask + white_bg * (1 - mask)).astype(np.uint8)
        
        debug_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Find Circle (หน้าปัด)
        # ใช้ Blur เพื่อลด noise ของการวาดด้วยมือ
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=50, param2=30, minRadius=60, maxRadius=160)

        if circles is None:
            return jsonify({"status": "fail", "message": "ไม่พบรูปวาดวงกลม (หน้าปัด) กรุณาวาดให้ชัดเจนขึ้น"}), 400

        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0, 0]
        cv2.circle(debug_img, (cx, cy), r, (0, 255, 0), 2) # วาดวงกลมเขียวใน debug

        # 3. Find Lines (เข็มนาฬิกา)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=25, maxLineGap=15)

        detected_angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # เช็คว่าเส้นเชื่อมโยงกับจุดศูนย์กลางหรือไม่ (ระยะห่างไม่เกิน 30% ของรัศมี)
                d1 = math.sqrt((x1-cx)**2 + (y1-cy)**2)
                d2 = math.sqrt((x2-cx)**2 + (y2-cy)**2)
                
                if d1 < r*0.3 or d2 < r*0.3:
                    # หาจุดปลายที่ไกลจากจุดศูนย์กลาง
                    far_x, far_y = (x2, y2) if d1 < d2 else (x1, y1)
                    # คำนวณมุม (0 deg = 3 o'clock, clockwise)
                    angle = math.degrees(math.atan2(far_y - cy, far_x - cx)) % 360
                    
                    # กรองมุมที่ซ้ำกัน (Grouping)
                    if not any(abs(angle - a) < 20 for a in detected_angles):
                        detected_angles.append(angle)
                        cv2.line(debug_img, (cx, cy), (int(far_x), int(far_y)), (255, 0, 0), 3)

        # 4. 11:10 Logic Calculation
        # เลข 2 (นาที) -> ประมาณ 330 องศา
        # เลข 11 (ชั่วโมง) -> 240 องศา (แต่ 11:10 เข็มชั่วโมงต้องขยับไปทางเลข 12 เล็กน้อย -> 245 องศา)
        targets = [330, 245]
        tolerance = 20 # ให้ความเพี้ยนได้ +/- 20 องศา
        
        matches = 0
        for t in targets:
            for d in detected_angles:
                diff = abs(t - d)
                if diff > 180: diff = 360 - diff # จัดการการวนรอบวงกลม
                if diff <= tolerance:
                    matches += 1
                    break

        # 5. Interpretation
        is_impaired = 0 if matches >= 2 else 1
        msg = "ปกติ (วาดตำแหน่งเข็มได้ถูกต้อง)" if is_impaired == 0 else "พบความผิดปกติ: ตำแหน่งเข็มไม่ถูกต้องหรือหาเข็มไม่พบ"
        
        if len(detected_angles) < 2:
            msg = "ตรวจไม่พบเข็มนาฬิกา 2 เข็มที่ลากออกจากจุดศูนย์กลาง"

        # Encode debug image
        _, buffer = cv2.imencode('.png', debug_img)
        debug_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": "success",
            "prediction": is_impaired,
            "interpretation": msg,
            "confidence": 0.9 if matches >= 2 else 0.5,
            "debug_image": debug_b64
        })

    except Exception as e:
        return jsonify({"status": "fail", "message": str(e)}), 500

if __name__ == '__main__':
    # แนะนำให้ปิด debug=True เมื่อขึ้น Production
    port = int(os.environ.get("PORT", 2569))
    app.run(host='0.0.0.0', port=port)