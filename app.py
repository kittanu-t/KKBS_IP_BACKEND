import os
import math
import base64
import logging
from io import BytesIO

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

MODEL_PATH = 'knn_minimal_model/obesity_model_minimal.joblib'
CLASS_NAMES = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]
# ปรับให้ตรงกับที่ใช้จริง (ในโค้ดเดิมมี 5 ตัว แต่ comment บอก 6 ผมยึดตาม list นะครับ)
REQUIRED_FEATURES = ['Age', 'Height', 'Weight', 'SMOKE', 'CALC']

# --- Model Loader ---
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return None

model = load_model()

# --- Helper Functions for Image Processing ---

def decode_base64_image(data_url):
    try:
        encoded = data_url.split(",")[1] if "," in data_url else data_url
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        # แปลงเป็น BGR สำหรับ OpenCV
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return None

def detect_clock_hands(image):
    # 1. Pre-processing: แปลงเป็นขาวดำและลด Noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # ใช้ Threshold แบบ Adaptive เพื่อแยกเส้นออกจากพื้นหลังได้ดีขึ้น
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 2. หาจุดศูนย์กลางของนาฬิกา (ใช้ Contours แทนการเดากึ่งกลางภาพ)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, (image.shape[1]//2, image.shape[0]//2)

    # หาพื้นที่ที่ใหญ่ที่สุด (ซึ่งก็คือตัววงกลมนาฬิกา)
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return None, (image.shape[1]//2, image.shape[0]//2)
    
    # คำนวณจุดศูนย์กลาง (Centroid) ของวงกลมที่วาด
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    center = (center_x, center_y)

    # 3. เตรียมเส้นสำหรับการตรวจจับ (Dilation)
    # ขยายเส้นให้หนาขึ้นเล็กน้อย เพื่อเชื่อมส่วนที่วาดขาดๆ หายๆ
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    edges = cv2.Canny(dilated, 50, 150)

    # 4. ตรวจจับเส้นตรง (HoughLinesP)
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,      # ปรับลงเพื่อให้เจอเส้นง่ายขึ้น
        minLineLength=25,  # รองรับเข็มที่สั้นลง
        maxLineGap=20      # ยอมรับช่องว่างระหว่างเส้นได้มากขึ้น
    )

    if lines is None:
        return None, center

    detected_hands = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # หาจุดที่อยู่ห่างจากจุดศูนย์กลางนาฬิกามากที่สุด (คือหัวเข็ม)
        dist1 = math.hypot(x1 - center_x, y1 - center_y)
        dist2 = math.hypot(x2 - center_x, y2 - center_y)
        
        far_point = (x2, y2) if dist2 > dist1 else (x1, y1)
        near_dist = min(dist1, dist2)

        # เงื่อนไข: เส้นต้องเริ่มใกล้จุดศูนย์กลาง (ไม่เกิน 60px) 
        # เพื่อป้องกันไม่ให้ไปจับเอาเส้นขอบวงกลมมาเป็นเข็ม
        if near_dist < 60:
            length = math.hypot(far_point[0] - center_x, far_point[1] - center_y)
            
            # คำนวณมุม (0 องศาที่เลข 12)
            angle = math.degrees(math.atan2(far_point[0] - center_x, center_y - far_point[1]))
            angle = (angle + 360) % 360
            
            detected_hands.append((length, angle))

    if len(detected_hands) < 2:
        return None, center

    # เรียงลำดับตามความยาว: ยาวสุด=เข็มนาที, รองลงมา=เข็มชั่วโมง
    detected_hands.sort(key=lambda x: x[0], reverse=True)
    
    # ป้องกันการจับเส้นซ้ำที่มุมใกล้กันเกินไป
    unique_hands = [detected_hands[0]]
    for h in detected_hands[1:]:
        if abs(h[1] - unique_hands[0][1]) > 20: # ถ้ามุมต่างกันเกิน 20 องศา ถือว่าเป็นคนละเข็ม
            unique_hands.append(h)
            break

    if len(unique_hands) < 2:
        return None, center

    return (unique_hands[0][1], unique_hands[1][1]), center
# --- API Routes ---

def analyze_clock_with_debug(image):
    # --- ส่วนเดิมของ detect_clock_hands แต่เพิ่มการวาดรูป ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # สร้างภาพสำหรับ Debug (Copy จากภาพต้นฉบับ)
    debug_vis = image.copy()
    
    # 1. หาจุดศูนย์กลาง
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, debug_vis

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0: return None, debug_vis
    
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    
    # วาดจุดศูนย์กลาง (สีแดง)
    cv2.circle(debug_vis, (cx, cy), 7, (0, 0, 255), -1)
    cv2.putText(debug_vis, "Center", (cx-20, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 2. ตรวจจับเส้น
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    edges = cv2.Canny(dilated, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=25, maxLineGap=20)

    detected_hands = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dist1 = math.hypot(x1 - cx, y1 - cy)
            dist2 = math.hypot(x2 - cx, y2 - cy)
            
            # วาดเส้นสีเทาบางๆ ทุกเส้นที่ AI เจอ (เพื่อดูว่ามันเห็นอะไรบ้าง)
            cv2.line(debug_vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

            far_point = (x2, y2) if dist2 > dist1 else (x1, y1)
            if min(dist1, dist2) < 60:
                length = math.hypot(far_point[0] - cx, far_point[1] - cy)
                angle = (math.degrees(math.atan2(far_point[0] - cx, cy - far_point[1])) + 360) % 360
                detected_hands.append({"length": length, "angle": angle, "point": far_point})

    # 3. คัดเลือกและวาดเข็มที่ตัดสินใจเลือก
    detected_hands.sort(key=lambda x: x['length'], reverse=True)
    
    final_hands = []
    if len(detected_hands) > 0:
        # เข็มที่ 1 (นาที - สีเขียว)
        h1 = detected_hands[0]
        final_hands.append(h1)
        cv2.line(debug_vis, (cx, cy), h1['point'], (0, 255, 0), 3)
        cv2.putText(debug_vis, f"Min: {int(h1['angle'])}deg", h1['point'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # ค้นหาเข็มที่ 2 (ชั่วโมง - สีฟ้า) ที่มุมต่างจากเข็มแรก
        for h in detected_hands[1:]:
            if abs(h['angle'] - h1['angle']) > 25:
                final_hands.append(h)
                cv2.line(debug_vis, (cx, cy), h['point'], (255, 165, 0), 3)
                cv2.putText(debug_vis, f"Hour: {int(h['angle'])}deg", h['point'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                break

    return final_hands, debug_vis

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "required_features": REQUIRED_FEATURES
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model is not available"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    # ตรวจสอบ Feature
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400

    try:
        # เตรียม DataFrame
        input_df = pd.DataFrame([{f: data[f] for f in REQUIRED_FEATURES}])
        
        prediction_idx = model.predict(input_df)[0]
        label = CLASS_NAMES[int(prediction_idx)]
        
        # หา Confidence
        confidence = "N/A"
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(input_df)))

        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/analyze-clock", methods=["POST"])
@app.route("/analyze-clock", methods=["POST"])
def analyze_clock():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data"}), 400

    img = decode_base64_image(data['image'])
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
    
    # ใช้ฟังก์ชัน Debug ตัวเดียวที่คืนค่าครบทั้งข้อมูลและรูปภาพ
    hands_list, debug_img = analyze_clock_with_debug(img)
    
    # แปลงภาพ Debug เป็น Base64
    _, buffer = cv2.imencode('.png', debug_img)
    debug_b64 = base64.b64encode(buffer).decode('utf-8')

    # ตรวจสอบว่าเจอเข็มครบ 2 เข็มไหม
    if len(hands_list) < 2:
        return jsonify({
            "status": "fail",
            "score": 0,
            "debug_image": f"data:image/png;base64,{debug_b64}",
            "interpretation": "ไม่สามารถตรวจพบเข็มนาฬิกาได้ครบถ้วน",
            "details": "Detected less than 2 hands"
        })

    # ดึงค่าองศาจาก List ของ Dictionary
    # hands_list[0] คือเข็มนาที (ยาวสุด), hands_list[1] คือเข็มชั่วโมง
    min_angle = hands_list[0]['angle']
    hour_angle = hands_list[1]['angle']
    
    score = 0
    score += 1 # ตรวจพบเข็มครบ
    
    if abs(min_angle - 0) < 15 or abs(min_angle - 360) < 15:
        score += 1
    if abs(hour_angle - 210) < 20:
        score += 1

    interpretations = {
        3: "ปกติ (Normal)",
        2: "มีความผิดปกติเล็กน้อย (Mild Impairment)",
        1: "ควรปรึกษาแพทย์ (Significant Impairment)",
        0: "ควรปรึกษาแพทย์ (Significant Impairment)"
    }

    return jsonify({
        "status": "success",
        "score": score,
        "interpretation": interpretations.get(score),
        "debug_image": f"data:image/png;base64,{debug_b64}",
        "details": {
            "hands_count": len(hands_list),
            "minute_angle": round(min_angle, 2),
            "hour_angle": round(hour_angle, 2)
        }
    })

if __name__ == '__main__':
    # แนะนำให้ปิด debug=True เมื่อขึ้น Production
    port = int(os.environ.get("PORT", 2569))
    app.run(host='0.0.0.0', port=port)