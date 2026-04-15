import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
import io
import base64
from PIL import Image as PILImage
# 统一导入
from ai_assistant import smart_correction, incomplete_completion, generate_report, polish_result

app = Flask(__name__)

# 模型加载
MODEL_PATH = 'best.pt'
print(f"⏳ 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

def apply_clahe_preprocessing(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb, enhanced

# --- 路由定义区 (必须在 app.run 之前) ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "未上传图片"}), 400

    file = request.files['image']
    conf_threshold = float(request.form.get('conf', 0.25))

    try:
        img_bytes = file.read()
        processed_rgb, _ = apply_clahe_preprocessing(img_bytes)
        if processed_rgb is None:
            return jsonify({"status": "error", "message": "图片解码失败"}), 400

        # 进行 YOLO 识别
        results = model.predict(source=processed_rgb, conf=conf_threshold, save=False)
        res = results[0]
        predictions = []
        if res.boxes is not None:
            for box in res.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                char_name = model.names[class_id]
                xyxy = box.xyxy[0].tolist()
                predictions.append({
                    "char": char_name,
                    "confidence": round(conf, 4),
                    "bbox": [int(x) for x in xyxy]
                })

        # 按置信度排序
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        # 生成增强后的图像 base64
        enhanced_pil = PILImage.fromarray(processed_rgb)
        buffered = io.BytesIO()
        enhanced_pil.save(buffered, format="JPEG")
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "status": "success",
            "count": len(predictions),
            "results": predictions,
            "enhanced_image_base64": enhanced_base64
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/smart_correction', methods=['POST'])
def api_smart_correction():
    data = request.json
    result = smart_correction(data.get('char_sequence'), data.get('low_conf_indices', []))
    return jsonify(result)

@app.route('/incomplete_completion', methods=['POST'])
def api_incomplete_completion():
    data = request.json
    result = incomplete_completion(data.get('char_sequence'))
    return jsonify(result)

@app.route('/generate_report', methods=['POST'])
def api_generate_report():
    data = request.json
    result = generate_report(data.get('char_sequence'), data.get('detections'))
    return jsonify(result)

@app.route('/polish', methods=['POST'])
def api_polish():
    data = request.json
    raw_data = data.get('raw_data')
    task_type = data.get('task_type')
    result = polish_result(raw_data, task_type)
    return jsonify({"polished_text": result})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# --- 启动区 (必须在文件最后) ---
if __name__ == '__main__':
    # 注意：debug=True 在开发阶段很有用，它会自动重载代码
    app.run(host='0.0.0.0', port=7000, debug=True)