import os
import time
from ultralytics import YOLO
from flask import Flask, jsonify, request, render_template, redirect, url_for
import base64
import numpy as np
import cv2

# Load your YOLOv8 model
model = YOLO('detect/train/weights/best.pt')  # Thay thế bằng đường dẫn tới mô hình của bạn

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

# Store the last JSON response
last_response = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):
    image = cv2.imdecode(np.asarray(bytearray(image_stream.read()), dtype=np.uint8), cv2.IMREAD_COLOR)

    start_time = time.time()
    results = model.predict(image, classes=0, conf=0.5)
    duration = int((time.time() - start_time) * 1000)  # tính thời gian xử lý (ms)

    detection_infos = []
    for result in results:
        for r in result.boxes:
            bbox = r.xyxy[0].cpu().numpy().astype(int)
            detection_infos.append({
                "classname": result.names[int(r.cls)],
                "confidence": float(r.conf),
                "classno": int(r.cls),
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3]),
                "w": int(image.shape[1]),
                "h": int(image.shape[0])
            })

        # Encode the image with detections
        im_bgr = result.plot(conf=False)
        retval, buffer = cv2.imencode('.png', im_bgr)
        detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return detection_infos, detection_img_base64, duration, image

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict/img', methods=['POST'])
def predict():
    global last_response

    if 'file' not in request.files:
        return jsonify({"code": 1, "message": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"code": 1, "message": "No selected file"})

    if file and allowed_file(file.filename):
        detection_infos, detection_img_base64, duration, original_image = predict_on_image(file.stream)

        retval, buffer = cv2.imencode('.png', original_image)
        original_img_base64 = base64.b64encode(buffer).decode('utf-8')

        response = {
            "code": 0,
            "message": "OK",
            "num_detected_image": len(detection_infos),
            "num_image": 1,
            "duration": duration,
            "infos": detection_infos,
            "base64_image": detection_img_base64
        }

        last_response = response  # Store the response for later use

        return render_template('result.html', original_img_data=original_img_base64, detection_img_data=detection_img_base64)

    return jsonify({"code": 1, "message": "Invalid file type"})

@app.route('/predict/img/infor', methods=['GET'])
def imginfo():
    global last_response
    return jsonify(last_response)

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=True, host='0.0.0.0', port=6868)