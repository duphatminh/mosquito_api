import os
import time
from ultralytics import YOLO
from flask import Flask, jsonify, request, render_template
import base64
import numpy as np
import cv2

# Load your YOLOv8 model
model = YOLO('detect/train/weights/best.pt')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Store the last JSON response
last_response = None

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):
    image = cv2.imdecode(np.asarray(bytearray(image_stream.read()), dtype=np.uint8), cv2.IMREAD_COLOR) # Read the image

    start_time = time.time() # Start the timer
    results = model.predict(image, classes=0, conf=0.5) # Predict on the image
    duration = int((time.time() - start_time) * 1000)  # ms

    detection_infos = [] # Store the detection information
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
            }) # Store the detection information

        # Encode the image with detections
        im_bgr = result.plot(conf=False) # Plot the image
        retval, buffer = cv2.imencode('.png', im_bgr) # Encode the image
        detection_img_base64 = base64.b64encode(buffer).decode('utf-8') # Convert the image to base64

    return detection_infos, detection_img_base64, duration, image # Return the detection information, the base64 image, the duration and the original image

@app.route('/', methods=['GET']) # Home page
def home():
    return render_template('index.html')

@app.route('/predict/img', methods=['POST']) # Predict on image
def predict():
    global last_response

    if 'file' not in request.files: # Check if the post request has the file part
        return jsonify({"code": 1, "message": "No file part"}) # If no file part in the request

    file = request.files['file'] # If the user does not select a file, the browser submits an empty file without a filename

    if file.filename == '': # If the file is empty
        return jsonify({"code": 1, "message": "No selected file"}) # If no file selected

    if file and allowed_file(file.filename): # If the file is valid
        detection_infos, detection_img_base64, duration, original_image = predict_on_image(file.stream) # Predict on the image

        retval, buffer = cv2.imencode('.png', original_image) # Encode the original image
        original_img_base64 = base64.b64encode(buffer).decode('utf-8') # Convert the image to base64

        response = {
            "code": 0,
            "message": "OK",
            "num_detected_image": len(detection_infos),
            "num_image": 1,
            "duration": duration,
            "infos": detection_infos,
            "base64_image": detection_img_base64
        } # Create the response JSON

        last_response = response  # Store the response for later use

        return render_template('result.html', original_img_data=original_img_base64, detection_img_data=detection_img_base64) # Return the result page
        #return jsonify(last_response)
    return jsonify({"code": 1, "message": "Invalid file type"}) # If the file is invalid

@app.route('/predict/img/info', methods=['GET']) # Get the last response
def imginfo():
    global last_response
    return jsonify(last_response) # Return the last response

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development') # Set the FLASK_ENV to development
    app.run(debug=True, host='0.0.0.0', port=6868)