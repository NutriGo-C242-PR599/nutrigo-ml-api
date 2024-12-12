from flask import Flask, request, jsonify, url_for
from app_utility import *
import time
import cv2
import numpy as np


interpreter = load_model('model/detect.tflite')
interpreter2 = load_model('model/v2/detect.tflite')

labels2_path = 'model/v2/labels.txt'

with open("words.txt", "r", encoding="utf-8") as file:
    words = [line.strip() for line in file]
    
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Server is running'
    })

@app.route('/detect', methods=['POST'])
def detect():
    if request.files.get('image') is None:
        return jsonify({
            'status': 'error',
            'message': 'image not found'
        })
    
    image = request.files['image']
    # Convert the file to a NumPy array
    file_bytes = np.frombuffer(image.read(), np.uint8)

    # Decode the NumPy array to an OpenCV image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    detections, rois, image_path, raw_image = tflite_detect_image(interpreter2, image, labels2_path)

    print(detections)
    ocr_result = ''
    data = []
    for idx, detection in enumerate(detections):
        if detection[0] != 'nutrition_box':
            continue
        
        _, _, xmin, ymin, xmax, ymax = detection
        width = xmax - xmin
        height = ymax - ymin
        x = idx+1
        y = xmin
        w = width
        h = height

        ocr_result = ocr_on_roi(image=raw_image, xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin)

        
        for word in words:
            lines = find_line_with_word(ocr_result, word)
            print(f"\nLines containing '{word}':")
            # data.append(lines)
            for line in lines:
                data.append({'word': word, 'data': line})
    image_url = url_for('static', filename=image_path)
    
    image_url = request.host_url + image_url
    return jsonify({
        'status': 'success',
        'message': 'image successfully detected',
        'ocr_result': ocr_result,
        'data': data,
        'image_url': image_url
    })