from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import io
import os
import re
import requests as req
import time

app = Flask(__name__)

MODEL_PATH = 'best.pt'
UPLOAD_FOLDER = os.path.abspath('./uploads')
YOLO_OUTPUT_FOLDER = os.path.abspath('./runs/detect/predict/labels')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

def read_bounding_boxes(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                bounding_boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return bounding_boxes

def crop_image(image, bounding_box):
    image_width, image_height = image.size
    x_center, y_center, width, height = bounding_box
    left = int((x_center - width / 2) * image_width)
    right = int((x_center + width / 2) * image_width)
    top = int((y_center - height / 2) * image_height)
    bottom = int((y_center + height / 2) * image_height)
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

def extract_text_from_image(image):
    image_io = io.BytesIO()
    image.save(image_io, format='JPEG')
    image_io.seek(0)
    image_data = image_io.read()

    post_data = b"------WebKitFormBoundary\r\nContent-Disposition: form-data; name=\"encoded_image\"; filename=\"download.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n" + \
        image_data + b"\r\n------WebKitFormBoundary--\r\n"
    headers = {
        'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundary',
        'Content-Length': str(len(post_data)),
        'Referer': 'https://lens.google.com/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    }

    response = req.post(
        f'https://lens.google.com/v3/upload?hl=en-VN&re=df&stcs={time.time_ns() // 10**6}&ep=subb', headers=headers, data=post_data)

    pattern = r'"([^"]*)",\[\[\[(.*?)\]\]\]'
    
    match = re.findall(pattern, response.text)
    
    extracted_text = ' '.join(re.findall(
        r'\"(.*?)\"]]', match[0][1])) if match else ''

    return extracted_text

@app.route('/', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        file.save(filepath)
        results = model.predict(source=filepath, save_txt=True, save=False, exist_ok=True)
        
        label_file_path = os.path.join(YOLO_OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.txt")
        if not os.path.exists(label_file_path):
            return jsonify({'error': f'Label file not found at {label_file_path}'}), 500

        bounding_boxes = read_bounding_boxes(label_file_path)
        if not bounding_boxes:
            return jsonify({'error': 'No bounding boxes found in the label file'}), 400
        
        image = Image.open(filepath)
        cropped_image = crop_image(image, bounding_boxes[0])  # Chỉ lấy bounding box đầu tiên
        
        text_data = extract_text_from_image(cropped_image)

        return jsonify({"text": text_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(label_file_path):
            os.remove(label_file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)