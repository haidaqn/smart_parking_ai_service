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
# Đổi tên OUTPUT_FOLDER thành YOLO_OUTPUT_FOLDER và cấu hình đúng thư mục đầu ra của YOLO
YOLO_OUTPUT_FOLDER = os.path.abspath('runs/detect/predict')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(YOLO_OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

def extract_text_from_image(image):
    width, height = image.size
    image = image.resize((width // 2, height // 2), Image.LANCZOS)
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    post_data = (
        b"------WebKitFormBoundary\r\n"
        b"Content-Disposition: form-data; name=\"encoded_image\"; filename=\"image.jpg\"\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" +
        image_buffer.read() +
        b"\r\n------WebKitFormBoundary--\r\n"
    )
    
    headers = {
        'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundary',
        'Referer': 'https://lens.google.com/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    }
    
    response = req.post(f'https://lens.google.com/v3/upload?hl=en-VN&re=df&stcs={time.time_ns() // 10**6}&ep=subb', headers=headers, data=post_data)
    text = re.findall(r'\"vi\".*?]\]\]', response.text)
    return text

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        file.save(filepath)
        
        results = model.predict(source=filepath, save_txt=True, save=True, exist_ok=True)
        
        output_image_path = os.path.join(YOLO_OUTPUT_FOLDER, filename)
        output_files = os.listdir(YOLO_OUTPUT_FOLDER)
        if not output_files:
            return jsonify({'error': 'No output files found in the YOLO output folder'}), 500

        # Đảm bảo rằng tệp đầu ra tồn tại
        if not os.path.exists(output_image_path):
            return jsonify({'error': f'Output file not found at {output_image_path}'}), 500

        processed_image = Image.open(output_image_path)
        text_data = extract_text_from_image(processed_image)

        if text_data:
            return jsonify(text_data)
        else:
            return jsonify({"error": "No text found"}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
