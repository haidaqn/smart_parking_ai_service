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
OUTPUT_FOLDER = os.path.abspath('./output')
YOLO_OUTPUT_FOLDER = os.path.abspath('./runs/detect/predict')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(YOLO_OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)
def extract_text_from_image(image):
    # Resize image for faster processing
    width, height = image.size
    image = image.resize((width // 2, height // 2), Image.LANCZOS)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

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
        'Content-Length': str(len(post_data)),
        'Referer': 'https://lens.google.com/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    }

    response = req.post(f'https://lens.google.com/v3/upload?hl=en-VN&re=df&stcs={time.time_ns() // 10**6}&ep=subb', headers=headers, data=post_data)
    
    text = re.findall(r'\"vi\".*?]\]\]', response.text)
    print(text)
    if not text:
        return None
    
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
        # Save the uploaded file to the UPLOAD_FOLDER
        file.save(filepath)
        
        # Predict and save the results in the YOLO_OUTPUT_FOLDER
        results = model.predict(source=filepath, save_txt=True, save=True, exist_ok=True)
        
        # Find all output files in the YOLO_OUTPUT_FOLDER
        output_files = os.listdir(YOLO_OUTPUT_FOLDER)
        if not output_files:
            return jsonify({'error': 'No output files found in the YOLO output folder'}), 500

        # Find the latest output file based on creation time
        latest_output_file = max([os.path.join(YOLO_OUTPUT_FOLDER, f) for f in output_files], key=os.path.getctime)

        if not os.path.exists(latest_output_file):
            return jsonify({'error': f'Output file not found at {latest_output_file}'}), 500

        # Set output file name
        output_filename = f"processed_{filename}"
        filePathOutput = os.path.join(OUTPUT_FOLDER, output_filename)

        # Save the processed image to the OUTPUT_FOLDER
        processed_image = Image.open(latest_output_file)
        processed_image.save(filePathOutput)

        # Open the original image for text analysis
        img = Image.open(filepath)
        text_data = extract_text_from_image(img)

        if text_data:
            return jsonify({"text": text_data, "output_image": filePathOutput})
        else:
            return jsonify({"error": "No text found", "output_image": filePathOutput}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
