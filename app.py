import os
import json
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import io
import base64

app = Flask(__name__)

# ULTRA-LIGHTWEIGHT VERSION - Uses mobile models
ocr = PaddleOCR(
    det_model_dir='https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.tar',
    rec_model_dir='https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.tar',
    use_angle_cls=False,  # Disable angle classification
    lang='en',
    use_gpu=False,
    det_db_thresh=0.3,
    det_db_box_thresh=0.6,
    det_db_unclip_ratio=1.5,
    max_text_length=25,
    rec_batch_num=6,  # Lower batch size
    drop_score=0.5,
    use_space_char=True,
    show_log=False
)
print("PaddleOCR initialized!")

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "PaddleOCR API is ready. Send POST to /ocr with image"
    })

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    try:
        # Get image from request
        data = request.get_json()
        
        # Option 1: Process local image file
        if 'image_path' in data:
            image_path = data['image_path']
            if not os.path.exists(image_path):
                return jsonify({"error": "Image file not found"}), 404
            
            # Perform OCR
            result = ocr.ocr(image_path, cls=False)
            
        # Option 2: Process base64 encoded image
        elif 'image_base64' in data:
            image_data = base64.b64decode(data['image_base64'])
            image = Image.open(io.BytesIO(image_data))
            
            # Save temporarily (PaddleOCR works better with file paths)
            temp_path = '/tmp/temp_image.png'
            image.save(temp_path)
            
            # Perform OCR
            result = ocr.ocr(temp_path, cls=False)
            
            # Clean up
            os.remove(temp_path)
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Format results
        text_results = []
        if result and result[0]:
            for line in result[0]:
                text_results.append({
                    "text": line[1][0],
                    "confidence": float(line[1][1]),
                    "bbox": line[0]
                })
        
        return jsonify({
            "success": True,
            "results": text_results,
            "text": " ".join([r["text"] for r in text_results])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ocr-demo', methods=['GET'])
def ocr_demo():
    """Demo endpoint to OCR the test image in the repo"""
    try:
        test_image = "test_image.png"
        if not os.path.exists(test_image):
            return jsonify({"error": "test_image.png not found in repo"}), 404
        
        result = ocr.ocr(test_image, cls=False)
        
        text_results = []
        if result and result[0]:
            for line in result[0]:
                text_results.append({
                    "text": line[1][0],
                    "confidence": float(line[1][1])
                })
        
        return jsonify({
            "success": True,
            "image": test_image,
            "results": text_results,
            "full_text": " ".join([r["text"] for r in text_results])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)