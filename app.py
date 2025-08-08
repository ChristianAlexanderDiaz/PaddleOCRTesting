import os
import json
import gc
import threading
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import io
import base64
import traceback

app = Flask(__name__)

# Thread lock for OCR operations
ocr_lock = threading.Lock()

# Ultra-light initialization (sacrifices some accuracy for stability)
ocr = PaddleOCR(
    det_model_name='en_PP-OCRv3_det_slim',  # Slim detection model
    rec_model_name='en_PP-OCRv3_rec_slim',  # Slim recognition model
    use_angle_cls=False,
    lang='en',
    use_gpu=False,
    show_log=False,
    det_db_thresh=0.3,
    det_db_box_thresh=0.6,
    max_text_length=25,
    rec_batch_num=1,
    cpu_threads=1,
    enable_mkldnn=False,  # Disable MKL-DNN to save memory
    det_limit_side_len=640,  # Smaller image processing
    rec_image_shape="3, 32, 320"  # Even smaller recognition
)

print("PaddleOCR initialized!")

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "PaddleOCR API is ready. Send POST to /ocr with image"
    })

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    try:
        # Acquire lock to prevent concurrent OCR operations
        with ocr_lock:
            data = request.get_json()
            
            if 'image_path' in data:
                image_path = data['image_path']
                if not os.path.exists(image_path):
                    return jsonify({"error": "Image file not found"}), 404
                
                # Perform OCR
                result = ocr.ocr(image_path, cls=False)
                
            elif 'image_base64' in data:
                image_data = base64.b64decode(data['image_base64'])
                image = Image.open(io.BytesIO(image_data))
                
                # Save temporarily
                temp_path = '/tmp/temp_image.png'
                image.save(temp_path)
                
                # Perform OCR
                result = ocr.ocr(temp_path, cls=False)
                
                # Clean up temp file immediately
                try:
                    os.remove(temp_path)
                except:
                    pass
            else:
                return jsonify({"error": "No image provided"}), 400
            
            # Format results
            text_results = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:  # Ensure valid structure
                        text_results.append({
                            "text": line[1][0],
                            "confidence": float(line[1][1]),
                            "bbox": line[0]
                        })
            
            response = {
                "success": True,
                "results": text_results,
                "text": " ".join([r["text"] for r in text_results])
            }
            
            # Clean up
            del result
            cleanup_memory()
            
            return jsonify(response)
            
    except Exception as e:
        cleanup_memory()
        print(f"Error in OCR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/ocr-demo', methods=['GET'])
def ocr_demo():
    """Demo endpoint to OCR the test image in the repo"""
    try:
        # Acquire lock to prevent concurrent OCR operations
        with ocr_lock:
            test_image = "test_image.png"
            if not os.path.exists(test_image):
                return jsonify({"error": "test_image.png not found in repo"}), 404
            
            # Perform OCR
            result = ocr.ocr(test_image, cls=False)
            
            text_results = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:  # Ensure valid structure
                        text_results.append({
                            "text": line[1][0],
                            "confidence": float(line[1][1])
                        })
            
            response = {
                "success": True,
                "image": test_image,
                "results": text_results,
                "full_text": " ".join([r["text"] for r in text_results])
            }
            
            # Clean up
            del result
            cleanup_memory()
            
            return jsonify(response)
            
    except Exception as e:
        cleanup_memory()
        print(f"Error in OCR demo: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

# Periodic garbage collection
def periodic_cleanup():
    while True:
        threading.Event().wait(30)  # Every 30 seconds
        cleanup_memory()

# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)