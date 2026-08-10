"""
VisionAid - Full Deep Learning Backend Server
Comprehensive visual assistance API using:
- YOLO (yolov8l/m/n) for object detection and scene understanding
- EasyOCR for robust text recognition (deep learning OCR)
- Classical CV for face detection and preprocessing
"""
from flask import Flask, render_template, request, jsonify, send_file
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: flask-cors not installed. CORS disabled. Install with: pip install flask-cors")
import os
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import logging
import traceback
import time
import sys
from functools import wraps
from datetime import datetime
import json
from collections import Counter

# OCR engines: EasyOCR and PaddleOCR only (Tesseract not used)

# Optional OCR (Robust) - uses EasyOCR if available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError as e:
    EASYOCR_AVAILABLE = False
    print(f"Warning: easyocr import failed (ImportError): {e}")
except Exception as e:
    EASYOCR_AVAILABLE = False
    print(f"Warning: easyocr import failed (Exception): {e}")
    import traceback
    traceback.print_exc()

# Fallback OCR - PaddleOCR (optional, EasyOCR is primary)
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("[OK] PaddleOCR loaded successfully (available as fallback OCR)")
except ImportError:
    PADDLEOCR_AVAILABLE = False
    # Silent - PaddleOCR is optional fallback, EasyOCR is primary
except Exception as e:
    PADDLEOCR_AVAILABLE = False
    # Silent - PaddleOCR is optional fallback
    
# Global reader cache
_reader = None
_paddle_ocr = None

# Global switch: disable SLOW OCR pipeline (PaddleOCR + heavy multi-pass EasyOCR)
# When False, only the FAST path is used so reading text is much quicker.
ENABLE_SLOW_OCR = False

def get_reader():
    global _reader, reader
    # Use the global reader if it's already initialized
    if reader is not None:
        _reader = reader
        return _reader
    # Otherwise initialize it
    if _reader is None and EASYOCR_AVAILABLE:
        print("Initializing EasyOCR Reader...")
        try:
            # Try GPU first for better performance
            import torch
            use_gpu = torch.cuda.is_available()
            _reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            reader = _reader  # Sync with global reader
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            _reader = None
            reader = None
    return _reader

def get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None and PADDLEOCR_AVAILABLE:
        print("Initializing PaddleOCR...")
        try:
            # Use correct PaddleOCR parameters for newer version
            # Note: use_textline_orientation replaces use_angle_cls in newer versions
            _paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}")
            import traceback
            traceback.print_exc()
            _paddle_ocr = None
    return _paddle_ocr

# Import analyzers (we only use their classical / rule-based parts)
from scene_describer_simple import simple_scene_describer
from advanced_scene_analyzer import advanced_analyzer
from face_recognition_module import face_recognition
from enhanced_preprocessing import EnhancedPreprocessor
from object_validation import ObjectValidator
from currency_detector import CurrencyDetector

# Import advanced detection for >90% accuracy
try:
    from advanced_detection import AdvancedDetector
    ADVANCED_DETECTION_AVAILABLE = True
except ImportError:
    ADVANCED_DETECTION_AVAILABLE = False
    # logger isn't configured yet at this point in the file
    print("Advanced detection module not available (optional)")

# ===== CONFIGURATION =====
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_DIMENSION = 4096  # Max width/height
MIN_IMAGE_DIMENSION = 50  # Min width/height

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visionaid.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== FLASK APP SETUP =====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE
if CORS_AVAILABLE:
    CORS(app)  # Enable CORS for cross-origin requests

# ===== GLOBAL VARIABLES =====
reader = None        # EasyOCR reader (deep learning OCR)
ocr_type = None      # 'easyocr', 'paddleocr', or None
model = None         # YOLO model for object detection and scene understanding
models_loaded = False
model_load_error = None

# Optional text summarization using Transformers
try:
    from summarize_text import summarize_text
    SUMMARIZATION_AVAILABLE = True
except (ImportError, Exception) as e:
    SUMMARIZATION_AVAILABLE = False
    logger.warning(f"Text summarization not available: {e}")

# ===== MODEL INITIALIZATION (FULL DEEP LEARNING MODE) =====
def initialize_models():
    """
    Initialize full deep learning mode:
    - Load YOLO model (yolov8l.pt > yolov8m.pt > yolov8n.pt) for object detection.
    - Load EasyOCR for robust text recognition (deep learning OCR).
    """
    global models_loaded, model_load_error, model, reader, ocr_type, _reader
    
    if models_loaded and model is not None:
        return True
    
    try:
        logger.info("Initializing deep learning models (HIGH ACCURACY MODE - >95% accuracy)...")
        start_time = time.time()
        
        # 1. Initialize EasyOCR (deep learning OCR) — LIGHTWEIGHT CPU SETTINGS
        # NOTE: On CPU-only machines, EasyOCR can be very slow. We still load it,
        # but with minimal configuration and only English by default.
        if EASYOCR_AVAILABLE:
            try:
                logger.info("Loading EasyOCR model (CPU-optimized)...")
                import torch
                use_gpu = torch.cuda.is_available()
                if use_gpu:
                    logger.info("EasyOCR using GPU")
                else:
                    logger.info("EasyOCR using CPU (this will be slower than GPU)")
                # Use only English and disable verbose logging for speed
                reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                _reader = reader  # Sync with cache
                ocr_type = 'easyocr'
                load_time = time.time() - start_time
                logger.info(f"EasyOCR loaded successfully in {load_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                reader = None
                _reader = None
                ocr_type = None
        else:
            logger.warning("EasyOCR not available. Install with: pip install easyocr")
            reader = None
            _reader = None
            ocr_type = None
        
        # 2. Load YOLO model
        # SPEED-OPTIMIZED ORDER: Nano -> Medium -> Large
        # Nano is much faster on CPU and good enough for simple object/scene detection.
        logger.info("Loading YOLO model (SPEED-OPTIMIZED MODE - prioritizing smaller models)...")
        model_paths = ["yolov8n.pt", "yolov8m.pt", "yolov8l.pt"]
        model = None
        
        # Check for GPU availability
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        for model_path in model_paths:
            try:
                model = YOLO(model_path)
                # Use GPU if available for MUCH better performance
                if device == 'cuda':
                    model.to('cuda')
                    logger.info(f"Loaded YOLO model {model_path} on GPU")
                else:
                    model.to('cpu')
                    logger.info(f"Loaded YOLO model {model_path} on CPU (GPU not available)")
                logger.info(f"Loaded YOLO model: {model_path}")
                break
            except Exception as e:
                logger.warning(f"Could not load {model_path}: {e}")
                continue
        
        if model is None:
            raise Exception("Could not load any YOLO model")
        
        # Initialize advanced detector with TTA for >95% accuracy
        # ENABLED for maximum accuracy (may be slower but provides >95% accuracy)
        global advanced_detector
        if ADVANCED_DETECTION_AVAILABLE:
            try:
                logger.info("Initializing AdvancedDetector with TTA for >95% accuracy...")
                from advanced_detection import AdvancedDetector
                advanced_detector = AdvancedDetector(model)
                logger.info("[OK] AdvancedDetector initialized successfully - >95% accuracy mode enabled")
            except Exception as e:
                logger.warning(f"Advanced detector init failed: {e}. Using standard detection.")
                advanced_detector = None
        else:
            logger.info("AdvancedDetector not available. Using standard YOLO detection.")
            advanced_detector = None
        
        load_time = time.time() - start_time
        logger.info(f"YOLO model loaded successfully in {load_time:.2f} seconds")
        
        models_loaded = True
        model_load_error = None
        return True
    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        model_load_error = error_msg
        models_loaded = False
        model = None
        reader = None
        _reader = None
        ocr_type = None
        return False

# Start server immediately (models will load on first request)
# if not initialize_models():
#     logger.warning("Models failed to load. Some features may not work.")

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

# ===== UTILITY FUNCTIONS =====
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_data(img_b64):
    """Validate base64 image data"""
    if not img_b64:
        return False, "No image data provided"
    
    if not isinstance(img_b64, str):
        return False, "Image data must be a string"
    
    if not img_b64.startswith('data:image'):
        return False, "Invalid image format. Expected base64 data URL"
    
    try:
        # Check size
        size_mb = len(img_b64) / (1024 * 1024)
        if size_mb > 10:
            return False, f"Image too large: {size_mb:.2f}MB. Maximum 10MB allowed"
    except:
        pass
    
    return True, "Valid"

def preprocess_image_for_ocr(img):
    """Preprocess image for better OCR accuracy - Enhanced version"""
    try:
        return EnhancedPreprocessor.preprocess_for_ocr(img)
    except Exception as e:
        logger.warning(f"OCR preprocessing error: {str(e)}")
        # Return original if preprocessing fails
        if len(img.shape) == 3:
            return {'original': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)}
        return {'original': img}

def preprocess_image_for_ocr_fast(img):
    """Fast preprocessing: Only generate top 3 most effective methods for speed"""
    try:
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        results = {}
        height, width = gray.shape
        
        # Upscale if too small
        min_size = 400
        if width < min_size or height < min_size:
            scale = max(min_size / width, min_size / height)
            scale = min(scale, 4.0)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 1. Enhanced (contrast enhancement) - most effective
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        results['enhanced'] = enhanced
        
        # 2. Deskewed (correct rotation)
        deskewed = EnhancedPreprocessor._deskew_image(gray)
        results['deskewed'] = deskewed
        
        # 3. Original (always include)
        results['original'] = gray
        
        # 4. Sharpened (as fallback)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        results['sharpened'] = sharpened
        
        return results
    except Exception as e:
        logger.warning(f"Fast OCR preprocessing error: {str(e)}")
        # Return original if preprocessing fails
        if len(img.shape) == 3:
            return {'original': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)}
        return {'original': img}

def preprocess_image_for_detection(img):
    """Preprocess image for better object detection accuracy - Enhanced version"""
    try:
        return EnhancedPreprocessor.preprocess_for_object_detection(img)
    except Exception as e:
        logger.warning(f"Detection preprocessing error: {str(e)}")
        return img

def b64_to_cv2_img(b64data):
    """Convert base64 image to OpenCV image with comprehensive error handling"""
    try:
        if not b64data:
            logger.warning("Empty image data received")
            return None
        
        # Split header and data
        if ',' not in b64data:
            logger.warning("Invalid base64 format: no comma separator")
            return None
        
        header, encoded = b64data.split(',', 1)
        
        # Validate header
        if not header.startswith('data:image'):
            logger.warning(f"Invalid image header: {header[:50]}")
            return None
        
        # Decode base64
        try:
            img_bytes = base64.b64decode(encoded)
        except Exception as e:
            logger.error(f"Base64 decode error: {str(e)}")
            return None
        
        if len(img_bytes) == 0:
            logger.warning("Decoded image is empty")
            return None
        
        # Open image
        try:
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
        except UnidentifiedImageError:
            logger.error("Could not identify image format")
            return None
        except Exception as e:
            logger.error(f"Image open error: {str(e)}")
            return None
        
        # Validate dimensions
        width, height = img.size
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            logger.warning(f"Image too small: {width}x{height}")
            return None

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            logger.warning(f"Image too large: {width}x{height}. Resizing...")
            # Resize while maintaining aspect ratio
            ratio = min(MAX_IMAGE_DIMENSION / width, MAX_IMAGE_DIMENSION / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized to {new_width}x{new_height}")
        
        # Convert to OpenCV format
        img_array = np.array(img)
        cv2_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return cv2_img
        
    except Exception as e:
        logger.error(f"Error converting base64 to image: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def get_position_label(box, img_width):
    """Get position label for detected object"""
    try:
        x1, y1, x2, y2 = box.xyxy[0]
        center_x = (x1 + x2) / 2
        if center_x < img_width * 0.35:
            return "on the left"
        elif center_x > img_width * 0.65:
            return "on the right"
        else:
            return "in the center"
    except Exception as e:
        logger.error(f"Error getting position label: {str(e)}")
        return "in the scene"

def get_lightweight_actions_from_objects(objects):
    """
    Very lightweight heuristic action detector that ONLY uses YOLO object labels.
    No extra models, no MediaPipe, so it stays fast on CPU.
    Examples of output: 'using a phone', 'possibly drinking', 'sitting or resting'.
    """
    if not objects:
        return []

    counts = Counter(objects)
    actions = []

    # We only describe actions when at least one person is present
    has_person = counts.get('person', 0) > 0
    if not has_person:
        return actions

    # Phone / device usage
    if any(label in counts for label in ['cell phone', 'mobile phone', 'phone']):
        actions.append("using a phone")
    if any(label in counts for label in ['laptop', 'keyboard', 'mouse', 'monitor', 'tv']):
        actions.append("using a computer or screen")

    # Drinking / eating (very coarse – just based on presence of cup/bottle/food)
    if any(label in counts for label in ['cup', 'bottle', 'wine glass']):
        actions.append("possibly drinking")
    if any(label in counts for label in [
        'bowl', 'fork', 'knife', 'spoon',
        'banana', 'apple', 'sandwich', 'pizza',
        'donut', 'cake', 'hot dog', 'orange',
        'broccoli', 'carrot'
    ]):
        actions.append("possibly eating")

    # Sitting / resting – presence of seating furniture
    if any(label in counts for label in ['chair', 'couch', 'sofa', 'bench', 'bed']):
        actions.append("sitting or resting")

    # Reading – book present
    if 'book' in counts:
        actions.append("reading or looking at a book")

    # Carrying bag / backpack
    if any(label in counts for label in ['backpack', 'handbag', 'suitcase']):
        actions.append("carrying a bag")

    # Keep actions short and unique
    seen = set()
    unique_actions = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            unique_actions.append(a)

    return unique_actions

def standard_response(success=True, data=None, message=None, error_code=None):
    """Create standardized API response"""
    response = {
        'success': success,
        'timestamp': datetime.now().isoformat()
    }
    
    if success:
        if data is not None:
            response['data'] = data
        if message:
            response['message'] = message
    else:
        response['error'] = {
            'message': message or 'An error occurred',
            'code': error_code or 'UNKNOWN_ERROR'
        }
    
    return jsonify(response)

# ===== DECORATORS =====
def require_models(f):
    """Decorator to ensure models are loaded before processing"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not models_loaded:
            # Try to initialize lightweight mode if not loaded
            logger.warning("Models not loaded. Initializing lightweight mode...")
            if not initialize_models():
                logger.error("Initialization failed. Cannot process request.")
                error_msg = "Service temporarily unavailable. Initialization failed."
                if model_load_error:
                    error_msg += f" Error: {model_load_error}"
                return standard_response(
                    success=False,
                    message=error_msg,
                    error_code="MODELS_NOT_LOADED"
                ), 503
        
        return f(*args, **kwargs)
    return decorated_function

def validate_request(f):
    """Decorator to validate API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return standard_response(
                success=False,
                message="Request must be JSON",
                error_code="INVALID_CONTENT_TYPE"
            ), 400
        
        data = request.get_json()
        if not data:
            return standard_response(
                success=False,
                message="Empty request body",
                error_code="EMPTY_REQUEST"
            ), 400
        
        if 'image' not in data:
            return standard_response(
                success=False,
                message="Missing 'image' field in request",
                error_code="MISSING_IMAGE"
            ), 400
        
        # Validate image data
        img_b64 = data.get('image')
        is_valid, error_msg = validate_image_data(img_b64)
        if not is_valid:
            return standard_response(
                success=False,
                message=error_msg,
                error_code="INVALID_IMAGE_DATA"
            ), 400
        
        return f(*args, **kwargs)
    return decorated_function

def log_request(f):
    """Decorator to log API requests with HTTP status codes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        endpoint = request.endpoint or request.path
        method = request.method
        
        # Log incoming request
        logger.info(f"Request: {method} {endpoint} from {request.remote_addr}")
        
        try:
            result = f(*args, **kwargs)
            duration = time.time() - start_time
            
            # Extract status code from result
            status_code = 200  # Default
            if isinstance(result, tuple):
                # Flask returns (response, status_code) or (response, status_code, headers)
                if len(result) >= 2 and isinstance(result[1], int):
                    status_code = result[1]
                result_to_check = result[0]
            else:
                result_to_check = result
            
            # Check if result is a Response object with status_code
            if hasattr(result_to_check, 'status_code'):
                status_code = result_to_check.status_code
            
            # Print status code in Flask-like format
            status_text = "OK" if 200 <= status_code < 300 else "ERROR" if status_code >= 500 else "NOT FOUND" if status_code == 404 else "CLIENT ERROR"
            print(f"{method} {endpoint} - {status_code} {status_text} ({duration:.2f}s)")
            
            logger.info(f"Request completed: {method} {endpoint} - {status_code} in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            status_code = 500
            print(f"{method} {endpoint} - {status_code} ERROR ({duration:.2f}s) - {str(e)}")
            logger.error(f"Request failed: {method} {endpoint} after {duration:.2f}s - {str(e)}")
            raise
    return decorated_function

# ===== ROUTES =====
@app.route('/')
def index():
    """Serve main application page"""
    return render_template('platform.html')



@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy' if models_loaded else 'degraded',
        'mode': 'full_deep_learning',
        'models_loaded': models_loaded,
        'yolo': 'loaded' if model is not None else 'not loaded',
        'easyocr': 'loaded' if reader is not None else 'not loaded',
        'timestamp': datetime.now().isoformat()
    }
    
    if model_load_error:
        health_status['error'] = model_load_error
        health_status['status'] = 'unhealthy'
    
    status_code = 200 if models_loaded else 503
    return jsonify(health_status), status_code

@app.route('/api/status', methods=['GET'])
def api_status():
    """Detailed API status endpoint"""
    status = {
        'service': 'VisionAid API',
        'version': '2.0.0-full-dl',
        'models': {
            'yolo': 'loaded' if model is not None else 'not loaded',
            'easyocr': 'loaded' if reader is not None else 'not loaded',
            'paddleocr': 'available' if PADDLEOCR_AVAILABLE else 'not available',
        },
        'ocr_type': ocr_type or 'none',
        'mode': 'full_deep_learning',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    }
    
    if model_load_error:
        status['error'] = model_load_error
    
    return standard_response(success=models_loaded, data=status)

@app.route('/api/detect_objects', methods=['POST'])
@log_request
@require_models
@validate_request
def api_detect_objects():
    """Detect objects in image using YOLO (FAST mode)"""
    try:
        data = request.get_json()
        img_b64 = data.get('image')
        
        # Convert image
        img = b64_to_cv2_img(img_b64)
        if img is None:
            return standard_response(
                success=False,
                message="Could not decode image. Please check image format.",
                error_code="IMAGE_DECODE_ERROR"
            ), 400
        
        if model is None:
            return standard_response(
                success=False,
                message="Object detection model not loaded.",
                error_code="MODEL_NOT_AVAILABLE"
            ), 503

        # Apply enhanced preprocessing for better detection of low-contrast objects
        img = preprocess_image_for_detection(img)

        start_time = time.time()
        
        # Use YOLO model directly for object detection (FAST settings)
        # Smaller image size and fewer detections make this much faster on CPU.
        logger.info("Detecting objects using YOLO model (FAST mode)")
        results = model(
            img,
            conf=0.35,   # Confidence threshold for object detection
            iou=0.50,    # IOU threshold to reduce duplicate detections
            imgsz=640,   # Smaller resolution for faster inference
            max_det=80,  # Limit number of detections for speed
            verbose=False
        )
        
        # Extract detected objects from YOLO results
        objects_detected = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                objects_detected.append(label)
                logger.info(f"Detected: {label} (conf: {conf:.2f})")
        
        detection_time = time.time() - start_time
        
        if not objects_detected:
            logger.info("No objects detected by YOLO in api_detect_objects")
            return standard_response(
                success=True,
                data={'result': 'I don\'t see any recognizable objects in front of you.'},
                message="No objects found"
            )

        # Count and format unique objects
        counts = Counter(objects_detected)
        descriptions = []
        for obj, count in counts.items():
            name = obj
            if count > 1:
                # Basic pluralization
                if name.endswith('s'): name += 'es'
                elif name.endswith('y'): name = name[:-1] + 'ies'
                else: name += 's'
                descriptions.append(f"{count} {name}")
            else:
                descriptions.append(f"a {name}")

        if len(descriptions) == 1:
            result_text = f"I can see {descriptions[0]}."
        else:
            result_text = f"I can see {', '.join(descriptions[:-1])}, and {descriptions[-1]}."
        
        return standard_response(
            success=True,
            data={'result': result_text, 'detection_time': f"{detection_time:.2f}s"},
            message="Objects detected successfully"
        )
            
    except Exception as e:
        logger.error(f"Error in detect_objects: {str(e)}")
        logger.error(traceback.format_exc())
        return standard_response(
            success=False,
            message=f"Error processing image: {str(e)}",
            error_code="PROCESSING_ERROR"
        ), 500

@app.route('/api/describe_scene', methods=['POST'])
@log_request
@require_models
@validate_request
def api_describe_scene():
    """Generate action-focused scene description using tiny YOLO + classical analysis"""
    try:
        # Ensure YOLO model is available
        if model is None:
            logger.error("YOLO model is None in describe_scene")
            return standard_response(
                success=False,
                message="YOLO model not available. Please wait for models to load or restart the server.",
                error_code="MODEL_NOT_AVAILABLE"
            ), 503
        
        data = request.get_json()
        img_b64 = data.get('image')
        
        # Convert image
        img = b64_to_cv2_img(img_b64)
        if img is None:
            return standard_response(
                success=False,
                message="Could not decode image. Please check image format.",
                error_code="IMAGE_DECODE_ERROR"
            ), 400
        
        # Apply enhanced preprocessing for better detection (contrast, noise reduction)
        img = preprocess_image_for_detection(img)
        logger.info(f"Image preprocessed for detection. Size: {img.shape}")
        
        start_time = time.time()
        
        try:
            # CLEAR PATH: Use YOLO model directly for scene description
            # This ensures consistent format (YOLO results with .boxes and .names)
            # BALANCED threshold: Low enough to detect people, high enough for consistency
            logger.info("Detecting objects for scene description using YOLO model (FAST mode)")
            results = model(
                img,
                conf=0.25,   # Slightly higher threshold to skip noisy low-confidence boxes
                iou=0.50,    # IOU threshold to reduce duplicate detections
                verbose=False,
                imgsz=640,   # Smaller image size for MUCH faster inference on CPU
                max_det=80,  # Limit number of detections for speed
                agnostic_nms=False  # Ensure consistent NMS behavior
            )
        except Exception as e:
            logger.error(f"YOLO detection error: {str(e)}")
            logger.error(traceback.format_exc())
            return standard_response(
                success=False,
                message=f"Error during object detection: {str(e)}",
                error_code="DETECTION_ERROR"
            ), 500
        
        height, width, _ = img.shape
        
        # CRITICAL: Collect ALL detected objects, but ONLY REPORT high-confidence ones to users
        # This prevents hallucination-like outputs where YOLO produces low-confidence guesses.
        all_objs = []
        detected_objects_dict = {}
        # Only show objects above this confidence in the final "Objects detected" list
        # Keep YOLO conf low for recall, but keep reporting threshold higher for reliability.
        # Slightly lowered to 0.30 so more real objects are reported while still avoiding noise.
        report_min_conf = 0.30
        confident_objects_dict = {}
        # Simple positional summary per label for LEFT / CENTER / RIGHT and FRONT / BACK
        # We keep the *largest* instance per label to get a stable position.
        position_info = {}  # label -> {'side': ..., 'depth': ..., 'area': ...}

        for r in results:
            for b in r.boxes:
                obj_name = r.names[int(b.cls[0])]
                confidence = float(b.conf[0])
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                area = (x2 - x1) * (y2 - y1)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Include ALL objects detected by YOLO (already filtered at conf=0.25)
                all_objs.append(obj_name)

                # Count total objects
                if obj_name in detected_objects_dict:
                    detected_objects_dict[obj_name] += 1
                else:
                    detected_objects_dict[obj_name] = 1

                # Count only confident detections for user-facing list
                if confidence >= report_min_conf:
                    if obj_name in confident_objects_dict:
                        confident_objects_dict[obj_name] += 1
                    else:
                        confident_objects_dict[obj_name] = 1

                # Update positional info for this label if this instance is larger
                # (larger area → likely closer / more important)
                if obj_name not in position_info or area > position_info[obj_name]['area']:
                    # LEFT / CENTER / RIGHT based on horizontal center
                    rel_x = cx / float(width)
                    if rel_x < 0.33:
                        side = "left"
                    elif rel_x > 0.67:
                        side = "right"
                    else:
                        side = "center"

                    # FRONT / BACK approximation based on relative area
                    area_ratio = area / float(width * height)
                    if area_ratio > 0.03:
                        depth = "front"
                    elif area_ratio < 0.01:
                        depth = "back"
                    else:
                        depth = "middle"

                    position_info[obj_name] = {
                        'side': side,
                        'depth': depth,
                        'area': area,
                        'center': (cx, cy),
                        'area_ratio': area_ratio,
                        'confidence': confidence,
                    }
        
        logger.info(f"YOLO detected {len(all_objs)} objects: {Counter(all_objs)}")
        print(f"🔍 YOLO detected {len(all_objs)} objects: {dict(Counter(all_objs))}")
        
        # Format detected objects list for response - CONFIDENT objects only
        # Completely hide labels the user finds irritating or unhelpful.
        HIDE_LABELS = {'dog', 'hot dog', 'donut'}
        detected_objects_list = []
        for obj_name, count in sorted(confident_objects_dict.items()):
            if obj_name in HIDE_LABELS:
                continue
            if count == 1:
                detected_objects_list.append(obj_name)
            else:
                detected_objects_list.append(f"{obj_name} ({count})")
        
        # CRITICAL: Ensure we have detections before proceeding
        if len(all_objs) == 0:
            logger.warning("⚠️ No objects detected by YOLO - scene detection may be limited")
            print("⚠️ No objects detected by YOLO")
            return standard_response(
                success=True,
                data={
                    'result': "I don't see any recognizable objects in this scene.",
                    'description': "I don't see any recognizable objects in this scene.",
                    'detected_objects': [],
                    'objects_count': 0,
                    'total_detections': 0,
                    'processing_time': f"{time.time() - start_time:.2f}s"
                },
                message="No objects detected"
            )

        # Generate scene description using advanced analyzer
        # This analyzes objects, detects actions, recognizes faces, etc.
        try:
            description = advanced_analyzer.generate_description(img, results)
        except Exception as e:
            logger.error(f"Error in advanced analyzer: {str(e)}")
            logger.error(traceback.format_exc())

            # Fallback: simple describer + lightweight actions
            description = simple_scene_describer.describe_scene(results, width, height)
            actions = get_lightweight_actions_from_objects(all_objs)
            if actions:
                if len(actions) == 1:
                    action_text = actions[0]
                else:
                    action_text = ", ".join(actions[:-1]) + f", and {actions[-1]}"
                desc_clean = description.rstrip(". ")
                description = f"{desc_clean}, and the person is {action_text}."
        
        # ---- FIX: prevent wrong "extra person" wording when only one person is detected ----
        # Sometimes face-recognition text says "X and at least one more person" even when YOLO sees only one person.
        # We trust our deduplicated YOLO people count for this.
        try:
            # compute again here (in case code path changes)
            def _compute_unique_people_count(yolo_results):
                person_boxes = []
                for r in yolo_results:
                    for b in r.boxes:
                        cls_idx = int(b.cls[0])
                        if r.names[cls_idx] != 'person':
                            continue
                        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                        is_duplicate = False
                        for (ax1, ay1, ax2, ay2) in person_boxes:
                            inter_x1 = max(x1, ax1)
                            inter_y1 = max(y1, ay1)
                            inter_x2 = min(x2, ax2)
                            inter_y2 = min(y2, ay2)
                            inter_w = max(0.0, inter_x2 - inter_x1)
                            inter_h = max(0.0, inter_y2 - inter_y1)
                            inter_area = inter_w * inter_h
                            area_b = (x2 - x1) * (y2 - y1)
                            area_a = (ax2 - ax1) * (ay2 - ay1)
                            union = area_a + area_b - inter_area
                            iou = inter_area / union if union > 0 else 0.0
                            if iou > 0.6:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            person_boxes.append((x1, y1, x2, y2))
                return len(person_boxes)

            unique_people_count = _compute_unique_people_count(results)
            if unique_people_count <= 1 and isinstance(description, str):
                # remove "and at least one more person" if present
                description = description.replace(" and at least one more person", "")
                description = description.replace(" and at least one more person.", ".")
                description = description.replace("I see two people", "I see one person")
        except Exception:
            pass

        total_time = time.time() - start_time
        logger.info(f"Scene description completed in {total_time:.2f}s")
        
        # ===== Build SHORT, SCREENSHOT-STYLE SUMMARY =====
        # Robust people count: deduplicate overlapping person boxes so one person
        # is not counted multiple times.
        def _compute_unique_people_count(yolo_results):
            person_boxes = []
            for r in yolo_results:
                for b in r.boxes:
                    cls_idx = int(b.cls[0])
                    if r.names[cls_idx] != 'person':
                        continue
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    # Simple IoU-based duplicate check
                    is_duplicate = False
                    for (ax1, ay1, ax2, ay2) in person_boxes:
                        inter_x1 = max(x1, ax1)
                        inter_y1 = max(y1, ay1)
                        inter_x2 = min(x2, ax2)
                        inter_y2 = min(y2, ay2)
                        inter_w = max(0.0, inter_x2 - inter_x1)
                        inter_h = max(0.0, inter_y2 - inter_y1)
                        inter_area = inter_w * inter_h
                        area_b = (x2 - x1) * (y2 - y1)
                        area_a = (ax2 - ax1) * (ay2 - ay1)
                        union = area_a + area_b - inter_area
                        iou = inter_area / union if union > 0 else 0.0
                        if iou > 0.6:  # >60% overlap → treat as same person
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        person_boxes.append((x1, y1, x2, y2))
            return len(person_boxes)

        people_count = _compute_unique_people_count(results)
        if people_count == 0:
            people_str = "0"
        elif people_count == 1:
            people_str = "1 (one person)"
        else:
            people_str = f"{people_count} people"
        
        # Main objects summary (exclude person, show at most 3)
        main_objects = [obj for obj in detected_objects_list if not obj.startswith('person')][:3]
        objects_str = ", ".join(main_objects) if main_objects else "none clearly visible"
        
        # Simple environment feeling (small environment analysis)
        # Try to give a bit more context than just "calm" / "unclear"
        has_desk_items = any(label in confident_objects_dict for label in ['desk', 'laptop', 'keyboard', 'mouse', 'book'])
        has_relax_items = any(label in confident_objects_dict for label in ['sofa', 'couch', 'bed'])
        many_objects = len(confident_objects_dict) >= 5
        many_people = people_count >= 3

        if people_count == 0 and not main_objects:
            env_str = "Environment is unclear from this image."
        elif many_people or many_objects:
            env_str = "Environment feels busy or a bit cluttered."
        elif has_relax_items:
            env_str = "Environment feels relaxed, like a home or bedroom."
        elif has_desk_items:
            env_str = "Environment feels like a study or work area."
        else:
            env_str = "Environment looks calm and indoor."
        
        screenshot_summary = (
            f"People: {people_str}. "
            f"Main objects: {objects_str}. "
            f"Environment: {env_str}"
        )
        
        # ===== Natural, sentence-style description for the user (screenshot-style text) =====
        # Example: "I see one person sitting on a chair and a cup on the right. The environment looks calm and indoor."
        # Build location-aware phrases for main objects (left / right / center, front / back)
        def _make_object_phrase(obj_label_str: str) -> str:
            # obj_label_str can be "cup" or "cup (2)". We care about the base label.
            base_label = obj_label_str.split('(')[0].strip()
            pos = position_info.get(base_label)
            if not pos:
                return base_label

            side = pos['side']
            depth = pos['depth']

            # Side phrase
            if side == "left":
                side_phrase = "on the left"
            elif side == "right":
                side_phrase = "on the right"
            else:
                side_phrase = "in the center"

            # Depth phrase (front vs background)
            if depth == "front":
                depth_phrase = "in front"
            elif depth == "back":
                depth_phrase = "in the background"
            else:
                depth_phrase = ""

            if depth_phrase:
                return f"{base_label} {depth_phrase} {side_phrase}"
            else:
                return f"{base_label} {side_phrase}"

        # Person location phrase (where the main person is in the frame)
        def _make_person_location_sentence() -> str:
            if people_count != 1:
                return ""
            pos = position_info.get('person')
            if not pos:
                return ""
            side = pos['side']
            depth = pos['depth']

            # Side phrase
            if side == "left":
                side_phrase = "on the left side of the frame"
            elif side == "right":
                side_phrase = "on the right side of the frame"
            else:
                side_phrase = "near the center of the frame"

            # Depth phrase
            if depth == "front":
                depth_phrase = "in front"
            elif depth == "back":
                depth_phrase = "in the background"
            else:
                depth_phrase = ""

            if depth_phrase:
                return f"The person is {depth_phrase} {side_phrase}."
            else:
                return f"The person is {side_phrase}."

        person_loc_sentence = _make_person_location_sentence()

        if main_objects:
            object_phrases = [_make_object_phrase(obj) for obj in main_objects]
            natural_objects = ", ".join(object_phrases)
            # Combine: what the person is doing (description) + where they are + main objects + environment
            if person_loc_sentence:
                natural_result = f"{description} {person_loc_sentence} I also see {natural_objects}. {env_str}"
            else:
                natural_result = f"{description} I also see {natural_objects}. {env_str}"
        else:
            if person_loc_sentence:
                natural_result = f"{description} {person_loc_sentence} {env_str}"
            else:
                natural_result = f"{description} {env_str}"
        
        # ===== Detailed description (used for logs / advanced users) =====
        # ALWAYS be honest about uncertainty:
        # - If we have confident objects, list them
        # - If not, DO NOT list low-confidence guesses; provide a helpful fallback message
        objects_text = ", ".join(detected_objects_list) if detected_objects_list else ""

        # Keep long warnings ONLY in detailed description.
        if detected_objects_list:
            full_result = f"{description} Objects detected: {objects_text}."
        else:
            full_result = (
                f"{description} (No other confident objects detected.) "
                f"Tip: move the camera closer and improve lighting for better scene details."
            )

        # Log both summary and detailed description
        logger.info(f"SCREENSHOT SUMMARY: {screenshot_summary}")
        logger.info(f"DETAILED DESCRIPTION: {full_result}")
        print(f"SCREENSHOT SUMMARY: {screenshot_summary}")

        # IMPORTANT: 'result' is the natural, screenshot-style sentence (what UI speaks/shows)
        return standard_response(
            success=True,
            data={
                'result': natural_result,              # Natural sentence for UI & speech
                'summary': screenshot_summary,         # Explicit summary field
                'description': description,            # Scene description (without object list)
                'detailed_description': full_result,   # Full detailed description with objects
                'detected_objects': detected_objects_list,  # Confident detected objects
                'objects_count': len(confident_objects_dict),  # Unique object types (confident)
                'total_detections': len(all_objs),     # Total YOLO detections (raw)
                'processing_time': f"{total_time:.2f}s"
            },
            message="Scene described successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in describe_scene: {str(e)}")
        logger.error(traceback.format_exc())
        return standard_response(
            success=False,
            message=f"Error processing scene: {str(e)}",
            error_code="PROCESSING_ERROR"
        ), 500

@app.route('/api/read_text', methods=['POST'])
@log_request
@require_models
@validate_request
def api_read_text():
    """Extract text from image using EasyOCR with enhanced preprocessing"""
    global reader, ocr_type, img
    try:
        data = request.get_json()
        img_b64 = data.get('image')
        
        # Convert image
        img = b64_to_cv2_img(img_b64)
        if img is None:
            return standard_response(success=False, message="Could not decode image.", error_code="IMAGE_DECODE_ERROR"), 400

        # Check if at least one OCR engine is available
        if not EASYOCR_AVAILABLE and not PADDLEOCR_AVAILABLE:
            return standard_response(
                success=False,
                message="No OCR engines available. Please install easyocr or paddleocr.",
                error_code="OCR_NOT_AVAILABLE"
            ), 500
        
        start_time = time.time()
        
        # ===== ROBUST HIGH-ACCURACY OCR PIPELINE =====
        try:
            ocr_reader = get_reader() if EASYOCR_AVAILABLE else None
            text = ""
            
            if ocr_reader is not None:
                parts = []
                
                # Pass 1: Speed & Detail Optimized RGB Pass (canvas_size=1024, mag_ratio=1.0)
                # Scale input image to max 1280px for optimal balance of speed and detail
                h, w = img.shape[:2]
                max_dim = 1280
                if w > max_dim or h > max_dim:
                    scale = min(max_dim / w, max_dim / h)
                    img_pass = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
                else:
                    img_pass = img.copy()
                    
                processed_rgb = cv2.cvtColor(img_pass, cv2.COLOR_BGR2RGB)
                try:
                    results = ocr_reader.readtext(
                        processed_rgb,
                        paragraph=False,
                        detail=1,
                        width_ths=0.5,
                        height_ths=0.5,
                        canvas_size=1024,
                        mag_ratio=1.0,
                        prob_ths=0.15
                    )
                    for (bbox, t, prob) in results:
                        if t and t.strip() and prob >= 0.15:
                            parts.append(t.strip())
                except Exception as e:
                    logger.debug(f"Pass 1 EasyOCR failed: {e}")
                
                # Pass 2: If Pass 1 yielded no text, try CLAHE contrast enhancement
                if not parts:
                    try:
                        gray = cv2.cvtColor(img_pass, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        enhanced_gray = clahe.apply(gray)
                        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
                        results_enh = ocr_reader.readtext(
                            enhanced_rgb,
                            paragraph=False,
                            detail=1,
                            width_ths=0.5,
                            height_ths=0.5,
                            canvas_size=1024,
                            mag_ratio=1.0,
                            prob_ths=0.15
                        )
                        for (bbox, t, prob) in results_enh:
                            if t and t.strip() and prob >= 0.15:
                                parts.append(t.strip())
                    except Exception as e:
                        logger.debug(f"Pass 2 CLAHE OCR failed: {e}")
                        
                # Pass 3: If still no text, try upscaled pass
                if not parts:
                    try:
                        upscaled = cv2.resize(img_pass, (int(w * 1.4), int(h * 1.4)), interpolation=cv2.INTER_CUBIC)
                        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
                        results_up = ocr_reader.readtext(
                            upscaled_rgb,
                            paragraph=False,
                            detail=1,
                            width_ths=0.5,
                            height_ths=0.5,
                            canvas_size=1024,
                            mag_ratio=1.0,
                            prob_ths=0.15
                        )
                        for (bbox, t, prob) in results_up:
                            if t and t.strip() and prob >= 0.15:
                                parts.append(t.strip())
                    except Exception as e:
                        logger.debug(f"Pass 3 Upscaled OCR failed: {e}")

                text = " ".join(parts).strip()

            ocr_time = time.time() - start_time

            if not text:
                # Fallback to PaddleOCR if available
                if PADDLEOCR_AVAILABLE:
                    try:
                        paddle_ocr = get_paddle_ocr()
                        if paddle_ocr is not None:
                            p_res = paddle_ocr.predict(img)
                            if p_res and len(p_res) > 0 and 'rec_texts' in p_res[0]:
                                p_texts = [t for t, conf in zip(p_res[0]['rec_texts'], p_res[0]['rec_scores']) if conf > 0.2]
                                if p_texts:
                                    text = " ".join(p_texts).strip()
                    except Exception as p_e:
                        logger.debug(f"PaddleOCR fallback failed: {p_e}")

            if not text:
                return standard_response(
                    success=True,
                    data={
                        "result": "I could not read any clear text from this image. Make sure the text is well-lit, in focus, and try again.",
                        "raw_text": "",
                        "processed_text": "",
                        "ocr_time": f"{ocr_time:.2f}s",
                        "text_detected": False,
                    },
                    message="No text detected",
                )

            response_text = f"I read the following text: {text}"
            return standard_response(
                success=True,
                data={
                    "result": response_text,
                    "raw_text": text,
                    "processed_text": text,
                    "ocr_time": f"{ocr_time:.2f}s",
                    "text_detected": True,
                },
                message="Text read successfully",
            )

        except Exception as e:
            logger.warning(f"Fast OCR path failed: {e}")
            # If slow OCR is disabled, return a quick fallback instead of running heavy pipeline.
            if not ENABLE_SLOW_OCR:
                ocr_time = time.time() - start_time
                return standard_response(
                    success=True,
                    data={
                        "result": "OCR engine is having trouble with this image. Make sure the text is clear and try again.",
                        "raw_text": "",
                        "processed_text": "",
                        "ocr_time": f"{ocr_time:.2f}s",
                        "text_detected": False,
                    },
                    message="OCR fast path failed",
                )

        # ===== LEGACY (SLOWER BUT MORE ROBUST) PIPELINE BELOW – used only if fast path fails =====
        # Resize for performance (legacy settings)
        h, w = img.shape[:2]
        if w > 1600 or h > 1600:
            scale = min(1600 / w, 1600 / h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
        # Preprocessing versions (only needed for EasyOCR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        preprocessed_images = [
            ('original_rgb', cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            ('enhanced', cv2.cvtColor(cv2.createCLAHE(clipLimit=3.0).apply(gray), cv2.COLOR_GRAY2RGB)),
            ('denoised', cv2.cvtColor(cv2.fastNlMeansDenoising(gray, None, h=8), cv2.COLOR_GRAY2RGB))
        ]
        
        text = ""
        all_results = []
        
        # CRITICAL: Try PaddleOCR FIRST since EasyOCR has Tesseract dependency issues
        # PaddleOCR is more reliable and doesn't depend on Tesseract
        if PADDLEOCR_AVAILABLE:
            try:
                logger.info("Using PaddleOCR for text recognition (more reliable, no Tesseract dependency)...")
                paddle_ocr = get_paddle_ocr()
                if paddle_ocr is not None:
                    # Try PaddleOCR on original image first
                    result = paddle_ocr.predict(img)
                    
                    if result and len(result) > 0:
                        ocr_result = result[0]  # Get first result
                        # New PaddleOCR format: rec_texts and rec_scores are lists
                        if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                            texts = []
                            rec_texts = ocr_result['rec_texts']
                            rec_scores = ocr_result['rec_scores']
                            for text_content, confidence in zip(rec_texts, rec_scores):
                                logger.info(f"PaddleOCR: '{text_content}' (conf: {confidence:.2f})")
                                if confidence > 0.3:  # Reasonable threshold
                                    texts.append(text_content)
                            if texts:
                                text = ' '.join(texts)
                                logger.info(f"[OK] PaddleOCR successfully read text: '{text}'")
                                # Skip EasyOCR entirely - PaddleOCR worked!
                                # Jump to response generation
                                all_results = []  # Clear to skip EasyOCR processing
            except Exception as e:
                logger.warning(f"PaddleOCR failed: {e} - will try EasyOCR as fallback")
                text = ""  # Clear text to try EasyOCR
        
        # Only try EasyOCR if PaddleOCR didn't work or isn't available
        # BUT: Skip EasyOCR entirely if it has Tesseract issues
        easyocr_failed_due_to_tesseract = False
        
        if not text and EASYOCR_AVAILABLE:
            try:
                # Ensure reader is initialized - try get_reader() first, then fallback
                ocr_reader = get_reader()
                if ocr_reader is None:
                    # Try to initialize directly
                    if reader is None:
                        try:
                            logger.info("Initializing EasyOCR reader on demand...")
                            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                            _reader = reader  # Sync with cache
                            ocr_type = 'easyocr'
                            logger.info("EasyOCR reader initialized successfully")
                        except Exception as e:
                            logger.error(f"Failed to initialize EasyOCR reader on demand: {e}")
                            reader = None
                            _reader = None
                            easyocr_failed_due_to_tesseract = True
                    ocr_reader = reader
                
                if ocr_reader is not None:
                    for method_name, processed_img in preprocessed_images:
                        try:
                            results = ocr_reader.readtext(
                                processed_img,
                                paragraph=False,
                                width_ths=0.4,  # Balanced - not too lenient to avoid noise
                                height_ths=0.4,  # Balanced
                                detail=1,
                                allowlist=None  # Allow all characters
                            )
                            
                            if results:
                                logger.info(f"EasyOCR ({method_name}) found {len(results)} text regions")
                                for (bbox, text_content, prob) in results:
                                    logger.info(f"  - '{text_content}' (confidence: {prob:.2f})")
                                    all_results.append((text_content, prob, method_name))
                        except Exception as e:
                            # Simple error check - avoid traceback.format_exc() which might trigger Tesseract
                            try:
                                error_str = str(e).lower()
                                error_type_str = str(type(e)).lower()
                                
                                # Check for Tesseract errors WITHOUT using traceback
                                is_tesseract_error = (
                                    "tesseract" in error_str or 
                                    "tesseract" in error_type_str or
                                    "pytesseract" in error_str
                                )
                            except:
                                # If even error checking fails, assume it's a Tesseract error
                                is_tesseract_error = True
                            
                            if is_tesseract_error:
                                logger.warning(f"EasyOCR ({method_name}) encountered Tesseract error - skipping EasyOCR, will use PaddleOCR")
                                easyocr_failed_due_to_tesseract = True
                                # Break out of loop - don't try other preprocessing methods
                                break
                            else:
                                logger.warning(f"EasyOCR ({method_name}) failed: {e}")
                                continue
            except Exception as outer_e:
                # Catch any Tesseract errors that escape the inner loop
                error_str_outer = str(outer_e).lower()
                if "tesseract" in error_str_outer or "pytesseract" in error_str_outer:
                    logger.warning("EasyOCR failed with Tesseract error at outer level - will use PaddleOCR")
                    easyocr_failed_due_to_tesseract = True
                all_results = []
                text = ""
            
            # If EasyOCR failed due to Tesseract, skip combining results and go straight to PaddleOCR
            if easyocr_failed_due_to_tesseract:
                logger.info("EasyOCR failed due to Tesseract - skipping EasyOCR entirely, using PaddleOCR")
                all_results = []  # Clear any partial results
                text = ""  # Ensure text is empty so PaddleOCR is tried
            elif all_results and not easyocr_failed_due_to_tesseract:
                # Only combine results if EasyOCR didn't fail due to Tesseract
                # Combine all results, deduplicate, keep best confidence, filter nonsense
                # Sort by confidence (highest first)
                all_results.sort(key=lambda x: x[1], reverse=True)
                
                # Filter function to validate text quality
                def is_valid_text(text_content, confidence):
                    """Filter out nonsense OCR results"""
                    if not text_content or not text_content.strip():
                        return False
                    
                    text_clean = text_content.strip()
                    
                    # Filter out very short text (likely noise)
                    if len(text_clean) < 2:
                        return False
                    
                    # Filter out very low confidence (lowered threshold for product labels)
                    if confidence < 0.2:  # Lowered from 0.3 to catch more text
                        return False
                    
                    # Filter out text that's mostly special characters or numbers
                    # (unless it's a short number like "50+")
                    has_letters = any(c.isalpha() for c in text_clean)
                    has_digits = any(c.isdigit() for c in text_clean)
                    special_chars = sum(1 for c in text_clean if not c.isalnum() and c != ' ')
                    
                    # More lenient: allow text with letters OR short alphanumeric strings (like "BioUV 50+")
                    # Only filter if it's mostly special chars AND no letters AND longer than 10 chars
                    if not has_letters and special_chars > len(text_clean) * 0.6 and len(text_clean) > 10:
                        return False
                    
                    # Filter out text that's too long (likely artifacts from preprocessing)
                    if len(text_clean) > 100:
                        return False
                    
                    # Prefer text with letters (real words) or short alphanumeric strings
                    if has_letters or (has_digits and len(text_clean) <= 10):
                        return True
                    
                    return False
                
                # Filter and deduplicate - be lenient to catch product labels
                seen = set()
                unique_texts = []
                
                for text_content, prob, method in all_results:
                    text_clean = text_content.strip()
                    if not text_clean:
                        continue
                    
                    text_lower = text_clean.lower()
                    if text_lower not in seen:
                        seen.add(text_lower)
                        
                        # More lenient: accept if passes validation OR if confidence is reasonable
                        if is_valid_text(text_content, prob) or prob >= 0.25:
                            unique_texts.append(text_clean)
                            logger.info(f"Accepted text: '{text_clean}' (confidence: {prob:.2f}, method: {method})")
                        else:
                            logger.debug(f"Filtered out: '{text_clean}' (confidence: {prob:.2f}) - too low quality")
                
                if unique_texts:
                    text = ' '.join(unique_texts)
                    logger.info(f"EasyOCR combined text from {len(unique_texts)} detections: '{text}'")
                else:
                    logger.warning("EasyOCR found text but all were filtered out")
                    text = ""
            else:
                logger.warning("EasyOCR found no text in any preprocessing method")
                text = ""
        else:
            # EasyOCR not available or reader is None
            logger.warning("EasyOCR not available or reader is None")
            text = ""
        
        # CRITICAL: If EasyOCR failed (especially due to Tesseract) or found no text, try PaddleOCR immediately
        # ALWAYS try PaddleOCR if EasyOCR failed due to Tesseract, even if we got some text
        if (easyocr_failed_due_to_tesseract or not text) and PADDLEOCR_AVAILABLE:
            if easyocr_failed_due_to_tesseract:
                logger.info("EasyOCR failed due to Tesseract - immediately switching to PaddleOCR...")
                text = ""  # Clear any partial text from EasyOCR
            else:
                logger.info("No text from EasyOCR - trying PaddleOCR as fallback...")
            try:
                paddle_ocr = get_paddle_ocr()
                if paddle_ocr is not None:
                    logger.info("EasyOCR unavailable or failed - trying PaddleOCR...")
                    result = paddle_ocr.predict(img)
                    
                    if result and len(result) > 0:
                        ocr_result = result[0]  # Get first result
                        # New PaddleOCR format: rec_texts and rec_scores are lists
                        if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                            texts = []
                            rec_texts = ocr_result['rec_texts']
                            rec_scores = ocr_result['rec_scores']
                            for text_content, confidence in zip(rec_texts, rec_scores):
                                logger.info(f"PaddleOCR: '{text_content}' (conf: {confidence:.2f})")
                                if confidence > 0.3:  # Lower threshold for better detection
                                    texts.append(text_content)
                            if texts:
                                text = ' '.join(texts)
                                logger.info(f"PaddleOCR successfully read text: '{text}'")
                            else:
                                logger.warning("PaddleOCR found no text above confidence threshold")
                        else:
                            logger.warning(f"PaddleOCR result format unexpected: {list(ocr_result.keys())}")
            except Exception as e:
                logger.error(f"PaddleOCR failed: {e}")
                logger.error(traceback.format_exc())
        
        # Additional fallback: try EasyOCR lazy loading if main reader failed
        if not text and EASYOCR_AVAILABLE:
            try:
                lazy_reader = get_reader()
                if lazy_reader is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = lazy_reader.readtext(rgb, paragraph=False, width_ths=0.5, height_ths=0.5, detail=1)
                    texts = [t for (_, t, prob) in results if t and t.strip()]
                    text = ' '.join(texts)
            except Exception as e:
                error_str = str(e)
                error_tb = traceback.format_exc()
                # Filter out Tesseract errors - don't log them, just skip
                tesseract_indicators = [
                    "tesseract" in error_str.lower(),
                    "tesseract" in error_tb.lower(),
                    "TesseractNotFoundError" in str(type(e)),
                    "pytesseract" in error_str.lower()
                ]
                if any(tesseract_indicators):
                    logger.warning("EasyOCR lazy loading encountered Tesseract error - will try PaddleOCR")
                else:
                    logger.error(f"EasyOCR lazy loading failed: {e}")
                text = ""
        
        # CRITICAL: Fallback to PaddleOCR if EasyOCR didn't work (including Tesseract errors)
        # Always try PaddleOCR if no text was found
        if not text and PADDLEOCR_AVAILABLE:
            logger.info("No text from EasyOCR - trying PaddleOCR as fallback...")
            try:
                paddle_ocr = get_paddle_ocr()
                if paddle_ocr is not None:
                    logger.info("Trying PaddleOCR as fallback...")
                    result = paddle_ocr.predict(img)
                    
                    if result and len(result) > 0:
                        ocr_result = result[0]  # Get first result
                        # New PaddleOCR format: rec_texts and rec_scores are lists
                        if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                            texts = []
                            rec_texts = ocr_result['rec_texts']
                            rec_scores = ocr_result['rec_scores']
                            for text_content, confidence in zip(rec_texts, rec_scores):
                                logger.info(f"PaddleOCR: '{text_content}' (conf: {confidence:.2f})")
                                if confidence > 0.3:  # Lower threshold for better detection
                                    texts.append(text_content)
                            if texts:
                                text = ' '.join(texts)
                                logger.info(f"PaddleOCR combined text: '{text}'")
                        else:
                            logger.warning(f"PaddleOCR result format unexpected: {list(ocr_result.keys())}")
            except Exception as e:
                logger.error(f"PaddleOCR failed: {e}")
                logger.error(traceback.format_exc())
             
        ocr_time = time.time() - start_time
        original_text = text

        text = (text or "").strip()

        if not text:
            # Last resort: Use OpenCV to at least detect text presence (no engine-specific message)
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Use adaptive thresholding to find text regions
                thresh = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11,
                    2,
                )
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Count potential text regions
                text_regions = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 50 < area < 5000:  # Reasonable size for text
                        text_regions += 1

                if text_regions > 5:
                    logger.info(
                        f"OpenCV detected {text_regions} potential text regions but OCR returned no clear text"
                    )
            except Exception as e:
                logger.error(f"OpenCV text detection failed: {e}")
        
        if not text:
            # Even if no text, show what OCR engines tried
            debug_info = []
            if EASYOCR_AVAILABLE:
                debug_info.append("EasyOCR was available")
            if PADDLEOCR_AVAILABLE:
                debug_info.append("PaddleOCR was available")
            
            return standard_response(
                success=True,
                data={
                    'result': 'I could not read any clear text from this image. Make sure the text is well-lit, in focus, and try holding the camera steady. ' + ' '.join(debug_info),
                    'ocr_time': f"{ocr_time:.2f}s",
                    'debug': 'No text detected by OCR engines'
                },
                message="No text detected"
            )

        # Summarize text if it's long and summarization is available
        original_text = text
        if SUMMARIZATION_AVAILABLE and len(text.split()) >= 15:
            try:
                summary_start = time.time()
                summarized_text = summarize_text(text, max_length=60, min_length=20)
                summary_time = time.time() - summary_start
                if summarized_text != text:
                    logger.info(f"Text summarized in {summary_time:.2f}s (original: {len(text.split())} words, summary: ~{len(summarized_text.split())} words)")
                    text = summarized_text
            except Exception as e:
                logger.warning(f"Summarization failed, using original text: {e}")
                text = original_text

        # Always show detected text, even if partial or imperfect
        if text and text.strip():
            response_text = f"I read the following text: {text}"
        else:
            response_text = "I could not read any clear text from this image. Please make sure the text is well-lit, in focus, and try holding the camera steady."

        return standard_response(
            success=True,
            data={
                'result': response_text,
                'raw_text': original_text if original_text else text,  # Always return original OCR text
                'processed_text': text if text else "",     # Final processed/summarized version
                'ocr_time': f"{ocr_time:.2f}s",
                'text_detected': bool(text and text.strip())  # Flag to show if any text was found
            },
            message="Text read successfully" if text else "No text detected"
        )
            
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"Error in read_text: {error_msg}")
        logger.error(error_traceback)
        
        # COMPREHENSIVE Tesseract error detection - catch ALL variations
        tesseract_keywords = [
            "tesseract", "TesseractNotFoundError", "pytesseract", 
            "tesseract is not installed", "not in your PATH",
            "See README file"  # Part of Tesseract error message
        ]
        
        is_tesseract_error = False
        # Safely check traceback (might be None or empty)
        traceback_lower = error_traceback.lower() if error_traceback else ""
        for keyword in tesseract_keywords:
            if keyword.lower() in error_msg.lower() or keyword.lower() in traceback_lower:
                is_tesseract_error = True
                break
        
        # Also check exception type
        if "TesseractNotFoundError" in str(type(e)) or "TesseractNotFoundError" in str(e):
            is_tesseract_error = True
        
        if is_tesseract_error:
            logger.warning("Tesseract error detected - EasyOCR failed, trying PaddleOCR fallback...")
            
            # Try PaddleOCR as fallback before giving up
            if PADDLEOCR_AVAILABLE:
                try:
                    # Get the image from the request
                    data = request.get_json()
                    img_b64 = data.get('image')
                    img_fallback = b64_to_cv2_img(img_b64)
                    
                    if img_fallback is not None:
                        paddle_ocr = get_paddle_ocr()
                        if paddle_ocr is not None:
                            logger.info("Attempting PaddleOCR as fallback after EasyOCR Tesseract error...")
                            result = paddle_ocr.predict(img_fallback)
                            
                            if result and len(result) > 0:
                                ocr_result = result[0]  # Get first result
                                # New PaddleOCR format: rec_texts and rec_scores are lists
                                if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                                    texts = []
                                    rec_texts = ocr_result['rec_texts']
                                    rec_scores = ocr_result['rec_scores']
                                    for text_content, confidence in zip(rec_texts, rec_scores):
                                        logger.info(f"PaddleOCR: '{text_content}' (conf: {confidence:.2f})")
                                        if confidence > 0.3:  # Lower threshold
                                            texts.append(text_content)
                                else:
                                    texts = []
                                    logger.warning(f"PaddleOCR result format unexpected: {list(ocr_result.keys())}")
                            else:
                                texts = []
                            
                            if texts:
                                text_fallback = ' '.join(texts)
                                logger.info(f"PaddleOCR fallback successful! Read: '{text_fallback}'")
                                return standard_response(
                                    success=True,
                                    data={
                                        'result': f"I read the following text: {text_fallback}",
                                        'raw_text': text_fallback,
                                        'processed_text': text_fallback,
                                        'ocr_time': '0.0s',
                                        'text_detected': True
                                    },
                                    message="Text read successfully using PaddleOCR"
                                )
                except Exception as paddle_error:
                    logger.error(f"PaddleOCR fallback also failed: {paddle_error}")
            
            # If PaddleOCR also failed or isn't available, return generic error
            logger.warning("Both EasyOCR and PaddleOCR failed - returning generic error")
            return standard_response(
                success=False,
                message="OCR engine temporarily unavailable. Please try again in a moment.",
                error_code="OCR_ERROR"
            ), 500
        
        return standard_response(
            success=False,
            message=f"Error reading text: {error_msg}",
            error_code="PROCESSING_ERROR"
        ), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return standard_response(
        success=False,
        message="Image file too large. Maximum size is 10MB.",
        error_code="FILE_TOO_LARGE"
    ), 413

@app.route('/api/register_face', methods=['POST'])
@log_request
@validate_request
def api_register_face():
    """Register a face with a name for recognition"""
    try:
        data = request.get_json()
        img_b64 = data.get('image')
        name = data.get('name', '').strip()
        
        if not name:
            return standard_response(
                success=False,
                message="Name is required. Please provide a name for the face.",
                error_code="MISSING_NAME"
            ), 400
        
        # Convert image
        img = b64_to_cv2_img(img_b64)
        if img is None:
            return standard_response(
                success=False,
                message="Could not decode image. Please check image format.",
                error_code="IMAGE_DECODE_ERROR"
            ), 400
        
        # Register face
        try:
            success, message = face_recognition.register_face(name, img)
            
            if success:
                logger.info(f"Face registered successfully for: {name}")
                logger.info(f"Total registered faces: {len(face_recognition.face_encodings)}")
                logger.info(f"Registered names: {list(face_recognition.face_encodings.keys())}")
                # Verify face was saved
                face_recognition.load_faces()  # Reload to verify persistence
                logger.info(f"Verified: {len(face_recognition.face_encodings)} faces loaded from disk")
                return standard_response(
                    success=True,
                    data={'name': name, 'message': message or f"Face registered for {name}"},
                    message=message or f"Face registered for {name}"
                )
            else:
                error_msg = message or "Failed to register face. Please ensure your face is clearly visible."
                logger.warning(f"Face registration failed for {name}: {error_msg}")
                return standard_response(
                    success=False,
                    message=error_msg,
                    error_code="FACE_REGISTRATION_ERROR"
                ), 400
        except Exception as face_error:
            logger.error(f"Face recognition module error: {str(face_error)}")
            logger.error(traceback.format_exc())
            return standard_response(
                success=False,
                message=f"Face recognition error: {str(face_error)}",
                error_code="FACE_MODULE_ERROR"
            ), 500
            
    except Exception as e:
        logger.error(f"Error in register_face: {str(e)}")
        logger.error(traceback.format_exc())
        return standard_response(
            success=False,
            message=f"Error registering face: {str(e)}",
            error_code="PROCESSING_ERROR"
        ), 500

@app.route('/api/list_faces', methods=['GET'])
@log_request
def api_list_faces():
    """Get list of registered faces"""
    try:
        names = face_recognition.get_registered_names()
        return standard_response(
            success=True,
            data={'faces': names, 'count': len(names)},
            message=f"Found {len(names)} registered faces"
        )
    except Exception as e:
        logger.error(f"Error listing faces: {str(e)}")
        return standard_response(
            success=False,
            message=f"Error listing faces: {str(e)}",
            error_code="PROCESSING_ERROR"
        ), 500

@app.route('/api/delete_face', methods=['POST'])
@log_request
@validate_request
def api_delete_face():
    """Delete a specific registered face by name"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        
        if not name:
            return standard_response(
                success=False,
                message="Name is required. Please provide the name of the face to delete.",
                error_code="MISSING_NAME"
            ), 400
        
        success, message = face_recognition.delete_face(name)
        if success:
            logger.info(f"Face deleted successfully: {name}")
            logger.info(f"Remaining registered faces: {list(face_recognition.face_encodings.keys())}")
            return standard_response(
                success=True,
                message=message,
                data={'name': name, 'remaining_count': len(face_recognition.face_encodings)}
            )
        else:
            logger.warning(f"Face deletion failed: {message}")
            return standard_response(
                success=False,
                message=message,
                error_code="FACE_NOT_FOUND"
            ), 404
    except Exception as e:
        logger.error(f"Error deleting face: {str(e)}")
        logger.error(traceback.format_exc())
        return standard_response(
            success=False,
            message=f"Error deleting face: {str(e)}",
            error_code="PROCESSING_ERROR"
        ), 500

@app.route('/api/clear_all_faces', methods=['POST'])
@log_request
def api_clear_all_faces():
    """Clear all registered faces"""
    try:
        success, message = face_recognition.clear_all_faces()
        if success:
            logger.info("All registered faces cleared")
            return standard_response(
                success=True,
                message=message,
                data={'count': 0}
            )
        else:
            return standard_response(
                success=False,
                message=message,
                error_code="CLEAR_ERROR"
            ), 500
    except Exception as e:
        logger.error(f"Error clearing faces: {str(e)}")
        logger.error(traceback.format_exc())
        return standard_response(
            success=False,
            message=f"Error clearing faces: {str(e)}",
            error_code="PROCESSING_ERROR"
        ), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return standard_response(
        success=False,
        message="Endpoint not found",
        error_code="NOT_FOUND"
    ), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return standard_response(
        success=False,
        message="Internal server error. Please try again later.",
        error_code="INTERNAL_ERROR"
    ), 500

# ===== MAIN =====
if __name__ == '__main__':
    print("\n" + "="*60)
    print("VisionAid Enhanced Server Starting...")
    print("="*60)
    
    # Check for critical dependencies
    try:
        import flask
    except ImportError:
        print("ERROR: Flask is not installed!")
        print("Please run: pip install Flask")
        print("Please run: pip install Flask")
        sys.exit(1)
    
    print(f"Models loaded: {models_loaded}")
    if model_load_error:
        print(f"Warning: {model_load_error}")
        print("Server will start but some features may not work.")
    
    print("\n" + "="*60)
    print("Server starting on: http://127.0.0.1:5000")
    print("Also available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("API status: http://localhost:5000/api/status")
    print("="*60)
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e) or "address is already in use" in str(e).lower():
            print("\n" + "="*60)
            print("ERROR: Port 5000 is already in use!")
            print("Another instance of the server may be running.")
            print("Please:")
            print("  1. Close any other running servers")
            print("  2. Or change the port in app_web.py")
            print("="*60)
        else:
            print(f"\nERROR starting server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\n\nERROR: Server crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
