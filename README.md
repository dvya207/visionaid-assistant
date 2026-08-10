# VisionAid Assistant

> **AI-Powered Visual & Scene Recognition Assistant for Visually Impaired Users**

VisionAid Assistant is an intelligent computer vision backend and web interface designed to provide scene description, real-time object detection, high-accuracy OCR text reading, face recognition, and action detection to assist visually impaired individuals in navigating their environment safely and independently.

---

## Key Features

- **Real-Time Object Detection**: Powered by Ultralytics YOLOv8 with Test-Time Augmentation (TTA) for spatial awareness and object identification.
- **High-Speed & Robust OCR**: Deep learning text recognition powered by CPU-optimized EasyOCR and PaddleOCR fallbacks with CLAHE contrast enhancement for low-light/poorly-lit text.
- **Action & Scene Description**: Automated scene context generation detailing poses, seating, reading, eating, and device usage.
- **Face Registration & Recognition**: Detects and remembers registered faces using OpenCV, MediaPipe, and DeepFace embeddings.
- **Web Application Interface**: Responsive web app with live camera stream processing, text-to-speech audio feedback, and clean UI design.
- **RESTful API**: Lightweight Flask server providing JSON endpoints for external client integrations.

---

## Tech Stack

- **Core**: Python 3.10+, Flask
- **Computer Vision & Deep Learning**: PyTorch, Ultralytics YOLOv8, OpenCV
- **OCR Engines**: EasyOCR, PaddleOCR
- **Face & Landmark Analysis**: OpenCV Haar Cascades, MediaPipe, DeepFace
- **Frontend**: HTML5, CSS3, JavaScript, SpeechSynthesis API

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/dvya207/visionaid-assistant.git
cd visionaid-assistant
```

### 2. Install Dependencies
```bash
pip install flask opencv-python numpy ultralytics easyocr pillow pyttsx3
```
*(Optional) For enhanced OCR fallback capability:*
```bash
pip install paddlepaddle paddleocr
```

### 3. Run Web Application Server
```bash
python app_web.py
```

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | System health check and model loading status |
| `/api/status` | `GET` | API version and active model parameters |
| `/api/detect_objects` | `POST` | Detect objects in base64 image input using YOLOv8 |
| `/api/describe_scene` | `POST` | Generate action-focused scene description |
| `/api/read_text` | `POST` | Extract and read text from image with multi-pass OCR |
| `/api/register_face` | `POST` | Register a new face with a person's name |
| `/api/list_faces` | `GET` | List all registered face names |
| `/api/delete_face` | `POST` | Remove a registered face |
| `/api/clear_all_faces` | `POST` | Clear all registered faces |

---

## License

MIT License
