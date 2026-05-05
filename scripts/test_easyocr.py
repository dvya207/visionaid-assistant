import cv2
from ultralytics import YOLO  # Use Ultralytics YOLOv8
import easyocr
import os
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
# YOLO model path (relative)
yolo_model_path = os.path.join("..", "yolov8n.pt")  # Update to your model name if needed

# Test image path (relative)
image_path = os.path.join("..", "images", "test_image.jpg")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # English

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
try:
    model = YOLO(yolo_model_path)
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    exit(1)

# Read image
img = cv2.imread(image_path)
if img is None:
    print(f"❌ Failed to load image: {image_path}")
    exit(1)

# -----------------------------
# OBJECT DETECTION
# -----------------------------
try:
    results = model(img)
except Exception as e:
    print(f"❌ YOLO inference failed: {e}")
    exit(1)

# Display detected objects
print("Detected Objects:")
detected_labels = set()
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        detected_labels.add(label)
        # Draw detection boxes on image
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
for label in detected_labels:
    print("-", label)

# -----------------------------
# TEXT READING (OCR)
# -----------------------------
ocr_results = reader.readtext(image_path)

print("\nDetected Text:")
for (bbox, text, prob) in ocr_results:
    print(f"- {text} | Confidence: {prob:.2f}")
    # Draw OCR boxes
    pts = [tuple(map(int, point)) for point in bbox]
    cv2.polylines(img, [np.array(pts)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(img, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Show image with detections and OCR
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
