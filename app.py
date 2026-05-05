import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Initialize text-to-speech
engine = pyttsx3.init()

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    exit(1)

# Open webcam
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("❌ Webcam not detected")
    exit()

print("✅ Webcam started. Press Q to quit. Press C to switch camera.")

last_spoken = set()
last_speak_time = 0
speak_interval = 2  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Run YOLO detection
        try:
            results = model(frame, stream=True)
        except Exception as e:
            print(f"❌ YOLO inference failed: {e}")
            break

        detected_objects = set()

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                detected_objects.add(label)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Speak only new detected objects at intervals
        current_time = time.time()
        new_objects = detected_objects - last_spoken
        if new_objects and (current_time - last_speak_time > speak_interval):
            for obj in sorted(new_objects):
                engine.say(obj)
            engine.runAndWait()
            last_spoken = detected_objects.copy()
            last_speak_time = current_time

        cv2.imshow("VisionAid - Object Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("🔄 Switching camera...")
            cap.release()
            camera_index += 1
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"⚠️ Camera {camera_index} not found. Reverting to default.")
                camera_index = 0
                cap = cv2.VideoCapture(camera_index)
            time.sleep(0.5)
finally:
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()
