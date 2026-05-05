import cv2
import easyocr
import pyttsx3
import time
import numpy as np  # Added import

# Initialize OCR and TTS
reader = easyocr.Reader(['en'])
engine = pyttsx3.init()

# Open camera (0 = default camera, change to 1 for external camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

print("✅ Camera started. Press Q to quit.")
last_text = ""
last_speak_time = 0
speak_interval = 2  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Convert frame to RGB for EasyOCR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # OCR
        ocr_results = reader.readtext(rgb_frame)
        detected_texts = [text for (_, text, prob) in ocr_results if prob > 0.5]
        combined_text = ' '.join(detected_texts)

        # Draw OCR boxes and text
        for (bbox, text, prob) in ocr_results:
            if prob > 0.5:
                pts = [tuple(map(int, point)) for point in bbox]
                cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(255, 0, 0), thickness=2)
                cv2.putText(frame, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Speak only if new text is detected and enough time has passed
        current_time = time.time()
        if combined_text and combined_text != last_text and (current_time - last_speak_time > speak_interval):
            print(f"Reading: {combined_text}")
            engine.say(combined_text)
            engine.runAndWait()
            last_text = combined_text
            last_speak_time = current_time

        cv2.imshow("VisionAid - Real-time Text Reader", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()
