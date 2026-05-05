import cv2
import easyocr
import pyttsx3
import speech_recognition as sr
from ultralytics import YOLO
import numpy as np
import time
from scene_describer import scene_describer
from advanced_scene_analyzer import AdvancedSceneAnalyzer

# Initialize TTS, OCR, YOLO, and Speech Recognition
print("Initializing modules...")
try:
    engine = pyttsx3.init()
    print("TTS initialized.")
except Exception as e:
    print(f"Error initializing TTS: {e}")

try:
    reader = easyocr.Reader(['en'])
    print("OCR initialized.")
except Exception as e:
    print(f"Error initializing OCR: {e}")

try:
    model = YOLO("yolov8m.pt")
    print("YOLO (Medium) initialized.")
except Exception as e:
    print(f"Error loading YOLO: {e}")

try:
    recognizer = sr.Recognizer()
    print("Speech Recognition initialized.")
except Exception as e:
    print(f"Error initializing Speech Recognition: {e}")

# Initialize Advanced Analyzer
try:
    advanced_analyzer = AdvancedSceneAnalyzer()
    if advanced_analyzer.load_models():
        print("Advanced Scene Analyzer initialized.")
    else:
        print(f"Advanced Analyzer loaded with limitations: {advanced_analyzer.load_error}")
except Exception as e:
    print(f"Error initializing Advanced Analyzer: {e}")
    advanced_analyzer = None

# Helper: Speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen_command():
    with sr.Microphone() as source:
        print("Listening for command...")
        # Short friendly prompt or distinct beep sound here would be better than long speech
        # speak("Ready.") 
        try:
            # Adjust ambient noise for better accuracy
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"Heard: {command}")
            return command
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            speak("Speech service offline.")
            return None
        except Exception:
            return None

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not detected.")
        return None
    
    # speak("Capturing...")
    ret, frame = cap.read()
    
    # Warmup / Stabilization
    for _ in range(5):
        cap.read()
        
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        speak("Failed to capture image.")
        return None
    return frame

def describe_scene(img):
    speak("Analyzing scene...")
    
    if advanced_analyzer and model:
        try:
            # Run YOLO first as advanced analyzer needs results
            results = model(img)
            # Use advanced generation
            description = advanced_analyzer.generate_description(img, results)
        except Exception as e:
            print(f"Advanced analysis failed: {e}")
            description = scene_describer.describe_image(img)
    else:
        description = scene_describer.describe_image(img)
        
    speak(description)

def detect_objects(img):
    speak("Detecting objects...")
    results = model(img)
    height, width, _ = img.shape
    
    # Use advanced analyzer helper if available, else simple logic
    if advanced_analyzer:
        # Re-use logic or build simple list
       pass # Already covered by describe_scene mostly, but let's do pure detection list
    
    detected_descriptions = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            # Simple position
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
            if x_center < width * 0.35: pos = "left"
            elif x_center > width * 0.65: pos = "right"
            else: pos = "center"
            
            detected_descriptions.append(f"{label} {pos}")
    
    if detected_descriptions:
        # De-duplicate smartly?
        speak("I see: " + ', '.join(detected_descriptions))
    else:
        speak("I don't see any distinct objects.")

def read_text(img):
    speak("Reading text...")
    try:
        ocr_results = reader.readtext(img)
        texts = [text for (_, text, prob) in ocr_results if prob > 0.4]
        if texts:
            speak("Text says: " + ' '.join(texts))
        else:
            speak("No clear text found.")
    except Exception as e:
        speak("Could not read text.")

if __name__ == "__main__":
    speak("Vision Assistant Ready. Say describe, object, or read text.")
    
    while True:
        command = listen_command()
        
        if not command:
            continue
            
        if "exit" in command or "stop" in command:
            speak("Stopping assistant.")
            break
            
        # Determine intent
        intent = None
        if any(w in command for w in ["describe", "scene", "what do you see"]):
            intent = "describe"
        elif any(w in command for w in ["object", "detect", "find"]):
            intent = "detect"
        elif any(w in command for w in ["read", "text", "words"]):
            intent = "read"
            
        if intent:
            img = capture_image()
            if img is not None:
                if intent == "describe":
                    describe_scene(img)
                elif intent == "detect":
                    detect_objects(img)
                elif intent == "read":
                    read_text(img)
        else:
            # speak("Say describe, object, or read.")
            pass
