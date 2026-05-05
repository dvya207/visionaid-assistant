import sys
import easyocr
import os

if len(sys.argv) < 2:
    print("Usage: python read_image_text.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    sys.exit(1)

reader = easyocr.Reader(['en'])
ocr_results = reader.readtext(image_path)

print("Detected Text:")
for (bbox, text, prob) in ocr_results:
    print(f"- {text} | Confidence: {prob:.2f}")
