"""
Quick diagnostic tool to check OCR availability
"""

print("=" * 50)
print("VisionAid OCR Diagnostic")
print("=" * 50)

# Check EasyOCR
print("\n1. Checking EasyOCR...")
try:
    import easyocr
    print("   [OK] EasyOCR module found")
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        print("   [OK] EasyOCR initialized successfully")
    except Exception as e:
        print(f"   [X] EasyOCR failed to initialize: {e}")
except ImportError:
    print("   [X] EasyOCR not installed")

# Check PaddleOCR
print("\n2. Checking PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    print("   [OK] PaddleOCR module found")
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("   [OK] PaddleOCR initialized successfully")
    except Exception as e:
        print(f"   [X] PaddleOCR failed to initialize: {e}")
except ImportError:
    print("   [X] PaddleOCR not installed")

print("\n" + "=" * 50)
print("Recommendation:")
print("   Install at least one OCR engine:")
print("   pip install easyocr")
print("   OR")
print("   pip install paddlepaddle paddleocr")
print("=" * 50)
