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
    print("   ✓ EasyOCR module found")
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        print("   ✓ EasyOCR initialized successfully")
    except Exception as e:
        print(f"   ✗ EasyOCR failed to initialize: {e}")
except ImportError:
    print("   ✗ EasyOCR not installed")

# Check PaddleOCR
print("\n2. Checking PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    print("   ✓ PaddleOCR module found")
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("   ✓ PaddleOCR initialized successfully")
    except Exception as e:
        print(f"   ✗ PaddleOCR failed to initialize: {e}")
except ImportError:
    print("   ✗ PaddleOCR not installed")

print("\n" + "=" * 50)
print("Recommendation:")
print("   Install at least one OCR engine:")
print("   pip install easyocr")
print("   OR")
print("   pip install paddlepaddle paddleocr")
print("=" * 50)
