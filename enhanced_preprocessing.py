"""
Enhanced Preprocessing Module for Computer Vision & OCR Tasks
Provides image preprocessing utilities for OCR, face recognition, and object detection.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EnhancedPreprocessor:
    """Pre-processing utilities for computer vision models"""
    
    @staticmethod
    def _deskew_image(img):
        """Deskew image based on minimum area rectangle of thresholded text regions"""
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
                
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) == 0:
                return img
                
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return img

    @staticmethod
    def preprocess_for_ocr(img):
        """Generates multiple preprocessed variations of an image for OCR"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        results = {}
        results['original'] = gray
        
        # Enhanced contrast with CLAHE
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            results['enhanced'] = clahe.apply(gray)
        except Exception:
            results['enhanced'] = gray
            
        # Deskewed
        results['deskewed'] = EnhancedPreprocessor._deskew_image(gray)
        
        # Adaptive thresholding
        try:
            results['thresh'] = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        except Exception:
            pass
            
        return results

    @staticmethod
    def preprocess_for_object_detection(img):
        """Preprocess image to enhance details for object detection"""
        try:
            # Gentle CLAHE contrast enhancement
            if len(img.shape) == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return img
        except Exception as e:
            logger.warning(f"Object detection preprocessing failed: {e}")
            return img

    @staticmethod
    def preprocess_for_face_recognition(face_image):
        """Preprocess cropped face image for face recognition model"""
        try:
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image.copy()
            # Equalize histogram for facial features
            equalized = cv2.equalizeHist(gray)
            return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.warning(f"Face recognition preprocessing failed: {e}")
            return face_image
