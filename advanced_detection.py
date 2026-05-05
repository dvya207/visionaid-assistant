"""
Advanced Detection Module for >90% Accuracy
Implements: Test-Time Augmentation, Multi-Scale Detection, Ensemble Methods
"""
import cv2
import numpy as np
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)

class AdvancedDetector:
    """Advanced detection techniques for maximum accuracy"""
    
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.device = 'cuda' if hasattr(yolo_model, 'device') and 'cuda' in str(yolo_model.device) else 'cpu'
    
    def test_time_augmentation(self, img, conf=0.10, iou=0.45):
        """
        Test-Time Augmentation: Run detection on multiple augmented versions
        Returns: Ensemble of all detections with weighted confidence
        """
        all_detections = []
        
        # 1. Original image
        results = self.model(img, conf=conf, iou=iou, imgsz=1024, max_det=200, verbose=False)
        all_detections.extend(self._extract_detections(results, weight=1.0))
        
        # 2. Horizontal flip (mirror)
        img_flipped = cv2.flip(img, 1)
        results = self.model(img_flipped, conf=conf, iou=iou, imgsz=1024, max_det=200, verbose=False)
        detections = self._extract_detections(results, weight=0.9)
        # Flip coordinates back
        for det in detections:
            h, w = img.shape[:2]
            det['bbox'] = (w - det['bbox'][2], det['bbox'][1], w - det['bbox'][0], det['bbox'][3])
        all_detections.extend(detections)
        
        # 3. Slightly brighter version
        img_bright = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
        results = self.model(img_bright, conf=conf, iou=iou, imgsz=1024, max_det=200, verbose=False)
        all_detections.extend(self._extract_detections(results, weight=0.85))
        
        # 4. Slightly darker version
        img_dark = cv2.convertScaleAbs(img, alpha=0.9, beta=-10)
        results = self.model(img_dark, conf=conf, iou=iou, imgsz=1024, max_det=200, verbose=False)
        all_detections.extend(self._extract_detections(results, weight=0.85))
        
        # 5. Enhanced contrast version
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        img_contrast = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
        results = self.model(img_contrast, conf=conf, iou=iou, imgsz=1024, max_det=200, verbose=False)
        all_detections.extend(self._extract_detections(results, weight=0.9))
        
        # 6. Multi-scale detection (slightly larger)
        h, w = img.shape[:2]
        scale = 1.1
        img_large = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        results = self.model(img_large, conf=conf, iou=iou, imgsz=1024, max_det=200, verbose=False)
        detections = self._extract_detections(results, weight=0.85)
        # Scale coordinates back
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            det['bbox'] = (x1/scale, y1/scale, x2/scale, y2/scale)
        all_detections.extend(detections)
        
        # 7. Multi-scale detection (slightly smaller)
        scale = 0.9
        img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        results = self.model(img_small, conf=conf, iou=iou, imgsz=1024, max_det=200, verbose=False)
        detections = self._extract_detections(results, weight=0.85)
        # Scale coordinates back
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            det['bbox'] = (x1/scale, y1/scale, x2/scale, y2/scale)
        all_detections.extend(detections)
        
        # Ensemble: Combine detections using weighted NMS
        final_detections = self._weighted_nms(all_detections, iou_threshold=0.5)
        
        return final_detections
    
    def _extract_detections(self, results, weight=1.0):
        """Extract detections from YOLO results"""
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                conf = float(box.conf[0]) * weight  # Weighted confidence
                x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0]]
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'class': cls
                })
        return detections
    
    def _weighted_nms(self, detections, iou_threshold=0.5):
        """
        Weighted Non-Maximum Suppression
        Combines overlapping detections by averaging their confidences
        """
        if not detections:
            return []
        
        # Group by class
        by_class = defaultdict(list)
        for det in detections:
            by_class[det['class']].append(det)
        
        final_detections = []
        
        for cls, class_dets in by_class.items():
            # Sort by confidence
            class_dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Apply weighted NMS
            kept = []
            for det in class_dets:
                x1, y1, x2, y2 = det['bbox']
                area = (x2 - x1) * (y2 - y1)
                
                # Check overlap with kept detections
                should_keep = True
                for kept_det in kept:
                    kx1, ky1, kx2, ky2 = kept_det['bbox']
                    karea = (kx2 - kx1) * (ky2 - ky1)
                    
                    # Calculate IoU
                    inter_x1 = max(x1, kx1)
                    inter_y1 = max(y1, ky1)
                    inter_x2 = min(x2, kx2)
                    inter_y2 = min(y2, ky2)
                    
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        union_area = area + karea - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        if iou > iou_threshold:
                            # Merge: average bbox and sum confidence
                            kept_det['bbox'] = (
                                (x1 + kx1) / 2,
                                (y1 + ky1) / 2,
                                (x2 + kx2) / 2,
                                (y2 + ky2) / 2
                            )
                            kept_det['confidence'] = max(kept_det['confidence'], det['confidence'])
                            should_keep = False
                            break
                
                if should_keep:
                    kept.append(det)
            
            final_detections.extend(kept)
        
        # Filter by final confidence threshold
        final_detections = [d for d in final_detections if d['confidence'] >= 0.10]
        
        # Sort by confidence
        final_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return final_detections[:200]  # Top 200
    
    def detect_with_tta(self, img, conf=0.10, iou=0.45):
        """
        Main detection method with Test-Time Augmentation
        Returns: List of detections with improved accuracy
        """
        try:
            detections = self.test_time_augmentation(img, conf=conf, iou=iou)
            return detections
        except Exception as e:
            logger.error(f"TTA detection failed: {e}")
            # Fallback to single detection
            results = self.model(img, conf=conf, iou=iou, imgsz=1024, max_det=200, verbose=False)
            return self._extract_detections(results, weight=1.0)

