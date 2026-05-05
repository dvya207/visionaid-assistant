"""
Object Detection Validation Module
Filters out false positives and misclassifications
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ObjectValidator:
    """Validates and filters object detections to reduce false positives"""
    
    # Common false positive patterns to filter
    FALSE_POSITIVE_PATTERNS = {
        'refrigerator': ['wall', 'door', 'cabinet', 'white surface'],
        'tv': ['wall', 'window', 'picture', 'poster', 'whiteboard'],
        'laptop': ['wall', 'window', 'picture'],
        'monitor': ['wall', 'window', 'picture'],
        'cell phone': ['wall', 'window', 'small object'],
        'remote': ['wall', 'small object'],
        'book': ['wall', 'poster', 'picture'],
        'clock': ['wall', 'picture', 'poster'],
    }
    
    # Objects that require high confidence (often misclassified)
    HIGH_CONFIDENCE_REQUIRED = {
        'refrigerator': 0.6,
        'tv': 0.6,
        'laptop': 0.55,
        'monitor': 0.55,
        'microwave': 0.55,
        'oven': 0.55,
        'kite': 0.65,  # Kites are often misclassified (statues, postures, etc.)
    }
    
    # Objects that are often too large (likely false positives)
    MAX_SIZE_RATIO = {
        'refrigerator': 0.4,  # Max 40% of image
        'tv': 0.5,  # Max 50% of image
        'laptop': 0.3,
        'monitor': 0.3,
        'kite': 0.15,  # Kites should be relatively small - large ones are likely statues/postures
    }
    
    @staticmethod
    def validate_detection(box, label, confidence, image):
        """
        Validate a single detection
        Returns: (is_valid, reason)
        """
        try:
            # Get bounding box coordinates
            x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0]]
            h, w = image.shape[:2]
            
            # Calculate box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            image_area = w * h
            size_ratio = box_area / image_area
            
            # 1. Check confidence threshold for specific objects
            if label in ObjectValidator.HIGH_CONFIDENCE_REQUIRED:
                min_confidence = ObjectValidator.HIGH_CONFIDENCE_REQUIRED[label]
                if confidence < min_confidence:
                    logger.info(f"Filtered {label}: confidence {confidence:.2f} < {min_confidence}")
                    return False, f"Low confidence for {label}"
            
            # 2. Check size constraints (too large = likely false positive)
            if label in ObjectValidator.MAX_SIZE_RATIO:
                max_ratio = ObjectValidator.MAX_SIZE_RATIO[label]
                if size_ratio > max_ratio:
                    logger.info(f"Filtered {label}: too large ({size_ratio:.2%} > {max_ratio:.2%})")
                    return False, f"{label} too large, likely false positive"
            
            # 3. Check if box is too small (likely noise) - more lenient for small objects like iron, hair drier
            # Small objects like iron, hair drier, remote, cell phone, spoon should be allowed even if small
            small_objects_allowed = ['hair drier', 'iron', 'remote', 'cell phone', 'toothbrush', 'mouse', 'keyboard', 'spoon', 'fork', 'knife']
            min_size_ratio = 0.0005 if label.lower() in small_objects_allowed else 0.001  # More lenient for small objects
            if size_ratio < min_size_ratio:
                logger.info(f"Filtered {label}: too small ({size_ratio:.2%})")
                return False, "Detection too small, likely noise"
            
            # 4. Extract region and analyze texture/variance
            x1_int = max(0, int(x1))
            y1_int = max(0, int(y1))
            x2_int = min(w, int(x2))
            y2_int = min(h, int(y2))
            
            if x2_int > x1_int and y2_int > y1_int:
                region = image[y1_int:y2_int, x1_int:x2_int]
                
                # Check for uniform areas (walls, plain surfaces)
                if ObjectValidator._is_uniform_region(region, label):
                    logger.info(f"Filtered {label}: uniform region detected (likely wall/surface)")
                    return False, "Uniform region, likely wall or plain surface"
                
                # Special check for kites - filter if it looks like a statue/posture (static, detailed)
                if label == 'kite':
                    if ObjectValidator._is_likely_statue(region):
                        logger.info(f"Filtered {label}: likely statue/posture, not a kite")
                        return False, "Kite detection likely a statue or posture"
                
                # Check for edge density (real objects have edges)
                if not ObjectValidator._has_sufficient_edges(region, label):
                    logger.info(f"Filtered {label}: insufficient edges")
                    return False, "Insufficient edge features"
            
            # 5. Check aspect ratio (some objects have specific ratios)
            aspect_ratio = box_width / box_height if box_height > 0 else 0
            if not ObjectValidator._has_valid_aspect_ratio(label, aspect_ratio):
                logger.info(f"Filtered {label}: invalid aspect ratio {aspect_ratio:.2f}")
                return False, "Invalid aspect ratio"
            
            return True, "Valid"
            
        except Exception as e:
            logger.warning(f"Validation error for {label}: {e}")
            # On error, be conservative - accept the detection
            return True, "Validation error, accepting"
    
    @staticmethod
    def _is_uniform_region(region, label):
        """Check if region is too uniform (likely wall/plain surface)"""
        try:
            if region.size == 0:
                return False
            
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Calculate standard deviation (uniform regions have low std)
            std_dev = np.std(gray)
            
            # Very uniform regions (walls, plain surfaces) have std < 15
            # But some objects like refrigerators/TVs on walls might be detected
            # So we need to be careful
            if std_dev < 10:
                # Check if it's a problematic label
                if label in ['refrigerator', 'tv', 'laptop', 'monitor']:
                    return True  # Likely a wall being misclassified
            
            # Also check variance in color channels
            if len(region.shape) == 3:
                b_std = np.std(region[:, :, 0])
                g_std = np.std(region[:, :, 1])
                r_std = np.std(region[:, :, 2])
                avg_std = (b_std + g_std + r_std) / 3
                
                if avg_std < 8 and label in ['refrigerator', 'tv', 'laptop', 'monitor']:
                    return True
            
            return False
        except:
            return False
    
    @staticmethod
    def _has_sufficient_edges(region, label):
        """Check if region has sufficient edge features"""
        try:
            if region.size == 0:
                return True  # Can't check, accept
            
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Real objects should have some edges
            # Very low edge density (< 0.05) suggests uniform surface
            if edge_density < 0.03:
                # But allow for some objects that might be smooth
                if label not in ['wall', 'floor', 'ceiling']:
                    return False
            
            return True
        except:
            return True
    
    @staticmethod
    def _is_likely_statue(region):
        """Check if detected 'kite' is likely a statue/posture (not a real kite)"""
        try:
            if region.size == 0:
                return False
            
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Statues/postures typically have:
            # 1. High detail/edges (not simple like a kite)
            # 2. More complex texture
            # 3. Higher edge density
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate texture complexity (variance)
            std_dev = np.std(gray)
            
            # Real kites are usually simple shapes with moderate detail
            # Statues/postures have very high detail (edge density > 0.15) and high variance
            # If edge density is very high (>0.15) and std_dev is high (>30), likely a statue
            if edge_density > 0.15 and std_dev > 30:
                return True
            
            # Also check if it's too detailed for a simple kite
            # Kites are usually simple geometric shapes, not complex sculptures
            if edge_density > 0.20:  # Very high detail = likely statue
                return True
            
            return False
        except:
            return False
    
    @staticmethod
    def _has_valid_aspect_ratio(label, aspect_ratio):
        """Check if aspect ratio is reasonable for the object"""
        # Most objects have reasonable aspect ratios (not too extreme)
        # Extremely wide or tall boxes are often false positives
        
        # Very extreme ratios (wider than 5:1 or taller than 5:1) are suspicious
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            # But some objects can be wide/tall
            if label in ['tv', 'monitor', 'laptop']:
                # These can be wide, but not extremely so
                if aspect_ratio > 3.0:
                    return False
            return False
        
        return True
    
    @staticmethod
    def filter_detections(results, image, min_confidence=0.4):
        """
        Filter all detections to remove false positives
        Returns: Filtered results
        """
        try:
            h, w = image.shape[:2]
            valid_detections = []
            
            for result in results:
                for box in result.boxes:
                    try:
                        cls = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = result.names[cls]
                        
                        # First, apply minimum confidence threshold
                        if confidence < min_confidence:
                            continue
                        
                        # Validate the detection
                        is_valid, reason = ObjectValidator.validate_detection(
                            box, label, confidence, image
                        )
                        
                        if is_valid:
                            valid_detections.append({
                                'box': box,
                                'label': label,
                                'confidence': confidence,
                                'cls': cls
                            })
                        else:
                            logger.debug(f"Filtered: {label} ({confidence:.2f}) - {reason}")
                    
                    except Exception as e:
                        logger.warning(f"Error filtering detection: {e}")
                        continue
            
            return valid_detections
            
        except Exception as e:
            logger.error(f"Error in filter_detections: {e}")
            # On error, return empty list (conservative)
            return []

