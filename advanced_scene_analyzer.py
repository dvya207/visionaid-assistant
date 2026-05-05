"""
Advanced Scene Analyzer - Comprehensive analysis for visually impaired users
Detects: emotions, actions, poses, clothing, environment, safety hazards, social context
"""
import cv2
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Import high-accuracy module
try:
    from high_accuracy_analyzer import high_accuracy_analyzer
    HIGH_ACC_AVAILABLE = True
except:
    HIGH_ACC_AVAILABLE = False

# Import enhanced action detector
try:
    from enhanced_action_detector import enhanced_action_detector
    ENHANCED_ACTION_AVAILABLE = True
except:
    ENHANCED_ACTION_AVAILABLE = False

class AdvancedSceneAnalyzer:
    def __init__(self):
        self.mp_face_detection = None
        self.mp_pose = None
        self.face_detector = None
        self.pose_detector = None
        self.loaded = False
        self.load_error = None
        
        # Proper pluralization for common objects
        self.irregular_plurals = {
            'person': 'people',
            'child': 'children',
            'man': 'men',
            'woman': 'women',
            'mouse': 'mice',
            'foot': 'feet',
            'tooth': 'teeth',
            'knife': 'knives',
            'leaf': 'leaves',
            'life': 'lives',
            'wife': 'wives',
            'sheep': 'sheep',
            'deer': 'deer',
            'fish': 'fish'
        }
    
    def pluralize(self, word, count):
        """Return proper plural form"""
        if count == 1:
            return word
        
        # Check irregular plurals
        if word in self.irregular_plurals:
            return self.irregular_plurals[word]
        
        # Regular plurals
        return word + 's'
    
    def load_models(self):
        """Load MediaPipe models (lazy loading) - supports both old and new APIs"""
        if self.loaded:
            return True
        
        # Import compatibility layer
        try:
            from mediapipe_compat import is_available, is_new_api, get_mediapipe_version
        except ImportError:
            # Fallback if compatibility layer not available
            def is_available():
                try:
                    import mediapipe
                    return True
                except ImportError:
                    return False
            def is_new_api():
                return False
            def get_mediapipe_version():
                try:
                    import mediapipe as mp
                    return getattr(mp, '__version__', 'unknown')
                except ImportError:
                    return None
        
        if not is_available():
            self.load_error = "MediaPipe not installed. Install with: pip install mediapipe==0.9.3.1"
            return False
        
        try:
            import mediapipe as mp
            use_new_api = is_new_api()
            mp_version = get_mediapipe_version()
            
            # Try to load legacy solutions even if modern MediaPipe
            try:
                # Legacy API (0.9.x - 0.10.x often still has solutions)
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_pose = mp.solutions.pose
                self.face_detector = self.mp_face_detection.FaceDetection(
                    model_selection=1,  # Use full range model for better accuracy
                    min_detection_confidence=0.3  # Lower threshold for better detection
                )
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,  # Maximum complexity for best accuracy
                    smooth_landmarks=True,  # Enable smoothing for better accuracy
                    enable_segmentation=False,
                    smooth_segmentation=False,
                    min_detection_confidence=0.3,  # Lower threshold to catch more poses
                    min_tracking_confidence=0.7  # Higher tracking for stability
                )
                
                self.loaded = True
                logger.info(f"MediaPipe models loaded successfully (version {mp_version})")
                return True
            except (AttributeError, Exception) as e:
                self.load_error = f"MediaPipe solutions not available: {e}"
                logger.warning(f"MediaPipe load error: {e}")
                return False
        except ImportError:
            self.load_error = "MediaPipe not installed. Install with: pip install mediapipe==0.9.3.1"
            return False
        except Exception as e:
            self.load_error = f"Error loading MediaPipe: {str(e)}"
            logger.error(f"MediaPipe load error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def analyze_environment(self, image, objects):
        """Detect indoor/outdoor and room type"""
        env_info = {}
        
        # Common outdoor objects
        outdoor_objects = {'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'airplane', 
                          'bird', 'dog', 'cat', 'tree', 'bench'}
        
        # Room-specific objects
        kitchen_objects = {'refrigerator', 'oven', 'microwave', 'sink', 'knife', 'cup', 'bowl'}
        bedroom_objects = {'bed', 'pillow'}
        office_objects = {'laptop', 'keyboard', 'mouse', 'monitor', 'book'}
        bathroom_objects = {'toilet', 'sink'}
        
        outdoor_count = sum(1 for obj in objects if obj in outdoor_objects)
        
        if outdoor_count >= 2:
            env_info['location'] = "outdoor scene"
        else:
            env_info['location'] = "indoor scene"
            
            # Detect room type
            if any(obj in objects for obj in kitchen_objects):
                env_info['room'] = "possibly a kitchen"
            elif any(obj in objects for obj in bedroom_objects):
                env_info['room'] = "possibly a bedroom"
            elif any(obj in objects for obj in office_objects):
                env_info['room'] = "possibly an office or workspace"
            elif any(obj in objects for obj in bathroom_objects):
                env_info['room'] = "possibly a bathroom"
        
        return env_info
    
    def analyze_brightness(self, image):
        """Enhanced lighting analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        lighting = {}
        
        if avg_brightness < 60:
            lighting['level'] = "very dark"
            lighting['time'] = "possibly nighttime or late evening"
        elif avg_brightness < 100:
            lighting['level'] = "dimly lit"
            lighting['time'] = "possibly early morning or dusk"
        elif avg_brightness > 200:
            lighting['level'] = "very bright"
            lighting['time'] = "possibly midday or strong artificial lighting"
        else:
            lighting['level'] = "well-lit"
        
        # High contrast might indicate strong shadows
        if std_brightness > 60:
            lighting['contrast'] = "high contrast with shadows"
        
        return lighting
    
    def estimate_distance(self, bbox, img_height):
        """Estimate relative distance based on object size and position"""
        x, y, w, h = bbox
        
        # Objects lower in frame and larger are typically closer
        bottom_y = y + h
        object_size = w * h
        image_size = img_height * img_height  # Approximate
        
        size_ratio = object_size / image_size
        
        # Bottom third of image = likely close
        if bottom_y > img_height * 0.7 and size_ratio > 0.1:
            return "very close"
        elif size_ratio > 0.15:
            return "close"
        elif size_ratio > 0.05:
            return "medium distance"
        else:
            return "far away"
    
    def detect_spatial_relationships(self, yolo_results, img_width, img_height):
        """Detect object relationships like 'person next to table'"""
        relationships = []
        objects_with_boxes = []
        
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = result.names[cls]
                x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0]]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                objects_with_boxes.append({
                    'label': label,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        # Check for "on top of" relationships (vertical stacking)
        for i, obj1 in enumerate(objects_with_boxes):
            for obj2 in objects_with_boxes[i+1:]:
                # Check if obj1 is above obj2 and horizontally aligned
                y_diff = obj2['center'][1] - obj1['center'][1]
                x_diff = abs(obj1['center'][0] - obj2['center'][0])
                
                # If one object is above another and centers are close
                if y_diff > 50 and x_diff < 100:
                    # Check if obj1 could be "on" obj2 (like cup on table)
                    if obj1['area'] < obj2['area'] * 2:  # Smaller object on larger
                        relationships.append(f"{obj1['label']} on {obj2['label']}")
        
        # Check for "next to" relationships (horizontal)
        for i, obj1 in enumerate(objects_with_boxes):
            for obj2 in objects_with_boxes[i+1:]:
                x_diff = abs(obj1['center'][0] - obj2['center'][0])
                y_diff = abs(obj1['center'][1] - obj2['center'][1])
                
                # If objects are side by side
                if x_diff > 100 and x_diff < img_width * 0.5 and y_diff < 150:
                    if obj1['label'] == 'person' or obj2['label'] == 'person':
                        other = obj2['label'] if obj1['label'] == 'person' else obj1['label']
                        relationships.append(f"person near {other}")
        
        if not relationships:
            # Fallback to simple proximity if no specific relations found
            for i, obj1 in enumerate(objects_with_boxes):
                for obj2 in objects_with_boxes[i+1:]:
                    dist_x = abs(obj1['center'][0] - obj2['center'][0])
                    dist_y = abs(obj1['center'][1] - obj2['center'][1])
                    
                    if dist_x < img_width * 0.15 and dist_y < img_height * 0.15:
                        # Determine relative position
                        if obj1['center'][0] < obj2['center'][0]:
                            rel_pos = "to the left of"
                        else:
                            rel_pos = "to the right of"
                        
                        relationships.append(f"{obj1['label']} is {rel_pos} {obj2['label']}")

        return relationships[:3]  # Limit to 3 most important
    
    def detect_doors_and_exits(self, objects):
        """Detect doors and exits"""
        door_info = []
        
        if 'door' in objects:
            door_info.append("Door detected in scene")
        
        if 'window' in objects:
            door_info.append("Window visible")
        
        return door_info
    
    def analyze_crowd_density(self, people_count, img_width, img_height):
        """Determine crowd density"""
        image_area = img_width * img_height
        
        if people_count == 0:
            return "empty, no people"
        elif people_count == 1:
            return "solitary"
        elif people_count == 2:
            return "pair"
        elif people_count <= 5:
            return "small group"
        elif people_count <= 10:
            return "moderate crowd"
        else:
            return "large crowd"
    
    def detect_vehicle_details(self, objects, yolo_results):
        """Analyze vehicle types and details"""
        vehicles = []
        vehicle_types = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'airplane', 'boat', 'train'}
        
        for obj in objects:
            if obj in vehicle_types:
                vehicles.append(obj)
        
        if vehicles:
            vehicle_counts = Counter(vehicles)
            return [f"{count} {self.pluralize(v, count)}" for v, count in vehicle_counts.items()]
        
        return []
    
    def analyze_facial_details(self, face_region):
        """Enhanced facial analysis - age, accessories"""
        details = []
        
        if face_region is None or face_region.size == 0:
            return details
        
        # Detect if wearing glasses (look for dark horizontal lines in upper half)
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        upper_half = gray[:int(gray.shape[0]*0.5), :]
        
        # Simple edge detection for glasses
        edges = cv2.Canny(upper_half, 50, 150)
        horizontal_edges = np.sum(edges, axis=1)
        
        if np.max(horizontal_edges) > upper_half.shape[1] * 0.3:
            details.append("wearing glasses")
        
        # Detect facial hair (look for texture in lower half)
        lower_half = gray[int(gray.shape[0]*0.6):, int(gray.shape[1]*0.2):int(gray.shape[1]*0.8)]
        if lower_half.size > 0:
            texture = np.std(lower_half)
            if texture > 40:
                details.append("possible facial hair")
        
        return details
    
    def analyze_scene_colors(self, image):
        """Detect dominant colors in the scene"""
        # Resize for faster processing
        small = cv2.resize(image, (100, 100))
        pixels = small.reshape(-1, 3)
        
        # Get dominant color
        dominant = np.median(pixels, axis=0)
        b, g, r = dominant
        
        colors = []
        
        # Identify dominant colors
        if r > 150 and g < 100 and b < 100:
            colors.append("reddish tones")
        elif r < 100 and g > 150 and b < 100:
            colors.append("greenish tones")
        elif r < 100 and g < 100 and b > 150:
            colors.append("bluish tones")
        elif r > 150 and g > 150 and b < 100:
            colors.append("yellowish tones")
        elif r < 80 and g < 80 and b < 80:
            colors.append("dark colors")
        elif r > 180 and g > 180 and b > 180:
            colors.append("bright or white tones")
        else:
            colors.append("mixed colors")
        
        return colors
    
    def get_directional_position(self, center_x, center_y, img_width, img_height):
        """Give directional guidance relative to camera"""
        directions = []
        
        # Horizontal
        if center_x < img_width * 0.3:
            directions.append("on your left")
        elif center_x > img_width * 0.7:
            directions.append("on your right")
        else:
            directions.append("ahead")
        
        # Vertical (for height)
        if center_y < img_height * 0.3:
            directions.append("at eye level or above")
        elif center_y > img_height * 0.7:
            directions.append("below")
        
        return " ".join(directions)
    
    def analyze_object_size(self, bbox, img_width, img_height):
        """Classify object size"""
        x, y, w, h = bbox
        area = w * h
        img_area = img_width * img_height
        ratio = area / img_area
        
        if ratio > 0.3:
            return "large"
        elif ratio > 0.1:
            return "medium-sized"
        elif ratio > 0.02:
            return "small"
        else:
            return "tiny"
    
    def detect_animal_behavior(self, objects):
        """Detect animals and possible behaviors"""
        animals = ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant']
        found_animals = [obj for obj in objects if obj in animals]
        
        if found_animals:
            animal_counts = Counter(found_animals)
            return [f"{count} {self.pluralize(animal, count)}" 
                   for animal, count in animal_counts.items()]
        return []
    
    def detect_group_activities(self, people_count, objects):
        """Infer possible group activities"""
        activities = []
        
        if people_count >= 2:
            # Conversation detection heuristically added if multiple people are present
            activities.append("possibly conversing")

            # Dining
            if any(obj in objects for obj in ['dining table', 'fork', 'knife', 'bowl', 'cup']):
                activities.append("possibly dining together")
            
            # Meeting
            if any(obj in objects for obj in ['laptop', 'book', 'chair']):
                activities.append("possibly in a meeting or studying")
            
            # Sports/recreation
            if any(obj in objects for obj in ['sports ball', 'tennis racket', 'frisbee']):
                activities.append("possibly playing sports")
            
            # TV watching
            if 'tv' in objects or 'remote' in objects:
                activities.append("possibly watching TV together")
        
        return activities
    
    def analyze_furniture_layout(self, objects, yolo_results):
        """Analyze furniture arrangement"""
        furniture = ['chair', 'couch', 'bed', 'dining table', 'desk']
        furniture_items = [obj for obj in objects if obj in furniture]
        
        if len(furniture_items) >= 3:
            return "well-furnished space"
        elif len(furniture_items) > 0:
            return "sparsely furnished"
        
        return None
    
    def detect_weather_clues(self, image, objects):
        """Detect weather indicators"""
        clues = []
        
        # Check for umbrella (rain indicator)
        if 'umbrella' in objects:
            clues.append("umbrella present - possibly rainy")
        
        # Check overall image for wet/gray tones (simplified)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg = np.mean(gray)
        std = np.std(gray)
        
        # Very uniform gray = possibly foggy/cloudy
        if std < 30 and 80 < avg < 150:
            clues.append("possibly cloudy or foggy weather")
        
        return clues
    
    def detect_safety_hazards(self, objects, yolo_results, img_width, img_height):
        """Detect potential safety concerns - ONLY CRITICAL ALERTS"""
        hazards = []
        
        # Check for stairs
        if 'stairs' in objects:
            hazards.append("ALERT: Stairs detected")
        
        # Check for fire/smoke
        if 'fire hydrant' in objects:
            hazards.append("Fire hydrant nearby")
        
        # REMOVED: "Obstacle ahead" for people - not a critical safety issue
        # Only report critical safety hazards like stairs
        
        return hazards
    
    def analyze_clothing_colors(self, image, person_bbox):
        """Detect dominant clothing colors"""
        try:
            x, y, w, h = person_bbox
            person_region = image[y:y+h, x:x+w]
            
            if person_region.size == 0:
                return "unknown"
            
            # Get dominant color
            pixels = person_region.reshape(-1, 3)
            dominant_color = np.median(pixels, axis=0)
            
            # Simple color naming
            b, g, r = dominant_color
            if r > 150 and g < 100 and b < 100:
                return "wearing red"
            elif r < 100 and g > 150 and b < 100:
                return "wearing green"
            elif r < 100 and g < 100 and b > 150:
                return "wearing blue"
            elif r > 150 and g > 150 and b < 100:
                return "wearing yellow"
            elif r < 80 and g < 80 and b < 80:
                return "wearing dark clothing"
            elif r > 180 and g > 180 and b > 180:
                return "wearing white or light clothing"
            else:
                return "wearing colored clothing"
        except:
            return ""
    
    def detect_social_context(self, faces, poses):
        """Analyze social interaction"""
        social = []
        
        if len(faces) == 0:
            social.append("No people visible")
        elif len(faces) == 1:
            social.append("One person alone")
        elif len(faces) == 2:
            social.append("Two people - possibly conversing")
        elif len(faces) >= 3:
            social.append(f"{len(faces)} people - group gathering")
        
        return social
    
    def analyze_detailed_pose(self, landmarks, image_shape):
        """Improved pose and action detection with better accuracy"""
        actions = []
        
        try:
            # Validate landmarks visibility
            def is_visible(lm):
                return hasattr(lm, 'visibility') and lm.visibility > 0.5
            
            # Get key landmarks with validation
            nose = landmarks[0] if len(landmarks) > 0 and is_visible(landmarks[0]) else None
            left_wrist = landmarks[15] if len(landmarks) > 15 and is_visible(landmarks[15]) else None
            right_wrist = landmarks[16] if len(landmarks) > 16 and is_visible(landmarks[16]) else None
            left_shoulder = landmarks[11] if len(landmarks) > 11 and is_visible(landmarks[11]) else None
            right_shoulder = landmarks[12] if len(landmarks) > 12 and is_visible(landmarks[12]) else None
            left_hip = landmarks[23] if len(landmarks) > 23 and is_visible(landmarks[23]) else None
            right_hip = landmarks[24] if len(landmarks) > 24 and is_visible(landmarks[24]) else None
            left_knee = landmarks[25] if len(landmarks) > 25 and is_visible(landmarks[25]) else None
            right_knee = landmarks[26] if len(landmarks) > 26 and is_visible(landmarks[26]) else None
            left_elbow = landmarks[13] if len(landmarks) > 13 and is_visible(landmarks[13]) else None
            right_elbow = landmarks[14] if len(landmarks) > 14 and is_visible(landmarks[14]) else None
            
            # Need at least basic landmarks
            if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
                return "standing", []
            
            # IMPROVED pose detection with angle calculations
            hip_y = (left_hip.y + right_hip.y) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Calculate body angle for better pose detection
            def calculate_angle(p1, p2, p3):
                """Calculate angle between three points"""
                try:
                    a = np.array([p1.x, p1.y])
                    b = np.array([p2.x, p2.y])
                    c = np.array([p3.x, p3.y])
                    ba = a - b
                    bc = c - b
                    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                except:
                    return 90
            
            # Determine pose with improved logic
            if left_knee and right_knee:
                knee_y = (left_knee.y + right_knee.y) / 2
                leg_angle = calculate_angle(left_hip, left_knee, landmarks[27] if len(landmarks) > 27 else left_knee)
                
                # Sitting: hips below knees, knees bent
                if hip_y > knee_y + 0.03 and leg_angle < 140:
                    pose = "sitting"
                # Standing: hips above knees, body upright
                elif hip_y < knee_y - 0.03 and shoulder_y < hip_y:
                    pose = "standing"
                else:
                    pose = "standing"  # Default
            else:
                # Fallback without knee data
                if hip_y > 0.75:
                    pose = "sitting"
                elif shoulder_y < 0.3:
                    pose = "lying or crouching"
                else:
                    pose = "standing"
            
            # ENHANCED action detection - eating/drinking AND hands covering face
            if left_wrist and right_wrist:
                mouth_y = nose.y + 0.06  # More accurate mouth position
                mouth_x = nose.x
                face_center_y = nose.y
                face_center_x = nose.x
                
                # Calculate normalized distance from hand to face
                def hand_to_face_dist(wrist, face_y, face_x):
                    dy = abs(wrist.y - face_y)
                    dx = abs(wrist.x - face_x)
                    return (dy**2 + dx**2)**0.5
                
                left_dist = hand_to_face_dist(left_wrist, face_center_y, face_center_x) if left_wrist else 1.0
                right_dist = hand_to_face_dist(right_wrist, face_center_y, face_center_x) if right_wrist else 1.0
                
                # DETECT HANDS COVERING FACE (hands very close to face center)
                left_covering_face = (left_dist < 0.12 and 
                                      left_wrist.y < nose.y + 0.1 and
                                      left_wrist.y > nose.y - 0.15)
                
                right_covering_face = (right_dist < 0.12 and 
                                       right_wrist.y < nose.y + 0.1 and
                                       right_wrist.y > nose.y - 0.15)
                
                if left_covering_face or right_covering_face:
                    if left_covering_face and right_covering_face:
                        actions.append("Person covering face with hands")
                    elif left_covering_face:
                        actions.append("Person covering face with left hand")
                    else:
                        actions.append("Person covering face with right hand")
                
                # Hand near mouth AND raised (above shoulder level) - eating/drinking
                left_dist_mouth = hand_to_face_dist(left_wrist, mouth_y, mouth_x) if left_wrist else 1.0
                right_dist_mouth = hand_to_face_dist(right_wrist, mouth_y, mouth_x) if right_wrist else 1.0
                
                left_eating = (left_dist_mouth < 0.16 and 
                              left_wrist.y < left_shoulder.y + 0.08 and
                              left_wrist.y > nose.y - 0.15)
                
                right_eating = (right_dist_mouth < 0.16 and 
                               right_wrist.y < right_shoulder.y + 0.08 and
                               right_wrist.y > nose.y - 0.15)
                
                if (left_eating or right_eating) and not (left_covering_face or right_covering_face):
                    actions.append("Person eating or drinking")
            
            # IMPROVED waving detection
            if left_wrist and right_wrist:
                left_raised = left_wrist.y < left_shoulder.y - 0.12
                right_raised = right_wrist.y < right_shoulder.y - 0.12
                
                if left_raised or right_raised:
                    # Check arm extension for waving
                    if left_raised and left_elbow:
                        arm_ext = abs(left_wrist.x - left_shoulder.x)
                        if arm_ext > 0.12:
                            actions.append("Person waving or gesturing")
                    elif right_raised and right_elbow:
                        arm_ext = abs(right_wrist.x - right_shoulder.x)
                        if arm_ext > 0.12:
                            actions.append("Person waving or gesturing")
            
            # Add pose if no specific actions detected
            if not actions:
                actions.append(f"Person {pose}")
            
        except Exception as e:
            logger.debug(f"Pose analysis error: {e}")
            return "standing", []
        
        return pose, actions
    
    def detect_eating_from_objects(self, objects, poses):
        """Infer eating from objects present"""
        eating_objects = {'cup', 'bowl', 'fork', 'knife', 'spoon', 'bottle', 'wine glass', 'sandwich'}
        
        has_eating_objects = any(obj in objects for obj in eating_objects)
        
        if has_eating_objects and len(poses) > 0:
            return "possibly eating or drinking (utensils/food items visible)"
        
        return None
    
    def generate_description(self, image, yolo_results):
        """
        Generate SIMPLE scene description with object detection
        Returns: Simple list of detected objects like "I see: person, chair, table, cup"
        """
        parts = []
        person_positions = [] # FIX: Initialize early to avoid UnboundLocalError
        
        # Extract objects and their positions quickly
        objects = []
        object_positions = []  # (label, center_x, center_y, bbox)
        h, w, _ = image.shape
        
        # Store w and h for nested functions to access (closure)
        img_width, img_height = w, h
        
        # Collect all detections first, then sort by confidence for CONSISTENCY
        # This ensures the same image always produces the same results
        all_detections = []
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = result.names[cls]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0]]
                all_detections.append({
                    'box': box,
                    'cls': cls,
                    'label': label,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })
        
        # Sort by confidence (highest first), then by label for consistent processing order
        # This ensures deterministic results - same image = same detection order
        all_detections.sort(key=lambda x: (-x['confidence'], x['label']))
        
        for det in all_detections:
            box = det['box']
            label = det['label']
            confidence = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            # Object detection thresholds - CRITICAL: Lower thresholds for better scene detection
            # Initialize default threshold - LOWERED for better object detection
            min_confidence = 0.25  # Lowered from 0.35 - better scene detection
            
            # Person detection - Balanced threshold for consistency
            # YOLO already filtered at 0.25, so we accept people at 0.20+ for stability
            if label == 'person':
                min_confidence = 0.20  # Balanced threshold - ensures people detected while maintaining consistency
            else:
                # Common/simple objects can be detected with lower threshold for scene description
                # NOTE: we intentionally do NOT include 'dog' here to avoid hallucinated dogs
                common_objects = ['chair', 'table', 'dining table', 'desk', 'cup', 'bottle', 
                                 'cell phone', 'laptop', 'book', 'cat', 'car', 'bicycle',
                                 'couch', 'sofa', 'bench', 'bed', 'tv', 'monitor', 'keyboard', 'mouse',
                                 'backpack', 'handbag', 'umbrella', 'suitcase', 'sports ball',
                                 'potted plant', 'vase', 'clock', 'refrigerator', 'microwave', 'oven']
                
                # Very high threshold for easily confused objects (often hallucinated)
                # Add 'dog', 'hot dog', and 'donut' here because user often sees false detections for these.
                easily_confused = ['bird', 'airplane', 'kite', 'helicopter', 'dog', 'hot dog', 'donut']
                # Even higher threshold for small/easily confused objects
                small_objects = ['scissors', 'knife', 'fork', 'spoon', 'toothbrush', 'hair drier', 'toilet paper', 'pen', 'pencil']
                if label in easily_confused:
                    min_confidence = 0.55  # Very high threshold - birds/airplanes often hallucinated
                elif label in small_objects:
                    min_confidence = 0.35  # Lowered from 0.40 - better detection of small objects
                elif label in common_objects:
                    min_confidence = 0.20  # Lowered from 0.25 - better scene detection
            
            if confidence < min_confidence:
                continue
            
            # Additional validation: filter out objects that are too small (likely false positives)
            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / (w * h)
            
            # Extra strict filtering for easily confused objects (birds, airplanes, etc.)
            if label in ['bird', 'airplane', 'kite', 'helicopter']:
                # Birds must be reasonably sized (at least 0.3% of image) AND high confidence
                if area_ratio < 0.003 or confidence < 0.55:
                    logger.debug(f"Filtered out likely false positive: {label} (area: {area_ratio:.4f}, conf: {confidence:.2f})")
                    continue
            
            # Skip very small detections (likely false positives)
            # CRITICAL: Never skip person detections - always include them
            # RELAXED: Lower area threshold for better object detection (0.05% instead of 0.1%)
            if area_ratio < 0.0005 and label != 'person':  # 0.05% of image, but always include people
                continue
                
            objects.append(label)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            object_positions.append({
                'label': label,
                'center': (center_x, center_y),
                'bbox': (x1, y1, x2, y2),
                'area_ratio': area_ratio,
                'confidence': confidence
            })
        
        # CRITICAL: Even if no objects pass filters, check if YOLO detected anything
        # This ensures we always provide some description if YOLO found objects
        if not objects:
            # Check if YOLO actually detected anything (even if filtered)
            yolo_detected_anything = False
            for result in yolo_results:
                if len(result.boxes) > 0:
                    yolo_detected_anything = True
                    break
            
            if yolo_detected_anything:
                # YOLO detected something but filters removed it - provide basic description
                logger.warning("YOLO detected objects but filters removed them - providing basic description")
                # Try to get at least person detection
                person_boxes = [box for r in yolo_results for box in r.boxes if r.names[int(box.cls[0])] == 'person']
                if person_boxes:
                    return "I see a person in the scene."
                else:
                    return "I can see some objects in the scene, but I'm having trouble identifying them clearly."
            else:
                return "I don't see any recognizable objects in this scene."
        
        # Filter out common false positives before counting
        filtered_objects = []
        filtered_object_positions = []
        # Common hallucinations / label confusions:
        # - small shiny things → scissors / knife / fork
        # - random textures / patterns → bird
        # - cups / bottles sometimes mis-labeled as vases
        # - shapes/patterns sometimes mis-labeled as dog / hot dog / donut
        false_positive_keywords = ['scissors', 'knife', 'fork', 'bird', 'vase', 'dog', 'hot dog', 'donut']
        
        for obj_pos in object_positions:
            label = obj_pos['label']
            # Skip if it's a known false positive with low area or low confidence
            if label in false_positive_keywords:
                area_ratio = obj_pos.get('area_ratio', 0)
                conf_val = obj_pos.get('confidence', 0)
                if label == 'bird':
                    # Extra strict for birds - require high confidence AND reasonable size
                    if area_ratio < 0.003 or conf_val < 0.55:
                        logger.debug(f"Filtered out false positive bird: (area: {area_ratio:.4f}, conf: {conf_val:.2f})")
                        continue
                elif label in ['dog', 'hot dog', 'donut']:
                    # Dogs / hot dogs / donuts: require larger size and higher confidence
                    if area_ratio < 0.005 or conf_val < 0.65:
                        logger.debug(f"Filtered out likely false dog/hot dog/donut: {label} (area: {area_ratio:.4f}, conf: {conf_val:.2f})")
                        continue
                elif label == 'vase':
                    # Vases should be reasonably big; small ones are often actually cups
                    if area_ratio < 0.004 or conf_val < 0.55:
                        logger.debug(f"Filtered out likely false vase (maybe cup): (area: {area_ratio:.4f}, conf: {conf_val:.2f})")
                        continue
                elif area_ratio < 0.005 or conf_val < 0.45:
                    logger.debug(f"Filtered out false positive: {label} (area: {area_ratio:.4f}, conf: {conf_val:.2f})")
                    continue
            filtered_objects.append(label)
            filtered_object_positions.append(obj_pos)
        
        objects = filtered_objects
        object_positions = filtered_object_positions
        
        object_counts = Counter(objects)
        
        # Use person boxes - CRITICAL: Include ALL person detections, even small ones
        # Lower area threshold to ensure people are detected even if far away
        people = [obj for obj in object_positions
                  if obj['label'] == 'person' and obj.get('area_ratio', 0) >= 0.005]  # Reduced from 0.02 to 0.005 (0.5%)
        
        # Additional check: if we have multiple people, verify they're not duplicates
        if len(people) > 1:
            # Check for overlapping person boxes (likely same person detected twice)
            unique_people = []
            for person in people:
                is_duplicate = False
                for existing in unique_people:
                    # Check if boxes overlap significantly
                    p1_box = person['bbox']
                    p2_box = existing['bbox']
                    # Calculate overlap ratio
                    x1_1, y1_1, x2_1, y2_1 = p1_box
                    x1_2, y1_2, x2_2, y2_2 = p2_box
                    overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
                    overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
                    overlap_area = overlap_x * overlap_y
                    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                    overlap_ratio = overlap_area / area1 if area1 > 0 else 0
                    if overlap_ratio > 0.5:  # More than 50% overlap = likely duplicate
                        is_duplicate = True
                        logger.debug(f"Filtered duplicate person detection (overlap: {overlap_ratio:.2f})")
                        break
                if not is_duplicate:
                    unique_people.append(person)
            people = unique_people
        
        people_count = len(people)
        
        # 1. SIMPLE ACTION DETECTION - Focus on reliable, common actions
        actions = []
        
        # SIMPLIFIED Helper function to check if object is near person
        def is_object_near_person(person_center, person_bbox, obj_center, obj_bbox, max_distance_ratio=0.35):
            """Check if object is near person with improved accuracy"""
            try:
                px, py = person_center
                ox, oy = obj_center
                
                # Calculate normalized distance (considering image dimensions)
                # Use w and h from outer scope - use the stored values
                img_w, img_h = img_width, img_height
            except NameError as e:
                logger.error(f"NameError in is_object_near_person: {e}, w={w if 'w' in locals() else 'undefined'}, h={h if 'h' in locals() else 'undefined'}")
                raise
            dx = abs(px - ox) / img_w
            dy = abs(py - oy) / img_h
            normalized_distance = (dx**2 + dy**2)**0.5
            
            # More accurate distance threshold - stricter to reduce false positives
            max_distance = max_distance_ratio
            
            # Check horizontal proximity
            horizontal_close = normalized_distance < max_distance
            
            # IMPROVED vertical check: object should be in reasonable vertical range
            # For eating: hands can be at various heights, so more lenient
            # For sitting: object should be near person's upper body
            py_bbox = person_bbox[1]  # Top of person bbox
            py_bbox_bottom = person_bbox[3] if len(person_bbox) > 3 else person_bbox[1] + (person_bbox[3] - person_bbox[1]) if len(person_bbox) > 3 else py + img_h * 0.3
            oy_bbox = obj_bbox[1]  # Top of object bbox
            
            # Vertical range: object can be from person's head level to below waist
            vertical_reasonable = (oy_bbox >= py_bbox - img_h * 0.15 and  # Not too far above head
                                  oy_bbox <= py_bbox_bottom + img_h * 0.2)  # Not too far below waist
            
            # Also check if bounding boxes overlap or are very close
            px1, py1 = person_bbox[0], person_bbox[1]
            px2, py2 = (person_bbox[2] if len(person_bbox) > 2 else px1 + img_w * 0.2,
                       person_bbox[3] if len(person_bbox) > 3 else py1 + img_h * 0.4)
            
            ox1, oy1 = obj_bbox[0], obj_bbox[1]
            ox2, oy2 = (obj_bbox[2] if len(obj_bbox) > 2 else ox1 + img_w * 0.1,
                        obj_bbox[3] if len(obj_bbox) > 3 else oy1 + img_h * 0.1)
            
            # Check for bbox overlap or close proximity
            overlap_x = max(0, min(px2, ox2) - max(px1, ox1))
            overlap_y = max(0, min(py2, oy2) - max(py1, oy1))
            has_overlap = overlap_x > 0 and overlap_y > 0
            
            # Close proximity even without overlap
            close_proximity = (abs(px1 - ox2) < img_w * 0.15 or abs(ox1 - px2) < img_w * 0.15) and \
                             (abs(py1 - oy2) < img_h * 0.2 or abs(oy1 - py2) < img_h * 0.2)
            
            return (horizontal_close and vertical_reasonable) or has_overlap or close_proximity
        
        # Helper function to check if person is interacting with object
        def is_person_interacting(person_bbox, obj_bbox, interaction_type='general'):
            """Check if person bbox overlaps or is very close to object bbox"""
            # Use w and h from outer scope - use the stored values
            img_w, img_h = img_width, img_height
            px1, py1, px2, py2 = person_bbox
            ox1, oy1, ox2, oy2 = obj_bbox
            
            # Calculate overlap
            overlap_x = max(0, min(px2, ox2) - max(px1, ox1))
            overlap_y = max(0, min(py2, oy2) - max(py1, oy1))
            overlap_area = overlap_x * overlap_y
            
            person_area = (px2 - px1) * (py2 - py1)
            obj_area = (ox2 - ox1) * (oy2 - oy1)
            
            # Check if there's significant overlap or very close proximity
            if interaction_type == 'sitting':
                # For sitting, person should overlap with furniture OR be very close
                # Lower threshold to 10% overlap for better detection
                overlap_ratio = overlap_area / person_area if person_area > 0 else 0
                # Also check if person is close to furniture (within 20% of image width/height)
                person_center_x = (px1 + px2) / 2
                person_center_y = (py1 + py2) / 2
                obj_center_x = (ox1 + ox2) / 2
                obj_center_y = (oy1 + oy2) / 2
                distance = ((person_center_x - obj_center_x)**2 + (person_center_y - obj_center_y)**2)**0.5
                max_distance = (img_w + img_h) * 0.15  # 15% of average dimension
                return overlap_ratio > 0.1 or (distance < max_distance and abs(person_center_y - obj_center_y) < img_h * 0.3)
            elif interaction_type == 'using' or interaction_type == 'eating':
                # For using/eating, object should be near person's upper body (hands area)
                person_mid_y = (py1 + py2) / 2
                obj_mid_y = (oy1 + oy2) / 2
                # More lenient for eating - hands move, so allow more vertical range
                vertical_close = abs(person_mid_y - obj_mid_y) < img_h * 0.25  # Increased from 0.15
                horizontal_close = abs((px1 + px2)/2 - (ox1 + ox2)/2) < img_w * 0.3
                return (vertical_close and horizontal_close) or overlap_area > 0
            else:
                # General: close proximity
                return overlap_area > 0 or (abs(px1 - ox2) < img_w * 0.1 and abs(py1 - oy2) < img_h * 0.1)
        
        benches_chairs = [obj for obj in object_positions if obj['label'] in ['bench', 'chair', 'couch', 'sofa']]
        
        # DEBUG: Log what we detected
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"DEBUG: Found {len(people)} people, {len(object_positions)} total objects")
        logger.info(f"DEBUG: Objects detected: {[obj['label'] for obj in object_positions[:10]]}")
        print(f"DEBUG: Found {len(people)} people, {len(object_positions)} total objects")
        print(f"DEBUG: Objects detected: {[obj['label'] for obj in object_positions[:10]]}")
        logger.info(f"DEBUG: Found {len(benches_chairs)} furniture items: {[f['label'] for f in benches_chairs]}")
        logger.info(f"DEBUG: Actions list initialized (empty): {actions}")
        
        # SIMPLE ACTION 1: Person sitting (most reliable)
        # Only use objects with high confidence to avoid false positives
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for furniture in benches_chairs:
                # CRITICAL: Only use furniture detected with high confidence
                if furniture.get('confidence', 0) < 0.40:
                    continue
                f_bbox = furniture['bbox']
                is_interacting = is_person_interacting(p_bbox, f_bbox, 'sitting')
                if is_interacting:
                    furniture_name = furniture['label']
                    actions.append(f"Person sitting on {furniture_name}")
                    logger.info(f"✅ Detected: Person sitting on {furniture_name} (conf: {furniture.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 2: Person using phone (very reliable)
        # CRITICAL: Only use phones detected with high confidence
        phone_like = [obj for obj in object_positions 
                     if obj['label'] in ['cell phone'] and obj.get('confidence', 0) >= 0.45]
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            px1, py1, px2, py2 = p_bbox
            
            for item in phone_like:
                item_center = item['center']
                item_bbox = item['bbox']
                ox1, oy1, ox2, oy2 = item_bbox
                
                # Stricter check: phone near person's upper body
                horizontal_close = abs(px - item_center[0]) < w * 0.25  # Reduced from 0.4
                vertical_close = oy1 >= py1 - h * 0.05 and oy1 <= py1 + (py2 - py1) * 0.4  # Stricter vertical
                
                # Check overlap - require actual overlap
                overlap_x = max(0, min(px2, ox2) - max(px1, ox1))
                overlap_y = max(0, min(py2, oy2) - max(py1, oy1))
                has_overlap = overlap_x > 0 and overlap_y > 0
                
                # Require BOTH overlap AND proper positioning for accuracy
                if has_overlap and (horizontal_close and vertical_close):
                    actions.append("Person using phone")
                    logger.info(f"✅ Detected: Person using phone (conf: {item.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 7: Person using laptop/computer (reliable)
        # CRITICAL: Only use devices detected with high confidence
        devices = [obj for obj in object_positions 
                  if obj['label'] in ['laptop', 'keyboard', 'mouse'] 
                  and obj.get('confidence', 0) >= 0.45]
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for device in devices:
                device_center = device['center']
                device_bbox = device['bbox']
                # Stricter distance threshold
                if is_object_near_person((px, py), p_bbox, device_center, device_bbox, 0.25):  # Reduced from 0.35
                    device_name = "computer" if device['label'] in ['laptop', 'keyboard', 'mouse'] else device['label']
                    actions.append(f"Person using {device_name}")
                    logger.info(f"✅ Detected: Person using {device_name} (conf: {device.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 3: Person reading (reliable)
        # CRITICAL: Only use reading materials detected with high confidence
        reading_materials = [obj for obj in object_positions 
                           if obj['label'] in ['book', 'newspaper', 'magazine']
                           and obj.get('confidence', 0) >= 0.40]
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for material in reading_materials:
                mat_center = material['center']
                mat_bbox = material['bbox']
                # Stricter distance threshold
                if is_object_near_person((px, py), p_bbox, mat_center, mat_bbox, 0.25):  # Reduced from 0.3
                    actions.append(f"Person reading {material['label']}")
                    logger.info(f"✅ Detected: Person reading {material['label']} (conf: {material.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 4: Person writing (reliable)
        # CRITICAL: Only use writing tools detected with high confidence
        writing_tools = [obj for obj in object_positions 
                       if obj['label'] in ['pen', 'pencil']
                       and obj.get('confidence', 0) >= 0.50]  # High threshold for small objects
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for tool in writing_tools:
                tool_center = tool['center']
                tool_bbox = tool['bbox']
                # Stricter distance threshold
                if is_object_near_person((px, py), p_bbox, tool_center, tool_bbox, 0.25):  # Reduced from 0.3
                    actions.append("Person writing")
                    logger.info(f"✅ Detected: Person writing (conf: {tool.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 5: Person at table (reliable)
        # CRITICAL: Only use tables detected with high confidence
        tables = [obj for obj in object_positions 
                 if obj['label'] in ['dining table', 'table', 'desk']
                 and obj.get('confidence', 0) >= 0.40]
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for table in tables:
                t_bbox = table['bbox']
                tx1, ty1, tx2, ty2 = t_bbox
                # Stricter check: person horizontally aligned with table
                if tx1 - w*0.15 <= px <= tx2 + w*0.15 and ty1 <= py <= ty2 + h*0.25:  # Reduced margins
                    # Check for laptop/computer (working) - with confidence checks
                    devices_near = [obj for obj in object_positions 
                                   if obj['label'] in ['laptop', 'keyboard', 'mouse'] 
                                   and obj.get('confidence', 0) >= 0.45
                                   and is_object_near_person((px, py), p_bbox, obj['center'], obj['bbox'], 0.25)]
                    
                    if devices_near:
                        actions.append("Person working at table")
                        logger.info(f"✅ Detected: Person working at table (conf: {table.get('confidence', 0):.2f})")
                    else:
                        actions.append("Person at table")
                        logger.info(f"✅ Detected: Person at table (conf: {table.get('confidence', 0):.2f})")
                    break
        
        # Detect person with pet/animal
        # CRITICAL: Only use pets detected with high confidence
        # Birds require even higher confidence (0.55) due to frequent hallucinations
        pets = []
        for obj in object_positions:
            if obj['label'] in ['dog', 'cat', 'horse', 'cow'] and obj.get('confidence', 0) >= 0.40:
                pets.append(obj)
            elif obj['label'] == 'bird' and obj.get('confidence', 0) >= 0.55 and obj.get('area_ratio', 0) >= 0.003:
                # Birds need higher confidence AND reasonable size
                pets.append(obj)
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for pet in pets:
                pet_center = pet['center']
                pet_bbox = pet['bbox']
                # Stricter distance threshold
                if is_object_near_person((px, py), p_bbox, pet_center, pet_bbox, 0.30):  # Reduced from 0.35
                    actions.append(f"Person with {pet['label']}")
                    logger.info(f"✅ Detected: Person with {pet['label']} (conf: {pet.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 6: Person eating (reliable)
        # CRITICAL: Only use food items detected with high confidence
        food_items = [obj for obj in object_positions 
                     if obj['label'] in [
                         'banana', 'apple', 'sandwich', 'pizza', 'hot dog', 'donut', 'cake',
                         'bowl', 'fork', 'knife', 'spoon'
                     ]
                     and obj.get('confidence', 0) >= 0.45]  # Higher threshold for utensils
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for food in food_items:
                food_center = food['center']
                food_bbox = food['bbox']
                # Food should be near person's upper body (hands area) - stricter
                if is_object_near_person((px, py), p_bbox, food_center, food_bbox, 0.25):  # Reduced from 0.35
                    if food['label'] in ['bowl', 'fork', 'knife', 'spoon']:
                        actions.append("Person eating")
                    else:
                        actions.append(f"Person eating {food['label']}")
                    logger.info(f"✅ Detected: Person eating (conf: {food.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 7: Person holding cup/bottle (reliable)
        # CRITICAL: Only use drink items detected with high confidence
        drink_items = [obj for obj in object_positions 
                      if obj['label'] in ['cup', 'bottle', 'wine glass']
                      and obj.get('confidence', 0) >= 0.45]
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for drink in drink_items:
                drink_center = drink['center']
                drink_bbox = drink['bbox']
                # Cup/bottle should be near person's upper body - stricter
                if is_object_near_person((px, py), p_bbox, drink_center, drink_bbox, 0.25):  # Reduced from 0.35
                    actions.append(f"Person holding {drink['label']}")
                    logger.info(f"✅ Detected: Person holding {drink['label']} (conf: {drink.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 8: Person holding electronic gadgets (reliable)
        electronic_gadgets = [obj for obj in object_positions 
                            if obj['label'] in [
                                'cell phone', 'laptop', 'keyboard', 'mouse', 'remote', 'tv', 'monitor', 'tablet'
                            ]
                            and obj.get('confidence', 0) >= 0.45]
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for gadget in electronic_gadgets:
                gadget_center = gadget['center']
                gadget_bbox = gadget['bbox']
                # Gadgets should be near person's upper body - stricter
                if is_object_near_person((px, py), p_bbox, gadget_center, gadget_bbox, 0.25):  # Reduced from 0.35
                    gadget_name = "phone" if gadget['label'] == 'cell phone' else gadget['label']
                    actions.append(f"Person holding {gadget_name}")
                    logger.info(f"✅ Detected: Person holding {gadget_name} (conf: {gadget.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 9: Person holding paper/document (reliable)
        # CRITICAL: Only use paper items detected with high confidence
        paper_items = [obj for obj in object_positions 
                      if obj['label'] in ['book', 'newspaper', 'magazine']
                      and obj.get('confidence', 0) >= 0.40]
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for paper in paper_items:
                paper_center = paper['center']
                paper_bbox = paper['bbox']
                # Stricter distance threshold
                if is_object_near_person((px, py), p_bbox, paper_center, paper_bbox, 0.25):  # Reduced from 0.3
                    actions.append(f"Person holding {paper['label']}")
                    logger.info(f"✅ Detected: Person holding {paper['label']} (conf: {paper.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 10: Person holding pen/pencil (reliable - separate from writing)
        # CRITICAL: Only use writing tools detected with high confidence
        writing_tools_holding = [obj for obj in object_positions 
                                if obj['label'] in ['pen', 'pencil']
                                and obj.get('confidence', 0) >= 0.50]  # High threshold for small objects
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for tool in writing_tools_holding:
                tool_center = tool['center']
                tool_bbox = tool['bbox']
                # Stricter distance threshold
                if is_object_near_person((px, py), p_bbox, tool_center, tool_bbox, 0.25):  # Reduced from 0.3
                    actions.append(f"Person holding {tool['label']}")
                    logger.info(f"✅ Detected: Person holding {tool['label']} (conf: {tool.get('confidence', 0):.2f})")
                    break
        
        # SIMPLE ACTION 11: Person holding/carrying other objects (reliable)
        # CRITICAL: Only use portable objects detected with high confidence
        portable_objects = [obj for obj in object_positions 
                          if obj['label'] in ['handbag', 'backpack', 'suitcase', 'umbrella', 'sports ball']
                          and obj.get('confidence', 0) >= 0.40]
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for obj in portable_objects:
                obj_center = obj['center']
                obj_bbox = obj['bbox']
                # Stricter distance threshold
                if is_object_near_person((px, py), p_bbox, obj_center, obj_bbox, 0.25):  # Reduced from 0.3
                    action_word = "carrying" if obj['label'] in ['handbag', 'backpack', 'suitcase'] else "holding"
                    actions.append(f"Person {action_word} {obj['label']}")
                    logger.info(f"✅ Detected: Person {action_word} {obj['label']} (conf: {obj.get('confidence', 0):.2f})")
                    break
        
        # Detect person near vehicle (getting in/out)
        vehicles = [obj for obj in object_positions if obj['label'] in [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle'
        ]]
        for person in people:
            px, py = person['center']
            p_bbox = person['bbox']
            for vehicle in vehicles:
                v_center = vehicle['center']
                v_bbox = vehicle['bbox']
                if is_object_near_person((px, py), p_bbox, v_center, v_bbox, 0.3):
                    actions.append(f"Person near {vehicle['label']}")
                    break
        
        # 2. ENHANCED POSE DETECTION (ONLY IF PEOPLE PRESENT)
        if people_count > 0:
            # Try enhanced action detector first (MediaPipe Holistic - BEST for actions)
            try:
                if ENHANCED_ACTION_AVAILABLE:
                    # Use enhanced action detector for each person
                    for person in people[:2]:  # Only first 2 people
                        p_bbox = person['bbox']
                        px1, py1, px2, py2 = p_bbox
                        # Convert to (x, y, w, h) format
                        person_bbox = (px1, py1, px2 - px1, py2 - py1)
                        
                        enhanced_actions = enhanced_action_detector.detect_actions_from_pose(image, person_bbox)
                        if enhanced_actions:
                            # Format actions nicely
                            for act in enhanced_actions:
                                formatted_act = f"Person {act}" if not act.startswith("Person") else act
                                if formatted_act.lower() not in [a.lower() for a in actions]:
                                    actions.append(formatted_act)
                            logger.info(f"Enhanced action detector found: {enhanced_actions}")
            except Exception as e:
                logger.warning(f"Enhanced action detection failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # Fallback to standard pose detection (MediaPipe Pose)
            try:
                poses = self.analyze_pose(image)
                if poses:
                    print(f"DEBUG: Pose detection found {len(poses)} poses")
                    for i, pose_data in enumerate(poses[:2]):  # Only first 2 people
                        pose_desc, pose_actions = self.analyze_detailed_pose(pose_data['landmarks'], image.shape)
                        print(f"DEBUG: Pose {i} - desc: {pose_desc}, actions: {pose_actions}")
                        if pose_actions:
                            # Only add if not already detected by enhanced detector
                            for act in pose_actions:
                                if act.lower() not in [a.lower() for a in actions]:
                                    actions.append(act)
                                    print(f"DEBUG: Added action from pose: {act}")
                else:
                    print(f"DEBUG: No poses detected")
            except Exception as e:
                print(f"DEBUG: Standard pose detection failed: {e}")
                logger.warning(f"Standard pose detection failed: {e}")
                import traceback
                traceback.print_exc()
            
            # IMPROVED: Also detect eating from objects even without pose detection
            # CRITICAL: Only use objects with high confidence to avoid false positives
            print(f"DEBUG: Current actions before object check: {actions}")
            if not any('eating' in a.lower() or 'drinking' in a.lower() for a in actions):
                eating_objects_near = []
                for person in people:
                    px, py = person['center']
                    p_bbox = person['bbox']
                    for obj in object_positions:
                        # CRITICAL: Require high confidence for eating-related objects
                        if obj.get('confidence', 0) < 0.45:
                            continue
                        if obj['label'] in ['cup', 'bottle', 'bowl', 'fork', 'knife', 'spoon', 
                                           'banana', 'apple', 'sandwich', 'pizza', 'hot dog', 'donut', 'cake']:
                            obj_center = obj['center']
                            obj_bbox = obj['bbox']
                            # Stricter: 0.25 distance ratio (reduced from 0.5) for accurate detection
                            if is_object_near_person((px, py), p_bbox, obj_center, obj_bbox, 0.25):
                                eating_objects_near.append(obj['label'])
                
                if eating_objects_near:
                    if any(obj in ['cup', 'bottle', 'wine glass'] for obj in eating_objects_near):
                        actions.append("Person drinking")
                    elif any(obj in ['fork', 'knife', 'spoon', 'bowl'] for obj in eating_objects_near):
                        actions.append("Person eating")
                    elif any(obj in ['banana', 'apple', 'sandwich', 'pizza', 'hot dog', 'donut', 'cake'] for obj in eating_objects_near):
                        actions.append("Person eating")
        
        # 3. SAFETY ALERTS (HIGH PRIORITY - ONLY CRITICAL)
        hazards = self.detect_safety_hazards(objects, yolo_results, w, h)
        # Only include critical safety alerts, remove "obstacle ahead" for people
        critical_hazards = [h for h in hazards if 'ALERT' in h or 'Stairs' in h or 'Fire' in h]
        if critical_hazards:
            parts.extend(critical_hazards)
        
        # 4. PEOPLE COUNT AND ACTIVITIES WITH FACE RECOGNITION
        if people_count > 0:
            # Try to recognize faces - ALWAYS attempt recognition for registered faces
            recognized_names = []
            try:
                # Try to import face recognition
                from face_recognition_module import face_recognition
                face_rec_available = True
                # Check if there are any registered faces
                has_registered_faces = len(face_recognition.face_encodings) > 0
            except Exception as import_error:
                face_rec_available = False
                has_registered_faces = False
                logger.debug(f"Face recognition module not available: {import_error}")
            
            # CRITICAL: Always try face recognition if we have registered faces and people detected
            # This ensures registered people are identified by name instead of just "person"
            if face_rec_available and has_registered_faces and people_count <= 5:
                try:
                    # Reload faces to ensure we have latest registrations
                    face_recognition.load_faces()
                    registered_count = len(face_recognition.face_encodings)
                    registered_list = list(face_recognition.face_encodings.keys())
                    logger.info(f"🔍 Attempting face recognition: {registered_count} registered faces available: {registered_list}")
                    print(f"🔍 Face recognition: {registered_count} registered faces: {registered_list}")
                    
                    # Try multiple face detection methods for better recognition
                    faces = self.analyze_faces(image)
                    logger.info(f"👤 Face detection (analyze_faces) found {len(faces)} faces in image")
                    print(f"👤 Face detection (analyze_faces) found {len(faces)} faces in image")
                    
                    if len(faces) == 0:
                        logger.warning("⚠️ No faces detected by analyze_faces() - trying alternative detection")
                        # Try using YOLO person detection to extract face regions
                        person_boxes = [box for r in yolo_results for box in r.boxes if r.names[int(box.cls[0])] == 'person']
                        for person_box in person_boxes[:5]:  # Increased to 5 people
                            x1, y1, x2, y2 = person_box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Try multiple face regions for better detection
                            # Region 1: Top 35% of person box (head area)
                            face_y1 = max(0, y1 - 10)  # Add padding above
                            face_y2 = y1 + int((y2 - y1) * 0.35)
                            face_x1 = max(0, x1 - 10)
                            face_x2 = min(w, x2 + 10)
                            face_region = image[face_y1:face_y2, face_x1:face_x2]
                            
                            if face_region.size > 0 and face_region.shape[0] > 40 and face_region.shape[1] > 40:
                                logger.info(f"🔍 Trying to match face from person box region 1 (size: {face_region.shape})")
                                # Use very lenient threshold to ensure recognition
                                name, confidence = face_recognition.find_matching_face(face_region, threshold=0.40)
                                logger.info(f"📊 Match result region 1: name={name}, confidence={confidence:.3f}")
                                if name and confidence > 0.35:  # Very lenient threshold - prioritize recognition
                                    if not any(n == name for n, _ in recognized_names):
                                        recognized_names.append((name, confidence))
                                        logger.info(f"✅ Recognized: {name} with confidence {confidence:.3f}")
                                        print(f"✅ Recognized: {name} with confidence {confidence:.3f}")
                                        continue  # Skip other regions if found
                                elif name:  # Even if below threshold, log it for debugging
                                    logger.debug(f"⚠️ Close match but below threshold: {name} (confidence: {confidence:.3f})")
                                    print(f"⚠️ Close match: {name} (confidence: {confidence:.3f}, threshold: 0.35)")
                            
                            # Region 2: Top 30% (slightly different area)
                            face_y1 = max(0, y1 - 5)
                            face_y2 = y1 + int((y2 - y1) * 0.30)
                            face_x1 = max(0, int((x1 + x2) / 2) - int((x2 - x1) * 0.6))  # Center-focused
                            face_x2 = min(w, int((x1 + x2) / 2) + int((x2 - x1) * 0.6))
                            face_region = image[face_y1:face_y2, face_x1:face_x2]
                            
                            if face_region.size > 0 and face_region.shape[0] > 40 and face_region.shape[1] > 40:
                                logger.info(f"🔍 Trying to match face from person box region 2 (size: {face_region.shape})")
                                # Use very lenient threshold to ensure recognition
                                name, confidence = face_recognition.find_matching_face(face_region, threshold=0.40)
                                logger.info(f"📊 Match result region 2: name={name}, confidence={confidence:.3f}")
                                if name and confidence > 0.35:  # Very lenient threshold
                                    if not any(n == name for n, _ in recognized_names):
                                        recognized_names.append((name, confidence))
                                        logger.info(f"✅ Recognized: {name} with confidence {confidence:.3f}")
                                        print(f"✅ Recognized: {name} with confidence {confidence:.3f}")
                                elif name:  # Log close matches for debugging
                                    logger.debug(f"⚠️ Close match but below threshold: {name} (confidence: {confidence:.3f})")
                                    print(f"⚠️ Close match: {name} (confidence: {confidence:.3f}, threshold: 0.35)")
                    
                    # Also try with detected faces from analyze_faces
                    for idx, face_info in enumerate(faces):
                        if 'bbox' in face_info:
                            x, y, w_box, h_box = face_info['bbox']
                            # Extract face region with MORE padding for better recognition
                            padding = 30  # Increased padding for better face extraction
                            face_region = image[max(0,y-padding):min(h,y+h_box+padding), max(0,x-padding):min(w,x+w_box+padding)]
                            
                            if face_region.size > 0 and face_region.shape[0] > 40 and face_region.shape[1] > 40:
                                logger.info(f"🔍 Trying to match face {idx+1} from analyze_faces (size: {face_region.shape})")
                                # Use very lenient threshold - 0.40 for histogram-based encodings
                                name, confidence = face_recognition.find_matching_face(face_region, threshold=0.40)
                                logger.info(f"📊 Match result for face {idx+1}: name={name}, confidence={confidence:.3f}")
                                if name and confidence > 0.35:  # Very lenient confidence threshold - prioritize recognition
                                    # Avoid duplicates
                                    if not any(n == name for n, _ in recognized_names):
                                        recognized_names.append((name, confidence))
                                        logger.info(f"✅ Recognized: {name} with confidence {confidence:.3f}")
                                        print(f"✅ Recognized: {name} with confidence {confidence:.3f}")
                                elif name:  # Log close matches for debugging
                                    logger.debug(f"⚠️ Face {idx+1} close match but below threshold: {name} (confidence: {confidence:.3f})")
                                    print(f"⚠️ Close match for face {idx+1}: {name} (confidence: {confidence:.3f}, threshold: 0.35)")
                                else:
                                    logger.debug(f"Face {idx+1} detected but no match (confidence: {confidence:.3f if confidence else 0})")
                    
                    logger.info(f"📋 Final recognition results: {recognized_names}")
                    print(f"📋 Final recognition: {len(recognized_names)}/{people_count} people recognized: {[n for n, _ in recognized_names]}")
                    
                    # CRITICAL: If we have registered faces but no matches, log warning
                    if len(recognized_names) == 0 and has_registered_faces:
                        logger.warning(f"⚠️ No faces matched despite {registered_count} registered faces available")
                        print(f"⚠️ WARNING: No faces matched! Registered faces: {registered_list}")
                        print(f"⚠️ This person may not be recognized. Check:")
                        print(f"   - Face is clearly visible and well-lit")
                        print(f"   - Face is facing camera")
                        print(f"   - Similar lighting/angle as registration")
                except Exception as face_error:
                    logger.error(f"❌ Face recognition error: {face_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    print(f"❌ Face recognition error: {face_error}")
                    traceback.print_exc()
            
            # SIMPLIFIED: Build simple people description
            # Get person positions (needed for actions, but not for description)
            person_positions = [obj_pos for obj_pos in object_positions if obj_pos['label'] == 'person']
            
            if recognized_names and len(recognized_names) == people_count:
                # All people recognized - simple list
                names_list = [name for name, _ in recognized_names]
                if people_count == 1:
                    parts.append(f"I see {names_list[0]}")
                elif people_count == 2:
                    parts.append(f"I see {names_list[0]} and {names_list[1]}")
                else:
                    names_str = ', '.join(names_list[:-1]) + f", and {names_list[-1]}"
                    parts.append(f"I see {names_str}")
            elif recognized_names:
                # Some people recognized - use names for recognized ones
                recognized = [name for name, _ in recognized_names]
                if people_count == 1:
                    # If we have 1 person and 1 recognition, use the name
                    parts.append(f"I see {recognized[0]}")
                else:
                    # Multiple people - use names for recognized ones
                    if len(recognized) == 1:
                        parts.append(f"I see {recognized[0]} and at least one more person")
                    else:
                        names_str = ', '.join(recognized[:-1]) + f", and {recognized[-1]}"
                        parts.append(f"I see {names_str}")
            else:
                # No recognition - but check if we should have recognized
                if face_rec_available and has_registered_faces:
                    # We have registered faces but didn't match - log this
                    try:
                        logger.warning(f"⚠️ Person detected but not recognized. Registered faces: {list(face_recognition.face_encodings.keys())}")
                        print(f"⚠️ Person detected but not recognized. Try:")
                        print(f"   - Better lighting")
                        print(f"   - Face directly facing camera")
                        print(f"   - Closer to camera")
                    except:
                        pass
                
                # No recognition - simple person count
                if people_count == 1:
                    parts.append("I see one person")
                elif people_count == 2:
                    parts.append("I see two people")
                else:
                    parts.append(f"I see {people_count} people")
            
            # SIMPLIFIED: Add only most important actions (max 2)
            logger.info(f"DEBUG: Total actions detected: {len(actions)}, actions={actions}")
            print(f"DEBUG: Total actions detected: {len(actions)}, actions={actions}")
            
            if actions:
                # Remove duplicates and keep only top 2 most important actions
                unique_actions = list(dict.fromkeys(actions))
                
                # Prioritize simple, clear actions
                simple_actions = [a for a in unique_actions if any(
                    word in a.lower() for word in ['sitting', 'standing', 'reading', 'drinking', 'using phone']
                )]
                other_actions = [a for a in unique_actions if a not in simple_actions]
                
                # Take max 2 actions
                actions_to_add = (simple_actions + other_actions)[:2]
                
                # Replace "Person" with name if recognized
                if recognized_names and len(recognized_names) == 1:
                    name = recognized_names[0][0]
                    actions_to_add = [action.replace("Person", name).replace("person", name.lower()) for action in actions_to_add]
                
                # Add actions as separate parts
                parts.extend(actions_to_add)
                print(f"DEBUG: Added {len(actions_to_add)} simple actions: {actions_to_add}")
            elif recognized_names:
                # No actions but person recognized - just say name
                if people_count == 1:
                    parts.append(f"I see {recognized_names[0][0]}")
                elif people_count == 2 and len(recognized_names) == 2:
                    parts.append(f"I see {recognized_names[0][0]} and {recognized_names[1][0]}")
        
        # 5. SIMPLE OBJECT LIST - Just list detected objects
        # Show objects that are NOT people (people are already mentioned above)
        non_person_objects = [obj for obj in objects if obj != 'person']
        if non_person_objects:
            # Get unique objects with counts (Counter already imported at top)
            obj_counts = Counter(non_person_objects)
            
            # Build simple list: "chair, table, cup"
            obj_list = []
            for obj, count in obj_counts.items():
                if count > 1:
                    obj_list.append(f"{count} {self.pluralize(obj, count)}")
                else:
                    obj_list.append(obj)
            
            # Limit to top 8 objects for simplicity
            obj_list = obj_list[:8]
            
            # Add simple object list
            if len(obj_list) == 1:
                parts.append(f"I also see {obj_list[0]}")
            elif len(obj_list) == 2:
                parts.append(f"I also see {obj_list[0]} and {obj_list[1]}")
            else:
                obj_text = ", ".join(obj_list[:-1]) + f", and {obj_list[-1]}"
                parts.append(f"I also see {obj_text}")
        
        # REMOVED: Mood detection and late action detection - simplified for simple scene detection
        # Actions are already detected and processed above
        
        # NOTE: We intentionally avoid adding generic mood sentences like
        # "Overall, the scene feels positive and friendly" because they can
        # repeat across very different scenes and make the system sound
        # inaccurate. Keep descriptions focused on concrete objects/actions.
        
        # Build SIMPLE final description - Just join parts with periods
        print(f"DEBUG: Final parts before joining: {parts}")
        print(f"DEBUG: Actions list at end: {actions}")
        print(f"DEBUG: People count: {people_count}")
        
        if parts:
            # Simple joining: "Part1. Part2. Part3."
            result = ". ".join(parts) + "."
            print(f"DEBUG: Final result: {result}")
            return result
        elif people_count > 0:
            # If we have people but no description
            return "I see a person."
        else:
            return "I can see objects in the scene."
    
    def analyze_faces(self, image):
        """Detect faces using MediaPipe or OpenCV fallback"""
        face_info = []
        
        # Try MediaPipe first
        if not self.loaded:
            if not self.load_models():
                # MediaPipe not available, use OpenCV fallback
                return self._detect_faces_opencv(image)
        
        # Use MediaPipe if available
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(image_rgb)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    face_region = image[max(0,y):min(h,y+height), max(0,x):min(w,x+width)]
                    expression = self._guess_expression(face_region)
                    
                    face_info.append({
                        'bbox': (x, y, width, height),
                        'expression': expression
                    })
        except:
            # Fallback to OpenCV if MediaPipe fails
            return self._detect_faces_opencv(image)
        
        # If no faces found with MediaPipe, try OpenCV
        if not face_info:
            face_info = self._detect_faces_opencv(image)
        
        return face_info
    
    def _detect_faces_opencv(self, image):
        """Detect faces using OpenCV Haar Cascade (fallback)"""
        face_info = []
        try:
            # Load OpenCV face detector
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if face_cascade.empty():
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                face_region = image[y:y+h, x:x+w]
                expression = self._guess_expression(face_region)
                
                face_info.append({
                    'bbox': (x, y, w, h),
                    'expression': expression
                })
        except Exception as e:
            print(f"OpenCV face detection error: {e}")
        
        return face_info
    
    def _guess_expression(self, face_region):
        """Enhanced expression estimation - detects tense, serious, smiling"""
        if face_region is None or face_region.size == 0:
            return "neutral"
        
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Detect edges for expression analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        # Analyze mouth region (lower third) for tension
        h, w = gray.shape
        mouth_region = gray[int(h*0.6):, :]
        mouth_edges = cv2.Canny(mouth_region, 50, 150)
        mouth_edge_density = np.sum(mouth_edges) / mouth_edges.size if mouth_edges.size > 0 else 0
        
        # Analyze eyebrow region (upper third) for tension
        eyebrow_region = gray[:int(h*0.4), :]
        eyebrow_edges = cv2.Canny(eyebrow_region, 50, 150)
        eyebrow_edge_density = np.sum(eyebrow_edges) / eyebrow_edges.size if eyebrow_edges.size > 0 else 0
        
        # Tense/serious: high eyebrow edge density (furrowed brow) + low mouth edge density (closed mouth)
        if eyebrow_edge_density > 0.12 and mouth_edge_density < 0.08:
            return "tense or serious"
        # Smiling: high mouth edge density (open mouth/teeth visible)
        elif mouth_edge_density > 0.15:
            return "expressive or smiling"
        # Neutral/calm
        elif edge_density > 0.15:
            return "expressive"
        else:
            return "calm or neutral"
    
    def analyze_pose(self, image):
        """Detect body pose - optimized for speed"""
        if not self.loaded:
            if not self.load_models():
                return []
        
        try:
            # Resize image for faster processing (if too large)
            h, w = image.shape[:2]
            if w > 640 or h > 480:
                scale = min(640/w, 480/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(image_rgb)
            
            pose_info = []
            if results.pose_landmarks:
                pose_info.append({
                    'landmarks': results.pose_landmarks.landmark
                })
            
            return pose_info
        except Exception as e:
            return []  # Return empty on error to avoid lag

# Global instance
advanced_analyzer = AdvancedSceneAnalyzer()
