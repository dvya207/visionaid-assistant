"""
Face Recognition Module
Stores and matches faces for person identification
"""
import cv2
import numpy as np
import os
import json
import pickle
from datetime import datetime

class FaceRecognition:
    def __init__(self, storage_dir='face_storage'):
        self.storage_dir = storage_dir
        self.face_encodings = {}  # {name: encoding}
        self.face_db_file = os.path.join(storage_dir, 'face_database.json')
        self.encodings_file = os.path.join(storage_dir, 'face_encodings.pkl')
        self.deepface_loaded = False
        self.DeepFace = None
        
        # Create storage directory
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        
        # Load existing faces
        self.load_faces()
        self.load_deepface()
    
    def load_deepface(self):
        """Load DeepFace for face recognition, with MediaPipe and OpenCV fallbacks"""
        if self.deepface_loaded:
            return True
        
        # Try DeepFace first
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.deepface_loaded = True
            self.use_mediapipe = False
            self.use_opencv = False
            return True
        except ImportError:
            # DeepFace not available, trying MediaPipe (silent - optional)
            # Fallback to MediaPipe
            try:
                # Import compatibility layer
                try:
                    from mediapipe_compat import is_available, is_new_api, get_mediapipe_version
                except ImportError:
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
                    raise ImportError("MediaPipe not available")
                
                import mediapipe as mp
                use_new_api = is_new_api()
                mp_version = get_mediapipe_version()
                
                if use_new_api:
                    # Silent - MediaPipe new API not supported, will use OpenCV fallback
                    raise ImportError("MediaPipe new API not supported")
                
                # Old API (0.9.x)
                try:
                    self.mp_face_detection = mp.solutions.face_detection
                    self.face_detector_mp = self.mp_face_detection.FaceDetection(
                        model_selection=1, min_detection_confidence=0.5
                        )
                    self.deepface_loaded = True
                    self.use_mediapipe = True
                    self.use_opencv = False
                    print(f"MediaPipe face recognition loaded successfully (version {mp_version})")
                    return True
                except AttributeError:
                    # Silent - MediaPipe old API not available, will use OpenCV fallback
                    raise ImportError("MediaPipe old API not available")
            except ImportError:
                # Silent - MediaPipe not available, using OpenCV Haar Cascade as fallback (this is fine)
                # Final fallback to OpenCV Haar Cascade (always available)
                try:
                    import cv2
                    # Load OpenCV face detector
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    if os.path.exists(cascade_path):
                        self.face_cascade = cv2.CascadeClassifier(cascade_path)
                        self.deepface_loaded = True
                        self.use_mediapipe = False
                        self.use_opencv = True
                        print("OpenCV face recognition loaded successfully")
                        return True
                    else:
                        print("OpenCV face cascade not found")
                        self.deepface_loaded = False
                        return False
                except Exception as e:
                    print(f"OpenCV face detection error: {e}")
                    self.deepface_loaded = False
                    return False
            except Exception as e:
                print(f"MediaPipe face detection error: {e}")
                # Try OpenCV fallback
                try:
                    import cv2
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    if os.path.exists(cascade_path):
                        self.face_cascade = cv2.CascadeClassifier(cascade_path)
                        self.deepface_loaded = True
                        self.use_mediapipe = False
                        self.use_opencv = True
                        print("OpenCV face recognition loaded as fallback")
                        return True
                except:
                    pass
                self.deepface_loaded = False
                return False
        except Exception as e:
            print(f"DeepFace error: {e}")
            return False
    
    def extract_face_encoding(self, face_image):
        """Extract face encoding/embedding from image"""
        if not self.deepface_loaded:
            return None
        
        try:
            if hasattr(self, 'use_opencv') and self.use_opencv:
                # Use OpenCV for face encoding (always available)
                return self._extract_face_encoding_opencv(face_image)
            elif hasattr(self, 'use_mediapipe') and self.use_mediapipe:
                # Use MediaPipe for face encoding (simpler, no DeepFace needed)
                return self._extract_face_encoding_mediapipe(face_image)
            else:
                # Use DeepFace
                embeddings = self.DeepFace.represent(
                    face_image,
                    model_name='VGG-Face',
                    enforce_detection=False
                )
                
                # Handle list returns (can have multiple faces)
                if isinstance(embeddings, list):
                    if len(embeddings) > 0:
                        embedding = embeddings[0]
                    else:
                        return None
                else:
                    embedding = embeddings
                
                # Extract the embedding vector from dict
                if isinstance(embedding, dict) and 'embedding' in embedding:
                    return np.array(embedding['embedding'])
                elif isinstance(embedding, dict):
                    for key, value in embedding.items():
                        if isinstance(value, (list, np.ndarray)):
                            return np.array(value)
                    return None
                elif isinstance(embedding, (list, np.ndarray)):
                    return np.array(embedding)
                else:
                    return None
        except Exception as e:
            print(f"Error extracting face encoding: {e}")
            return None
    
    def _extract_face_encoding_mediapipe(self, face_image):
        """Extract face encoding using MediaPipe (simpler fallback)"""
        try:
            import cv2
            # Convert to RGB for MediaPipe
            if len(face_image.shape) == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Detect face
            results = self.face_detector_mp.process(rgb_image)
            
            if results.detections and len(results.detections) > 0:
                # Get face region
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = face_image.shape[:2]
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Extract face region
                face_region = face_image[max(0,y):min(h,y+height), max(0,x):min(w,x+width)]
                
                if face_region.size > 0:
                    # Resize to standard size for encoding
                    face_resized = cv2.resize(face_region, (160, 160))
                    # Convert to grayscale and flatten as simple encoding
                    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
                    # Use histogram features as encoding
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    # Normalize
                    hist = hist / (hist.sum() + 1e-7)
                    return hist.flatten()
            
            return None
        except Exception as e:
            print(f"MediaPipe encoding error: {e}")
            return None
    
    def _extract_face_encoding_opencv(self, face_image):
        """Extract face encoding using OpenCV Haar Cascade (always available)"""
        try:
            import cv2
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get first face
                x, y, w, h = faces[0]
                face_region = gray[y:y+h, x:x+w]
                
                if face_region.size > 0:
                    # Resize to standard size
                    face_resized = cv2.resize(face_region, (160, 160))
                    # Use histogram features as encoding
                    hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
                    # Normalize
                    hist = hist / (hist.sum() + 1e-7)
                    return hist.flatten()
            
            return None
        except Exception as e:
            print(f"OpenCV encoding error: {e}")
            return None
    
    def register_face(self, name, face_image):
        """Register a face with a name - Enhanced with preprocessing"""
        if not self.deepface_loaded:
            return False, "Face recognition not available. Please ensure MediaPipe is installed."
        
        try:
            # Enhanced preprocessing for better face recognition
            try:
                from enhanced_preprocessing import EnhancedPreprocessor
                preprocessed = EnhancedPreprocessor.preprocess_for_face_recognition(face_image)
            except:
                preprocessed = face_image
            
            # Extract face encoding (works with both DeepFace and MediaPipe)
            encoding = self.extract_face_encoding(preprocessed)
            
            if encoding is None:
                return False, "Could not detect face in image. Make sure your face is clearly visible and well-lit."
            
            # Store encoding
            self.face_encodings[name] = encoding
            print(f"✅ Stored encoding for {name} in memory")
            
            # Save to disk immediately
            self.save_faces()
            
            # Verify it was saved
            if os.path.exists(self.encodings_file):
                print(f"✅ Verified: Face file exists at {self.encodings_file}")
            else:
                print(f"⚠️ Warning: Face file not found after save")
            
            return True, f"Face registered successfully for {name}"
        except Exception as e:
            return False, f"Error registering face: {str(e)}"
    
    def find_matching_face(self, face_image, threshold=0.6):
        """Find matching face from registered faces - Enhanced with preprocessing"""
        # Reload faces from disk to ensure we have latest registrations
        self.load_faces()
        
        if not self.deepface_loaded or len(self.face_encodings) == 0:
            print(f"Face recognition not available or no faces registered. Loaded: {self.deepface_loaded}, Faces: {len(self.face_encodings)}")
            return None, 0.0
        
        print(f"Searching for match among {len(self.face_encodings)} registered faces: {list(self.face_encodings.keys())}")
        
        try:
            # Enhanced preprocessing for better face recognition
            try:
                from enhanced_preprocessing import EnhancedPreprocessor
                preprocessed = EnhancedPreprocessor.preprocess_for_face_recognition(face_image)
            except:
                preprocessed = face_image
            
            # Extract encoding from input face
            input_encoding = self.extract_face_encoding(preprocessed)
            
            if input_encoding is None:
                return None, 0.0
            
            # Compare with all registered faces
            best_match = None
            best_distance = float('inf')
            best_similarity = 0.0
            
            # Adjust threshold based on encoding method
            # CRITICAL: Use lenient thresholds for better recognition of registered faces
            if (hasattr(self, 'use_mediapipe') and self.use_mediapipe) or (hasattr(self, 'use_opencv') and self.use_opencv):
                # For histogram-based encodings, use LOWER threshold (0.5 = 50% similarity)
                # This is more lenient to account for lighting/angle/expression differences
                effective_threshold = 0.5  # Lowered from 0.8 for better recognition
            else:
                # For DeepFace, use original threshold
                effective_threshold = threshold
            
            for name, stored_encoding in self.face_encodings.items():
                # For histogram-based encodings (MediaPipe/OpenCV), use multiple comparison methods
                if (hasattr(self, 'use_mediapipe') and self.use_mediapipe) or (hasattr(self, 'use_opencv') and self.use_opencv):
                    # Method 1: Correlation coefficient
                    try:
                        input_flat = input_encoding.flatten()
                        stored_flat = stored_encoding.flatten()
                        
                        # Ensure same length
                        min_len = min(len(input_flat), len(stored_flat))
                        input_flat = input_flat[:min_len]
                        stored_flat = stored_flat[:min_len]
                        
                        correlation = np.corrcoef(input_flat, stored_flat)[0, 1]
                        if np.isnan(correlation):
                            corr_similarity = 0.0
                        else:
                            corr_similarity = max(0, correlation)  # Correlation can be negative
                    except:
                        corr_similarity = 0.0
                    
                    # Method 2: Histogram intersection
                    try:
                        intersection = np.minimum(input_flat, stored_flat).sum()
                        union = np.maximum(input_flat, stored_flat).sum()
                        if union > 0:
                            hist_similarity = intersection / union
                        else:
                            hist_similarity = 0.0
                    except:
                        hist_similarity = 0.0
                    
                    # Method 3: Cosine similarity on normalized histograms
                    try:
                        input_norm = input_flat / (np.linalg.norm(input_flat) + 1e-7)
                        stored_norm = stored_flat / (np.linalg.norm(stored_flat) + 1e-7)
                        cosine_sim = np.dot(input_norm, stored_norm)
                        cosine_sim = max(0, cosine_sim)  # Ensure non-negative
                    except:
                        cosine_sim = 0.0
                    
                    # Combine similarities (weighted average)
                    similarity = (0.4 * corr_similarity + 0.3 * hist_similarity + 0.3 * cosine_sim)
                    distance = 1 - similarity
                else:
                    # For DeepFace, use cosine similarity
                    dot_product = np.dot(input_encoding, stored_encoding)
                    norm_input = np.linalg.norm(input_encoding)
                    norm_stored = np.linalg.norm(stored_encoding)
                    
                    if norm_input > 0 and norm_stored > 0:
                        similarity = dot_product / (norm_input * norm_stored)
                        distance = 1 - similarity  # Convert similarity to distance
                    else:
                        continue
                
                if distance < best_distance:
                    best_distance = distance
                    best_similarity = similarity
                    best_match = name
            
            # Enhanced matching: use adaptive threshold based on encoding method
            # CRITICAL: Use VERY LENIENT thresholds to ensure registered faces are recognized
            if (hasattr(self, 'use_mediapipe') and self.use_mediapipe) or (hasattr(self, 'use_opencv') and self.use_opencv):
                # Histogram-based: use VERY LENIENT threshold (0.35 = 35% similarity minimum)
                # This ensures registered faces are recognized even with different lighting/angles/expressions
                # Lower threshold = more lenient = better recognition of registered faces
                match_threshold = max(0.35, min(threshold, 0.55))  # Between 0.35-0.55 for histogram-based
            else:
                # DeepFace: use cosine similarity (lower is better for distance)
                match_threshold = 1 - effective_threshold
            
            # Check if match is good enough
            if best_match and best_similarity >= match_threshold:
                print(f"✅ Match found: {best_match} with similarity {best_similarity:.3f} (threshold: {match_threshold:.3f})")
                return best_match, best_similarity
            
            # If close to threshold, still return match (for better recognition)
            # Use 85% of threshold for close matches (even more lenient)
            if best_match and best_similarity >= match_threshold * 0.85:  # 85% of threshold
                print(f"⚠️ Close match: {best_match} with similarity {best_similarity:.3f} (85% of threshold: {match_threshold * 0.85:.3f})")
                return best_match, best_similarity
            
            # Return best match even if below threshold (for debugging/logging)
            if best_match:
                print(f"❌ Below threshold: {best_match} with similarity {best_similarity:.3f} (threshold: {match_threshold:.3f})")
                return None, best_similarity  # Return similarity for logging even if below threshold
            
            return None, 0.0
        except Exception as e:
            print(f"Error finding matching face: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def save_faces(self):
        """Save face encodings to disk"""
        try:
            # Ensure storage directory exists
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
            
            # Save encodings as pickle (binary)
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.face_encodings, f)
            print(f"✅ Saved {len(self.face_encodings)} face encodings to {self.encodings_file}")
            
            # Save metadata as JSON
            metadata = {
                'registered_names': list(self.face_encodings.keys()),
                'count': len(self.face_encodings),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.face_db_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Saved face metadata to {self.face_db_file}")
        except Exception as e:
            print(f"❌ Error saving faces: {e}")
            import traceback
            traceback.print_exc()
    
    def load_faces(self):
        """Load face encodings from disk"""
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    self.face_encodings = pickle.load(f)
                print(f"✅ Loaded {len(self.face_encodings)} registered faces from disk: {list(self.face_encodings.keys())}")
            else:
                self.face_encodings = {}
                print("ℹ️ No face storage file found - starting with empty database")
        except Exception as e:
            print(f"❌ Error loading faces: {e}")
            import traceback
            traceback.print_exc()
            self.face_encodings = {}
    
    def get_registered_names(self):
        """Get list of registered names"""
        return list(self.face_encodings.keys())
    
    def delete_face(self, name):
        """Delete a registered face"""
        if name in self.face_encodings:
            del self.face_encodings[name]
            self.save_faces()
            return True, f"Face deleted for {name}"
        return False, f"Face not found for {name}"
    
    def clear_all_faces(self):
        """Clear all registered faces"""
        try:
            # Clear in-memory storage FIRST
            self.face_encodings = {}
            
            # Delete files
            if os.path.exists(self.encodings_file):
                os.remove(self.encodings_file)
                print(f"Deleted: {self.encodings_file}")
            if os.path.exists(self.face_db_file):
                os.remove(self.face_db_file)
                print(f"Deleted: {self.face_db_file}")
            
            # Save empty database to ensure it stays empty
            self.save_faces()
            
            # Verify files are empty or don't exist
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    loaded = pickle.load(f)
                    if len(loaded) > 0:
                        print(f"WARNING: File still contains {len(loaded)} faces after clear!")
                        # Force clear
                        self.face_encodings = {}
                        self.save_faces()
            
            print("All registered faces cleared successfully")
            return True, "All registered faces cleared successfully"
        except Exception as e:
            print(f"Error clearing faces: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Error clearing faces: {str(e)}"

# Global instance
face_recognition = FaceRecognition()

