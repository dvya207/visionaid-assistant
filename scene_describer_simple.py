"""
Simplified Scene Describer using only YOLO object detection
This provides basic scene understanding without requiring transformers library
"""
from collections import Counter

class SimpleSceneDescriber:
    def __init__(self):
        self.loaded = True
        self.load_error = None
    
    def describe_scene(self, yolo_results, img_width, img_height):
        """
        Generate a simple scene description from YOLO detections
        NO TECHNICAL DETAILS - ONLY OBJECTS AND ACTIONS
        """
        objects = []
        
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = result.names[cls]
                confidence = float(box.conf[0])
                
                if confidence < 0.4:
                    continue
                    
                objects.append(label)
        
        if not objects:
            return "I don't see any recognizable objects in this scene."
        
        # Count objects
        object_counts = Counter(objects)
        
        # Build description - SIMPLE, NO POSITIONS OR TECHNICAL DETAILS
        parts = []
        
        # Overall summary
        total = len(objects)
        unique = len(object_counts)
        
        if total == 1:
            parts.append(f"I see one {objects[0]}")
        elif unique == 1:
            parts.append(f"I see {total} {objects[0]}s")
        else:
            # List main objects (max 3)
            main_objects = [f"{count} {obj}" + ("s" if count > 1 else "") 
                          for obj, count in object_counts.most_common(3)]
            parts.append(f"I see {', '.join(main_objects)}")
        
        return ". ".join(parts) + "."

# Global instance
simple_scene_describer = SimpleSceneDescriber()
