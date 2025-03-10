"""
Face detector using the face_recognition library.
"""

import logging
import cv2
import numpy as np

from ..core.base import BaseFaceDetector
from ..core.data import DetectionResult, COLOR_FORMAT
from ..core.utils import ensure_face_size

logger = logging.getLogger(__name__)


class FaceRecognitionDetector(BaseFaceDetector):
    """Face detector using face_recognition library (which uses dlib internally)"""
    
    def __init__(self, model_type="hog"):
        """
        Initialize the face_recognition detector.
        
        Args:
            model_type: Type of model to use, 'hog' (CPU) or 'cnn' (GPU)
        """
        super().__init__("FaceRecognition")
        self.model_type = model_type
    
    def detect(self, image_path):
        """
        Detect a face in an image using face_recognition.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DetectionResult with face in RGB format, or None if no face is detected
        """
        try:
            import face_recognition
            
            # face_recognition has its own image loading function that loads as RGB
            try:
                image = face_recognition.load_image_file(image_path)
            except Exception as e:
                logger.error(f"[FaceRecognition] Error loading image {image_path}: {e}")
                return None
            
            # Detect face locations
            face_locations = face_recognition.face_locations(
                image, 
                model=self.model_type
            )
            
            if face_locations:
                # face_locations format is (top, right, bottom, left)
                top, right, bottom, left = face_locations[0]
                
                # Extract the face region
                face = image[top:bottom, left:right]
                
                # Ensure minimum size for embedding models
                face = ensure_face_size(face)
                
                # Return detection result (already in RGB format)
                return DetectionResult(
                    face_image=face,
                    color_format=COLOR_FORMAT.RGB,
                    bounding_box=(top, right, bottom, left)
                )
            else:
                logger.warning(f"[FaceRecognition] No faces detected in image: {image_path}")
                return None
                
        except ImportError:
            logger.error("Failed to import face_recognition. Please install with: pip install face_recognition")
            return None
        except Exception as e:
            logger.error(f"[FaceRecognition] Detection error: {e}")
            return None
