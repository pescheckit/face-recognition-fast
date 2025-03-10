"""
OpenCV-based face detector using Haar Cascades.
"""

import logging
import cv2

from ..core.base import BaseFaceDetector
from ..core.data import DetectionResult, COLOR_FORMAT
from ..core.utils import load_image, ensure_face_size

logger = logging.getLogger(__name__)


class OpenCVDetector(BaseFaceDetector):
    """Face detector using OpenCV Haar Cascades"""
    
    def __init__(self):
        """Initialize the OpenCV face detector"""
        super().__init__("OpenCV")
        self._cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._model = None
    
    def _load_model(self):
        """Load the Haar Cascade classifier if not already loaded"""
        if self._model is None:
            self._model = cv2.CascadeClassifier(self._cascade_path)
            logger.debug("Loaded OpenCV Haar Cascade face detector")
    
    def detect(self, image_path):
        """
        Detect a face in an image using Haar Cascades.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DetectionResult with face in BGR format, or None if no face is detected
        """
        try:
            # Load the image
            image = load_image(image_path)
            if image is None:
                logger.error(f"[OpenCV] Could not load image: {image_path}")
                return None
            
            # Load the face detector model
            self._load_model()
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self._model.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Extract the first (largest) face
                x, y, w, h = faces[0]
                face = image[y:y+h, x:x+w]
                
                # Ensure minimum size for embedding models
                face = ensure_face_size(face)
                
                # Return detection result
                return DetectionResult(
                    face_image=face,
                    color_format=COLOR_FORMAT.BGR,
                    bounding_box=(x, y, w, h)
                )
            else:
                logger.warning(f"[OpenCV] No faces detected in image: {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"[OpenCV] Detection error: {e}")
            return None
