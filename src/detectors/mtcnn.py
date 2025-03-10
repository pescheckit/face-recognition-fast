"""
MTCNN-based face detector.
"""

import logging
import cv2

from ..core.base import BaseFaceDetector
from ..core.data import DetectionResult, COLOR_FORMAT
from ..core.utils import load_image, ensure_face_size

logger = logging.getLogger(__name__)


class MTCNNDetector(BaseFaceDetector):
    """Face detector using MTCNN (Multi-task Cascaded Convolutional Networks)"""
    
    def __init__(self):
        """Initialize the MTCNN face detector"""
        super().__init__("MTCNN")
        self._model = None
    
    def _load_model(self):
        """Load the MTCNN detector if not already loaded"""
        if self._model is None:
            try:
                from mtcnn import MTCNN
                self._model = MTCNN()
                logger.debug("Loaded MTCNN face detector")
            except ImportError:
                logger.error("Failed to import MTCNN. Please install with: pip install mtcnn")
                raise
    
    def detect(self, image_path):
        """
        Detect a face in an image using MTCNN.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DetectionResult with face in BGR format, or None if no face is detected
        """
        try:
            # Load the image
            image = load_image(image_path)
            if image is None:
                logger.error(f"[MTCNN] Could not load image: {image_path}")
                return None
            
            # Load the MTCNN detector
            self._load_model()
            
            # MTCNN expects RGB input
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self._model.detect_faces(image_rgb)
            
            if faces:
                # Get the first (highest confidence) face
                face_data = faces[0]
                x, y, width, height = face_data['box']
                confidence = face_data.get('confidence', 1.0)
                landmarks = face_data.get('keypoints', None)
                
                # Extract the face region
                face = image[y:y+height, x:x+width]
                
                # Ensure minimum size for embedding models
                face = ensure_face_size(face)
                
                # Return detection result
                return DetectionResult(
                    face_image=face,
                    color_format=COLOR_FORMAT.BGR,
                    confidence=confidence,
                    bounding_box=(x, y, width, height),
                    landmarks=landmarks
                )
            else:
                logger.warning(f"[MTCNN] No faces detected in image: {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"[MTCNN] Detection error: {e}")
            return None
