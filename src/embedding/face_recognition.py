"""
Face embedding model using the face_recognition library.
"""

import logging
import numpy as np

from ..core.base import BaseFaceEmbedding
from ..core.data import EmbeddingResult
from ..core.utils import validate_face_image

logger = logging.getLogger(__name__)


class FaceRecognitionEmbedding(BaseFaceEmbedding):
    """Face embedding model using the face_recognition library (dlib-based)"""
    
    def __init__(self):
        """Initialize the face_recognition embedding model"""
        super().__init__("FaceRecognition", embedding_size=128)
    
    def embed(self, face_image):
        """
        Extract embedding from a face image using face_recognition.
        
        Args:
            face_image: Face image as numpy array (RGB format expected)
            
        Returns:
            EmbeddingResult with 128-d embedding, or None if embedding fails
        """
        try:
            import face_recognition
            
            # Validate input face image
            if not validate_face_image(face_image):
                logger.error("[FaceRecognition] Invalid face image provided")
                return None
            
            # Convert to expected format (face_recognition is sensitive about array type)
            face_image = np.ascontiguousarray(face_image, dtype=np.uint8)
            
            # Generate face encodings
            encodings = face_recognition.face_encodings(face_image)
            
            if not encodings:
                logger.warning("[FaceRecognition] Failed to generate encoding")
                return None
            
            # Return the first encoding as embedding result
            return EmbeddingResult(
                embedding=encodings[0],
                model_name=self.name,
                embedding_size=self.embedding_size
            )
            
        except ImportError:
            logger.error("Failed to import face_recognition. Please install with: pip install face_recognition")
            return None
        except Exception as e:
            logger.error(f"[FaceRecognition] Embedding error: {e}")
            return None
