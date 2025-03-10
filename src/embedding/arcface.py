"""
Face embedding model using ArcFace.
"""

import logging
import cv2
import numpy as np
import os

from ..core.base import BaseFaceEmbedding
from ..core.data import EmbeddingResult
from ..core.utils import validate_face_image

logger = logging.getLogger(__name__)


class ArcFaceEmbedding(BaseFaceEmbedding):
    """Face embedding model using ArcFace (InsightFace framework)"""
    
    def __init__(self):
        """Initialize the ArcFace embedding model"""
        super().__init__("ArcFace", embedding_size=512)
        self._model = None
        self._model_path = os.path.expanduser('~/.insightface/models/buffalo_l/w600k_r50.onnx')
    
    def _load_model(self):
        """Load the ArcFace model if not already loaded"""
        if self._model is None:
            try:
                from insightface.model_zoo import get_model
                
                # Try to load model with GPU first
                try:
                    self._model = get_model(self._model_path)
                    self._model.prepare(ctx_id=0)  # GPU
                    logger.debug("Loaded ArcFace embedding model on GPU")
                except Exception as e:
                    logger.warning(f"Failed to load ArcFace model on GPU: {e}")
                    # Fallback to CPU
                    self._model = get_model(self._model_path)
                    self._model.prepare(ctx_id=-1)  # CPU
                    logger.debug("Loaded ArcFace embedding model on CPU")
                    
            except ImportError:
                logger.error("Failed to import insightface. Please install with: pip install insightface onnxruntime")
                raise
            except Exception as e:
                logger.error(f"Failed to load ArcFace model: {e}")
                raise
    
    def embed(self, face_image):
        """
        Extract embedding from a face image using ArcFace.
        
        Args:
            face_image: Face image as numpy array (RGB format expected)
            
        Returns:
            EmbeddingResult with 512-d embedding, or None if embedding fails
        """
        try:
            # Validate input face image
            if not validate_face_image(face_image):
                logger.error("[ArcFace] Invalid face image provided")
                return None
            
            # Load the model if not already loaded
            self._load_model()
            
            # Resize to the required input size (112x112 for ArcFace)
            face_resized = cv2.resize(face_image, (112, 112))
            
            # Convert to the format expected by the model
            # ArcFace expects NCHW format (batch, channels, height, width)
            face_array = np.transpose(face_resized, (2, 0, 1))  # (3, 112, 112)
            face_array = np.expand_dims(face_array, axis=0)  # (1, 3, 112, 112)
            face_array = face_array.astype(np.float32)
            
            # Normalize the image (ArcFace preprocessing)
            face_array = (face_array - 127.5) / 127.5
            
            # Generate embedding
            embedding = self._model.forward(face_array)
            
            # Return embedding result
            return EmbeddingResult(
                embedding=embedding.flatten(),
                model_name=self.name,
                embedding_size=self.embedding_size
            )
            
        except ImportError:
            logger.error("Failed to import required libraries for ArcFace. Please install insightface and onnxruntime.")
            return None
        except Exception as e:
            logger.error(f"[ArcFace] Embedding error: {e}")
            return None
            
    def cleanup(self):
        """Clean up resources"""
        # Force cleanup for ONNX models which might hold GPU memory
        if self._model is not None:
            try:
                # Some ONNX models have a session that should be closed
                if hasattr(self._model, 'session') and hasattr(self._model.session, 'close'):
                    self._model.session.close()
            except:
                pass
            
        self._model = None
        super().cleanup()
