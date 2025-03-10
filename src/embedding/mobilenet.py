"""
Face embedding model using MobileNetV2.
"""

import logging
import cv2
import numpy as np

from ..core.base import BaseFaceEmbedding
from ..core.data import EmbeddingResult
from ..core.utils import validate_face_image

logger = logging.getLogger(__name__)


class MobileNetEmbedding(BaseFaceEmbedding):
    """Face embedding model using MobileNetV2 (pretrained on ImageNet)"""
    
    def __init__(self):
        """Initialize the MobileNet embedding model"""
        super().__init__("MobileNet", embedding_size=1280)
        self._model = None
    
    def _load_model(self):
        """Load the MobileNetV2 model if not already loaded"""
        if self._model is None:
            try:
                import tensorflow as tf
                from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
                from tensorflow.keras.layers import GlobalAveragePooling2D
                from tensorflow.keras.models import Model
                
                # Configure GPU memory growth to avoid OOM errors
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        logger.error(f"GPU memory configuration error: {e}")
                
                # Create the model with ImageNet weights
                base_model = MobileNetV2(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
                
                # Add global average pooling to get a feature vector
                self._model = Model(
                    inputs=base_model.input,
                    outputs=GlobalAveragePooling2D()(base_model.output)
                )
                
                logger.debug("Loaded MobileNetV2 embedding model")
                
            except ImportError:
                logger.error("Failed to import TensorFlow. Please install with: pip install tensorflow")
                raise
    
    def _preprocess_image(self, image):
        """
        Preprocess image for MobileNetV2.
        
        Args:
            image: Input face image
            
        Returns:
            Preprocessed image ready for model input
        """
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            
            # Resize to the required input size
            image_resized = cv2.resize(image, (224, 224))
            
            # Add batch dimension
            image_batch = np.expand_dims(image_resized, axis=0)
            
            # Apply MobileNet-specific preprocessing
            return preprocess_input(image_batch)
            
        except ImportError:
            logger.error("Failed to import TensorFlow preprocessing")
            raise
    
    def embed(self, face_image):
        """
        Extract embedding from a face image using MobileNetV2.
        
        Args:
            face_image: Face image as numpy array (RGB format expected)
            
        Returns:
            EmbeddingResult with 1280-d embedding, or None if embedding fails
        """
        try:
            # Validate input face image
            if not validate_face_image(face_image):
                logger.error("[MobileNet] Invalid face image provided")
                return None
            
            # Load the model if not already loaded
            self._load_model()
            
            # Preprocess the image
            preprocessed_face = self._preprocess_image(face_image)
            
            # Generate embedding
            embedding = self._model.predict(preprocessed_face, verbose=0)
            
            # Return embedding result
            return EmbeddingResult(
                embedding=embedding.flatten(),
                model_name=self.name,
                embedding_size=self.embedding_size
            )
            
        except ImportError:
            logger.error("Failed to import required libraries for MobileNet. Please install TensorFlow.")
            return None
        except Exception as e:
            logger.error(f"[MobileNet] Embedding error: {e}")
            return None
