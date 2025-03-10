"""
Face embedding model using DeepFace.
"""

import logging
import cv2
import numpy as np
import os
import tempfile

from ..core.base import BaseFaceEmbedding
from ..core.data import EmbeddingResult
from ..core.utils import validate_face_image, safe_temp_file

logger = logging.getLogger(__name__)


class DeepFaceEmbedding(BaseFaceEmbedding):
    """Face embedding model using DeepFace"""
    
    def __init__(self, model_name="ArcFace"):
        """
        Initialize the DeepFace embedding model.
        
        Args:
            model_name: Name of the model to use. Options:
                - "ArcFace" (default, 512-d)
                - "Facenet" (128-d)
                - "Facenet512" (512-d)
                - "VGG-Face" (2048-d)
                - "OpenFace" (128-d)
                - "DeepFace" (4096-d)
                - "DeepID" (160-d)
                - "SFace" (512-d)
        """
        self.model_name = model_name
        
        # Set embedding size based on model
        if model_name == "ArcFace":
            embedding_size = 512
        elif model_name == "Facenet":
            embedding_size = 128
        elif model_name == "Facenet512":
            embedding_size = 512
        elif model_name == "VGG-Face":
            embedding_size = 2048
        elif model_name == "OpenFace":
            embedding_size = 128
        elif model_name == "DeepFace":
            embedding_size = 4096
        elif model_name == "DeepID":
            embedding_size = 160
        elif model_name == "SFace":
            embedding_size = 512
        else:
            logger.warning(f"Unknown model '{model_name}', defaulting to ArcFace")
            model_name = "ArcFace"
            embedding_size = 512
        
        super().__init__(f"DeepFace-{model_name}", embedding_size)
    
    def embed(self, face_image):
        """
        Extract embedding from a face image using DeepFace.
        
        Args:
            face_image: Face image as numpy array (RGB format expected)
            
        Returns:
            EmbeddingResult with model-specific embedding, or None if embedding fails
        """
        try:
            from deepface import DeepFace
            
            # Validate input face image
            if not validate_face_image(face_image):
                logger.error("[DeepFace] Invalid face image provided")
                return None
            
            # DeepFace API often requires saving to a temporary file
            temp_path, success = safe_temp_file(face_image)
            if not success:
                logger.error("[DeepFace] Failed to create temporary file")
                return None
            
            try:
                # Try to get embedding using represent
                embedding_objs = DeepFace.represent(
                    img_path=temp_path,
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend="skip"  # Skip detection since we already have a cropped face
                )
                
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                if not embedding_objs or len(embedding_objs) == 0:
                    logger.warning(f"[DeepFace] No embedding returned for {self.model_name}")
                    return None
                
                # Extract embedding from result
                vec = embedding_objs[0]['embedding']
                embedding = np.array(vec, dtype=np.float32)
                
                # Return embedding result
                return EmbeddingResult(
                    embedding=embedding,
                    model_name=self.name,
                    embedding_size=self.embedding_size
                )
                
            except Exception as e:
                # Clean up temp file if an exception occurred
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                logger.error(f"[DeepFace] represent() failed: {e}")
                
                # Try alternative approach if the first fails
                try:
                    logger.info("[DeepFace] Trying alternative approach with verify()")
                    
                    # Create a duplicate temp file for the second image
                    temp_path2, success = safe_temp_file(face_image)
                    if not success:
                        return None
                    
                    # Sometimes the newer DeepFace versions use different parameter names
                    # Let's try the verify function which returns embeddings internally
                    result = DeepFace.verify(
                        img1_path=temp_path,
                        img2_path=temp_path2,
                        model_name=self.model_name,
                        enforce_detection=False,
                        detector_backend="skip"
                    )
                    
                    # Clean up temporary files
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    if os.path.exists(temp_path2):
                        os.unlink(temp_path2)
                    
                    # Extract embedding from verification result (if available)
                    if hasattr(result, 'embedding1') and result.embedding1 is not None:
                        embedding = np.array(result.embedding1, dtype=np.float32)
                        return EmbeddingResult(
                            embedding=embedding,
                            model_name=self.name,
                            embedding_size=self.embedding_size
                        )
                    else:
                        logger.error("[DeepFace] Couldn't extract embedding from verify result")
                        return None
                        
                except Exception as e2:
                    # Clean up any remaining temporary files
                    for p in [temp_path, temp_path2]:
                        if p and os.path.exists(p):
                            try:
                                os.unlink(p)
                            except:
                                pass
                    
                    logger.error(f"[DeepFace] Alternative approach failed: {e2}")
                    return None
                
        except ImportError:
            logger.error("Failed to import deepface. Please install with: pip install deepface")
            return None
        except Exception as e:
            logger.error(f"[DeepFace] Embedding error: {e}")
            return None
