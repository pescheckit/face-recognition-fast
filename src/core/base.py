"""
Base classes for the face recognition system.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np

from .data import DetectionResult, EmbeddingResult
from .utils import cleanup

logger = logging.getLogger(__name__)


class BaseFaceDetector:
    """Base class for face detector implementations"""
    
    def __init__(self, name: str):
        """
        Initialize the face detector.
        
        Args:
            name: The name of the detector
        """
        self.name = name
        self._model = None
    
    def detect(self, image_path: str) -> Optional[DetectionResult]:
        """
        Detect a face in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DetectionResult object or None if no face is detected
        """
        raise NotImplementedError("Subclasses must implement detect()")
    
    def _load_model(self):
        """Load model if not already loaded"""
        pass  # Override in subclasses if needed
    
    def cleanup(self):
        """Clean up resources"""
        self._model = None
        cleanup()


class BaseFaceEmbedding:
    """Base class for face embedding implementations"""
    
    def __init__(self, name: str, embedding_size: int):
        """
        Initialize the face embedding model.
        
        Args:
            name: The name of the embedding model
            embedding_size: The size of the embedding vector
        """
        self.name = name
        self.embedding_size = embedding_size
        self._model = None
    
    def embed(self, face_image: np.ndarray) -> Optional[EmbeddingResult]:
        """
        Extract embedding from a face image.
        
        Args:
            face_image: The face image as a numpy array
            
        Returns:
            EmbeddingResult object or None if embedding fails
        """
        raise NotImplementedError("Subclasses must implement embed()")
    
    def _load_model(self):
        """Load model if not already loaded"""
        pass  # Override in subclasses if needed
    
    def cleanup(self):
        """Clean up resources"""
        self._model = None
        cleanup()


class FaceRecognitionPipeline:
    """Pipeline to handle detection and embedding with proper error handling and fallbacks"""
    
    def __init__(self, 
                 detector: BaseFaceDetector,
                 embedding_model: BaseFaceEmbedding,
                 fallback_detectors: List[BaseFaceDetector] = None):
        """
        Initialize the face recognition pipeline.
        
        Args:
            detector: The primary face detector
            embedding_model: The face embedding model
            fallback_detectors: List of fallback detectors to try if primary fails
        """
        self.detector = detector
        self.embedding_model = embedding_model
        self.fallback_detectors = fallback_detectors or []
    
    def process_image(self, image_path: str) -> Tuple[Optional[DetectionResult], Optional[EmbeddingResult]]:
        """
        Process an image through the pipeline: detect face, then extract embedding.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (DetectionResult, EmbeddingResult) or (None, None) if processing fails
        """
        
        # Try primary detector
        detection_result = self.detector.detect(image_path)
        detector_used = self.detector.name
        
        # Try fallbacks if primary fails
        if detection_result is None and self.fallback_detectors:
            for fallback in self.fallback_detectors:
                logger.info(f"[{self.detector.name}] Falling back to {fallback.name}")
                detection_result = fallback.detect(image_path)
                if detection_result is not None:
                    detector_used = fallback.name
                    break
        
        # Return None if all detectors failed
        if detection_result is None:
            logger.warning(f"All detectors failed for image: {image_path}")
            return None, None
        
        logger.debug(f"Face detected using {detector_used}")
        
        # Ensure face is in RGB format for embedding
        rgb_face = detection_result.to_rgb()
        
        # Extract embedding
        embedding_result = self.embedding_model.embed(rgb_face.face_image)
        
        if embedding_result is None:
            logger.warning(f"Embedding failed for detected face in image: {image_path}")
            return detection_result, None
            
        return detection_result, embedding_result
    
    def cleanup(self):
        """Clean up resources"""
        self.detector.cleanup()
        self.embedding_model.cleanup()
        for detector in self.fallback_detectors:
            detector.cleanup()
