"""
Factory classes for creating face recognition components and pipelines.
"""

import logging
from typing import Dict, List, Optional, Type, Any

from .base import BaseFaceDetector, BaseFaceEmbedding, FaceRecognitionPipeline

logger = logging.getLogger(__name__)

# These will be populated by the detector and embedding modules
AVAILABLE_DETECTORS: Dict[str, Type[BaseFaceDetector]] = {}
AVAILABLE_EMBEDDINGS: Dict[str, Type[BaseFaceEmbedding]] = {}


def register_detector(name: str, detector_class: Type[BaseFaceDetector]) -> None:
    """
    Register a detector class with the factory.
    
    Args:
        name: Name identifier for the detector
        detector_class: The detector class to register
    """
    AVAILABLE_DETECTORS[name] = detector_class
    logger.debug(f"Registered detector: {name}")


def register_embedding(name: str, embedding_class: Type[BaseFaceEmbedding]) -> None:
    """
    Register an embedding class with the factory.
    
    Args:
        name: Name identifier for the embedding model
        embedding_class: The embedding class to register
    """
    AVAILABLE_EMBEDDINGS[name] = embedding_class
    logger.debug(f"Registered embedding model: {name}")


class RecognitionMethodFactory:
    """Factory to create pipelines for different recognition methods"""
    
    # Default configuration for each method
    METHOD_CONFIGS = {
        "opencv": {
            "detector": "opencv",
            "embedding": "face_recognition",
            "fallbacks": []
        },
        "mtcnn": {
            "detector": "mtcnn",
            "embedding": "face_recognition",
            "fallbacks": ["opencv"]
        },
        "facerecognition": {
            "detector": "face_recognition",
            "embedding": "face_recognition",
            "fallbacks": ["mtcnn", "opencv"]
        },
        "mobilenet": {
            "detector": "mtcnn",
            "embedding": "mobilenet",
            "fallbacks": ["face_recognition", "opencv"]
        },
        "arcface": {
            "detector": "arcface",
            "embedding": "arcface",
            "fallbacks": ["face_recognition", "mtcnn"]
        },
        "deepface": {
            "detector": "mtcnn",
            "embedding": "deepface",
            "fallbacks": ["face_recognition", "opencv"]
        }
    }
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """
        Get a list of all available recognition methods.
        
        Returns:
            List of method names
        """
        return list(cls.METHOD_CONFIGS.keys())
    
    @classmethod
    def create_detector(cls, detector_name: str) -> Optional[BaseFaceDetector]:
        """
        Create a detector instance by name.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            BaseFaceDetector instance or None if not found
        """
        if detector_name not in AVAILABLE_DETECTORS:
            logger.warning(f"Detector '{detector_name}' not found")
            return None
        
        try:
            return AVAILABLE_DETECTORS[detector_name]()
        except Exception as e:
            logger.error(f"Error creating detector '{detector_name}': {e}")
            return None
    
    @classmethod
    def create_embedding(cls, embedding_name: str) -> Optional[BaseFaceEmbedding]:
        """
        Create an embedding model instance by name.
        
        Args:
            embedding_name: Name of the embedding model
            
        Returns:
            BaseFaceEmbedding instance or None if not found
        """
        if embedding_name not in AVAILABLE_EMBEDDINGS:
            logger.warning(f"Embedding model '{embedding_name}' not found")
            return None
        
        try:
            return AVAILABLE_EMBEDDINGS[embedding_name]()
        except Exception as e:
            logger.error(f"Error creating embedding model '{embedding_name}': {e}")
            return None
    
    @classmethod
    def create_pipeline(cls, method_name: str) -> Optional[FaceRecognitionPipeline]:
        """
        Create a complete pipeline for the specified method.
        
        Args:
            method_name: Name of the recognition method
            
        Returns:
            FaceRecognitionPipeline instance or None if creation fails
        """
        if method_name not in cls.METHOD_CONFIGS:
            logger.error(f"Unknown method: {method_name}")
            return None
        
        config = cls.METHOD_CONFIGS[method_name]
        
        # Create detector
        detector = cls.create_detector(config["detector"])
        if detector is None:
            logger.error(f"Failed to create detector for method '{method_name}'")
            return None
        
        # Create embedding model
        embedding = cls.create_embedding(config["embedding"])
        if embedding is None:
            logger.error(f"Failed to create embedding model for method '{method_name}'")
            return None
        
        # Create fallback detectors
        fallbacks = []
        for fallback_name in config["fallbacks"]:
            fallback = cls.create_detector(fallback_name)
            if fallback is not None:
                fallbacks.append(fallback)
        
        # Create and return the pipeline
        return FaceRecognitionPipeline(detector, embedding, fallbacks)
    
    @classmethod
    def create_custom_pipeline(cls, 
                              detector_name: str, 
                              embedding_name: str, 
                              fallbacks: List[str] = None) -> Optional[FaceRecognitionPipeline]:
        """
        Create a custom pipeline with specified components.
        
        Args:
            detector_name: Name of the detector
            embedding_name: Name of the embedding model
            fallbacks: List of fallback detector names
            
        Returns:
            FaceRecognitionPipeline instance or None if creation fails
        """
        # Create detector
        detector = cls.create_detector(detector_name)
        if detector is None:
            return None
        
        # Create embedding model
        embedding = cls.create_embedding(embedding_name)
        if embedding is None:
            return None
        
        # Create fallback detectors
        fallback_detectors = []
        if fallbacks:
            for fallback_name in fallbacks:
                fallback = cls.create_detector(fallback_name)
                if fallback is not None:
                    fallback_detectors.append(fallback)
        
        # Create and return the pipeline
        return FaceRecognitionPipeline(detector, embedding, fallback_detectors)
