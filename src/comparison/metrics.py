"""
Distance and similarity metrics for comparing face embeddings.
"""

import logging
import numpy as np
from scipy.spatial import distance
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Thresholds for different embedding types
# These values should be tuned based on your specific dataset and requirements
DEFAULT_THRESHOLDS = {
    "FaceRecognition": 0.6,  # face_recognition default is 0.6
    "MobileNet": 0.75,       # Arbitrary, should be tuned
    "ArcFace": 0.68,         # From InsightFace recommendations
    "DeepFace-ArcFace": 0.68,
    "DeepFace-Facenet": 0.4,
    "DeepFace-Facenet512": 0.3,
    "DeepFace-VGG-Face": 0.4,
    "DeepFace-OpenFace": 0.4,
    "DeepFace-DeepFace": 0.3,
    "DeepFace-DeepID": 0.4,
    "DeepFace-SFace": 0.5,
    # Default fallback
    "default": 0.5
}


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculates the Euclidean (L2) distance between two embeddings.
    Lower values indicate higher similarity.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Euclidean distance between embeddings
    """
    return np.linalg.norm(embedding1 - embedding2)


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two embeddings.
    Higher values (closer to 1.0) indicate higher similarity.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity as a value between -1.0 and 1.0
    """
    return 1 - distance.cosine(embedding1, embedding2)


def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculates the cosine distance between two embeddings.
    Lower values indicate higher similarity.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine distance as a value between 0.0 and 2.0
    """
    return distance.cosine(embedding1, embedding2)


def compare_embeddings(embedding1: np.ndarray, 
                      embedding2: np.ndarray, 
                      model_name: str = None) -> Dict[str, Any]:
    """
    Comprehensive comparison of two embeddings using multiple metrics.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        model_name: Name of the model used for embedding (for threshold selection)
        
    Returns:
        Dictionary with various distance and similarity metrics
    """
    if embedding1.shape != embedding2.shape:
        logger.error(f"Embedding shapes don't match: {embedding1.shape} vs {embedding2.shape}")
        return None
    
    try:
        # Calculate various metrics
        l2_distance = euclidean_distance(embedding1, embedding2)
        similarity = cosine_similarity(embedding1, embedding2)
        cos_distance = cosine_distance(embedding1, embedding2)
        
        # Get appropriate threshold
        threshold = get_threshold(model_name)
        
        # Determine match status
        is_match_value = similarity > threshold
        
        return {
            "euclidean_distance": l2_distance,
            "cosine_similarity": similarity,
            "cosine_similarity_percent": similarity * 100,
            "cosine_distance": cos_distance,
            "threshold": threshold,
            "is_match": is_match_value,
            "model_name": model_name
        }
        
    except Exception as e:
        logger.error(f"Error comparing embeddings: {e}")
        return None


def get_threshold(model_name: str = None) -> float:
    """
    Get the appropriate similarity threshold for a given model.
    
    Args:
        model_name: Name of the model used for embedding
        
    Returns:
        Similarity threshold value
    """
    if model_name is None:
        return DEFAULT_THRESHOLDS["default"]
    
    return DEFAULT_THRESHOLDS.get(model_name, DEFAULT_THRESHOLDS["default"])


def is_match(embedding1: np.ndarray, 
            embedding2: np.ndarray, 
            model_name: str = None,
            threshold: float = None) -> bool:
    """
    Determine if two face embeddings match.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        model_name: Name of the model used for embedding (for threshold selection)
        threshold: Custom threshold (overrides model-based threshold if provided)
        
    Returns:
        True if faces match, False otherwise
    """
    if threshold is None:
        threshold = get_threshold(model_name)
    
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity > threshold
