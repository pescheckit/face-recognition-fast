"""
Utility functions for the face recognition system.
"""

import gc
import os
import tempfile
import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Define constants
MIN_FACE_SIZE = (112, 112)  # Minimum size for face recognition models


def cleanup():
    """Force garbage collection to free memory"""
    gc.collect()


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from path, with error handling.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array or None if loading fails
    """
    try:
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def validate_face_image(face_image: np.ndarray) -> bool:
    """
    Validate if a face image is usable for recognition.
    
    Args:
        face_image: Face image as numpy array
        
    Returns:
        True if image is valid, False otherwise
    """
    if face_image is None:
        return False
    if face_image.shape[0] == 0 or face_image.shape[1] == 0:
        return False
    if len(face_image.shape) != 3:  # Must be a color image
        return False
    return True


def ensure_face_size(face_image: np.ndarray, min_size=MIN_FACE_SIZE) -> np.ndarray:
    """
    Resize face if it's smaller than minimum required size.
    
    Args:
        face_image: Face image as numpy array
        min_size: Minimum (height, width) in pixels
        
    Returns:
        Resized face image if needed, or original if already large enough
    """
    if face_image is None:
        return None
        
    h, w = face_image.shape[:2]
    if h < min_size[0] or w < min_size[1]:
        scale = max(min_size[0] / h, min_size[1] / w)
        face_image = cv2.resize(face_image, (int(w * scale), int(h * scale)))
    return face_image


def safe_temp_file(face_image: np.ndarray, suffix='.jpg') -> Tuple[str, bool]:
    """
    Safely create a temporary file with proper cleanup handling.
    
    Args:
        face_image: Image to save to temporary file
        suffix: File extension
        
    Returns:
        Tuple of (file_path, success_flag)
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp_path = temp.name
            cv2.imwrite(temp_path, face_image)
        return temp_path, True
    except Exception as e:
        logger.error(f"Error creating temporary file: {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        return None, False


def format_bytes(bytes: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024 or unit == 'GB':
            return f"{bytes:.2f} {unit}"
        bytes /= 1024


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 255] range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image as uint8
    """
    if image.dtype != np.uint8:
        # Handle constant value images
        if np.max(image) == np.min(image):
            return np.ones(image.shape, dtype=np.uint8) * 128  # Use mid-gray
        else:
            # Normalize to [0, 255] range
            normalized = ((image - np.min(image)) * (255.0 / (np.max(image) - np.min(image))))
            return normalized.astype(np.uint8)
    return image
