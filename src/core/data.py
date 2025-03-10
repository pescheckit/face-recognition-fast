"""
Data classes for the face recognition system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import cv2

# Define color format enum
COLOR_FORMAT = Enum('COLOR_FORMAT', ['RGB', 'BGR'])


@dataclass
class DetectionResult:
    """Class to store face detection results"""
    face_image: np.ndarray  # The cropped face image
    color_format: COLOR_FORMAT  # RGB or BGR
    confidence: float = 1.0  # Detection confidence (0-1)
    bounding_box: tuple = None  # (x, y, width, height) or (top, right, bottom, left)
    landmarks: dict = None  # Facial landmarks if available
    
    def to_rgb(self) -> 'DetectionResult':
        """Convert face image to RGB if needed"""
        if self.color_format == COLOR_FORMAT.BGR:
            self.face_image = cv2.cvtColor(self.face_image, cv2.COLOR_BGR2RGB)
            self.color_format = COLOR_FORMAT.RGB
        return self
    
    def to_bgr(self) -> 'DetectionResult':
        """Convert face image to BGR if needed"""
        if self.color_format == COLOR_FORMAT.RGB:
            self.face_image = cv2.cvtColor(self.face_image, cv2.COLOR_RGB2BGR)
            self.color_format = COLOR_FORMAT.BGR
        return self


@dataclass
class EmbeddingResult:
    """Class to store face embedding results"""
    embedding: np.ndarray  # The face embedding vector
    model_name: str  # Name of the model used
    embedding_size: int  # Size of the embedding vector
    
    def normalize(self) -> 'EmbeddingResult':
        """Normalize the embedding vector to unit length"""
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm
        return self


@dataclass
class ComparisonResult:
    """Class to store face comparison results"""
    distance: float  # L2 distance between embeddings
    similarity: float  # Cosine similarity (0-100%)
    is_match: bool = False  # Whether the faces match based on threshold
    processing_time: float = 0.0  # Total processing time in seconds
    memory_usage: int = 0  # Memory usage in bytes
    cpu_percent: float = 0.0  # CPU usage as percentage
    
    def __str__(self) -> str:
        """String representation of comparison results"""
        match_str = "MATCH" if self.is_match else "NO MATCH"
        return (
            f"Comparison Results:\n"
            f"  - Status: {match_str}\n"
            f"  - Similarity: {self.similarity:.2f}%\n"
            f"  - Distance: {self.distance:.4f}\n"
            f"  - Processing Time: {self.processing_time:.4f} seconds\n"
            f"  - Memory Usage: {self.memory_usage / (1024 * 1024):.2f} MB\n"
            f"  - CPU Usage: {self.cpu_percent:.2f}%"
        )


@dataclass
class BenchmarkResult:
    """Class to store benchmark results"""
    method: str  # Method name
    status: str  # "OK" or "FAILED"
    comparison: Optional[ComparisonResult] = None
    error: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of benchmark results"""
        if self.status == "FAILED":
            return f"{self.method:<15} | FAILED | Error: {self.error}"
        
        comp = self.comparison
        return (
            f"{self.method:<15} | {self.status:<7} | "
            f"Time: {comp.processing_time:.4f}s | "
            f"Mem: {comp.memory_usage/(1024*1024):.2f}MB | "
            f"CPU: {comp.cpu_percent:.2f}% | "
            f"Distance: {comp.distance:.4f} | "
            f"Similarity: {comp.similarity:.2f}%"
        )
