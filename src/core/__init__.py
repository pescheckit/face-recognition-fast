"""
Core module for face recognition system.
Contains base classes, data models, and utilities.
"""

from .data import (
    DetectionResult,
    EmbeddingResult,
    ComparisonResult,
    BenchmarkResult,
    COLOR_FORMAT
)

from .base import (
    BaseFaceDetector,
    BaseFaceEmbedding,
    FaceRecognitionPipeline
)

from .factory import RecognitionMethodFactory

from .utils import (
    cleanup,
    load_image,
    validate_face_image,
    ensure_face_size,
    safe_temp_file
)

__all__ = [
    'DetectionResult',
    'EmbeddingResult',
    'ComparisonResult',
    'BenchmarkResult',
    'COLOR_FORMAT',
    'BaseFaceDetector',
    'BaseFaceEmbedding',
    'FaceRecognitionPipeline',
    'RecognitionMethodFactory',
    'cleanup',
    'load_image',
    'validate_face_image',
    'ensure_face_size',
    'safe_temp_file'
]