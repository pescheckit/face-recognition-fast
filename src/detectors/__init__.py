"""
Detector implementations for face recognition system.
"""

# Import detector classes
from .opencv import OpenCVDetector
from .mtcnn import MTCNNDetector
from .face_recognition import FaceRecognitionDetector
from .arcface import ArcFaceDetector
from .deepface_detector import DeepFaceDetector

# Import factory registration function
from ..core.factory import register_detector

# Register all detectors with the factory
register_detector("opencv", OpenCVDetector)
register_detector("mtcnn", MTCNNDetector)
register_detector("face_recognition", FaceRecognitionDetector)
register_detector("arcface", ArcFaceDetector)
register_detector("deepface", DeepFaceDetector)

__all__ = [
    'OpenCVDetector',
    'MTCNNDetector',
    'FaceRecognitionDetector',
    'ArcFaceDetector',
    'DeepFaceDetector'
]
