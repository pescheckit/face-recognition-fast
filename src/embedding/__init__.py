"""
Embedding implementations for face recognition system.
"""

# Import embedding classes
from .face_recognition import FaceRecognitionEmbedding
from .mobilenet import MobileNetEmbedding
from .arcface import ArcFaceEmbedding
from .deepface_embedding import DeepFaceEmbedding

# Import factory registration function
from ..core.factory import register_embedding

# Register all embeddings with the factory
register_embedding("face_recognition", FaceRecognitionEmbedding)
register_embedding("mobilenet", MobileNetEmbedding)
register_embedding("arcface", ArcFaceEmbedding)
register_embedding("deepface", DeepFaceEmbedding)
register_embedding("facenet", lambda: DeepFaceEmbedding(model_name="Facenet"))
register_embedding("facenet512", lambda: DeepFaceEmbedding(model_name="Facenet512"))
register_embedding("vggface", lambda: DeepFaceEmbedding(model_name="VGG-Face"))
register_embedding("sface", lambda: DeepFaceEmbedding(model_name="SFace"))

__all__ = [
    'FaceRecognitionEmbedding',
    'MobileNetEmbedding',
    'ArcFaceEmbedding',
    'DeepFaceEmbedding'
]
