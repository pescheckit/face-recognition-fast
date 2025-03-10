"""
Face detector using InsightFace's RetinaFace.
"""

import logging
import cv2
import numpy as np

from ..core.base import BaseFaceDetector
from ..core.data import DetectionResult, COLOR_FORMAT
from ..core.utils import load_image, ensure_face_size

logger = logging.getLogger(__name__)


class ArcFaceDetector(BaseFaceDetector):
    """Face detector using InsightFace's RetinaFace"""
    
    def __init__(self):
        """Initialize the ArcFace detector"""
        super().__init__("ArcFace")
        self._model = None
    
    def _load_model(self):
        """Load the InsightFace detector if not already loaded"""
        if self._model is None:
            try:
                import insightface
                self._model = insightface.app.FaceAnalysis(allowed_modules=['detection'])
                logger.debug("Loaded InsightFace RetinaFace detector")
            except ImportError:
                logger.error("Failed to import insightface. Please install with: pip install insightface")
                raise
    
    def detect(self, image_path):
        """
        Detect a face in an image using InsightFace's RetinaFace.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DetectionResult with face in RGB format, or None if no face is detected
        """
        try:
            # Load the image
            image = load_image(image_path)
            if image is None:
                logger.error(f"[ArcFace] Could not load image: {image_path}")
                return None
            
            # Load the detector model
            self._load_model()
            
            # Try different detection sizes and contexts
            det_sizes = [(640, 640), (1280, 1280)]
            ctx_options = [0, -1]  # First try GPU (0), then CPU (-1)
            
            faces = None
            successful_ctx = None
            successful_size = None
            
            # Try combinations of detection sizes and contexts
            for ctx_id in ctx_options:
                for det_size in det_sizes:
                    try:
                        self._model.prepare(ctx_id=ctx_id, det_size=det_size)
                        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        faces = self._model.get(img_rgb)
                        
                        if len(faces) > 0:
                            successful_ctx = ctx_id
                            successful_size = det_size
                            break  # If faces found, exit the inner loop
                    except Exception as e:
                        logger.warning(f"[ArcFace] Detection failed with ctx_id={ctx_id}, det_size={det_size}: {e}")
                        continue
                
                if faces and len(faces) > 0:
                    break  # If faces found, exit the outer loop
            
            if faces is None or len(faces) == 0:
                logger.warning(f"[ArcFace] No faces detected in image: {image_path}")
                return None
            
            logger.debug(f"[ArcFace] Face detected with ctx_id={successful_ctx}, det_size={successful_size}")
            
            # Sort faces by area (largest first)
            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            
            # Extract the largest face
            face = faces[0]
            box = face.bbox.astype(int)  # (left, top, right, bottom)
            x1, y1, x2, y2 = box
            
            # Add padding around the face (5%)
            padding = int(((x2 - x1) + (y2 - y1)) / 20)
            h, w = image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract the face region
            cropped = image[y1:y2, x1:x2]
            
            # Ensure minimum size for embedding models
            cropped = ensure_face_size(cropped)
            
            # Convert to RGB for consistency
            rgb_face = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            
            # Return detection result
            return DetectionResult(
                face_image=rgb_face,
                color_format=COLOR_FORMAT.RGB,
                bounding_box=(x1, y1, x2-x1, y2-y1),
                landmarks=face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
            )
            
        except ImportError:
            logger.error("Failed to import required libraries for ArcFace. Please install with: pip install insightface")
            return None
        except Exception as e:
            logger.error(f"[ArcFace] Detection error: {e}")
            return None
