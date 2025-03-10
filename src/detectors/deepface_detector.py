"""
Face detector using DeepFace with multiple backends.
"""

import logging
import cv2
import os
import numpy as np

from ..core.base import BaseFaceDetector
from ..core.data import DetectionResult, COLOR_FORMAT
from ..core.utils import load_image, ensure_face_size, normalize_image

logger = logging.getLogger(__name__)


class DeepFaceDetector(BaseFaceDetector):
    """Face detector using DeepFace with multiple backends"""
    
    def __init__(self):
        """Initialize the DeepFace detector"""
        super().__init__("DeepFace")
        # Detector backends to try in order of preference
        self.detector_backends = ["retinaface", "mtcnn", "opencv", "ssd", "dlib", "mediapipe"]
    
    def detect(self, image_path):
        """
        Detect a face in an image using DeepFace.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DetectionResult with face in RGB format, or None if no face is detected
        """
        try:
            from deepface import DeepFace
            
            # Load the image
            image = load_image(image_path)
            if image is None:
                logger.error(f"[DeepFace] Could not load image: {image_path}")
                return None
            
            # Check if image is valid
            if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                logger.error(f"[DeepFace] Invalid image dimensions: {image.shape}")
                return None
            
            # First, try direct extract_faces with different backends
            for backend in self.detector_backends:
                try:
                    logger.debug(f"[DeepFace] Trying detection with {backend} backend")
                    
                    # Some detector backends require specific image preprocessing
                    if backend == "opencv" or backend == "ssd":
                        # These detectors might need uint8 format
                        img = normalize_image(image)
                    else:
                        img = image
                    
                    # Try to extract faces
                    faces = DeepFace.extract_faces(
                        img_path=image_path,  # DeepFace can load directly from path
                        detector_backend=backend,
                        enforce_detection=False,
                        align=True
                    )
                    
                    if faces and len(faces) > 0 and 'face' in faces[0] and faces[0]['face'] is not None:
                        face = faces[0]['face']
                        confidence = faces[0].get('confidence', 1.0)
                        region = faces[0].get('facial_area', None)
                        
                        # Check if the extracted face has valid dimensions
                        if face.shape[0] > 0 and face.shape[1] > 0:
                            # DeepFace usually returns in RGB format
                            return DetectionResult(
                                face_image=face,
                                color_format=COLOR_FORMAT.RGB,
                                confidence=confidence,
                                bounding_box=region
                            )
                
                except Exception as e:
                    logger.warning(f"[DeepFace] Detection with {backend} backend failed: {e}")
                    continue
            
            # If all direct methods failed, try to use the verify method as a fallback
            # Sometimes verify() will detect a face even when extract_faces() fails
            try:
                logger.debug("[DeepFace] Trying detection via verify method")
                
                # Create a small reference image (a white square)
                ref_img = np.ones((112, 112, 3), dtype=np.uint8) * 255
                
                # Try to verify against our dummy reference
                result = DeepFace.verify(
                    img1_path=image_path,
                    img2_path=ref_img,
                    enforce_detection=False,
                    detector_backend="retinaface"
                )
                
                # If verification ran, a face was detected
                # We can check if the function saved a detected face
                aligned_face_path = os.path.join(os.getcwd(), "DeepFace", "face_1.jpg")
                if os.path.exists(aligned_face_path):
                    # Load the detected face
                    face = cv2.imread(aligned_face_path)
                    if face is not None:
                        # Convert to RGB
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        
                        # Clean up the temporary file
                        try:
                            os.remove(aligned_face_path)
                        except:
                            pass
                        
                        return DetectionResult(
                            face_image=face_rgb,
                            color_format=COLOR_FORMAT.RGB
                        )
            
            except Exception as e:
                logger.warning(f"[DeepFace] Verify method detection failed: {e}")
            
            logger.warning(f"[DeepFace] All detection methods failed for image: {image_path}")
            return None
            
        except ImportError:
            logger.error("Failed to import deepface. Please install with: pip install deepface")
            return None
        except Exception as e:
            logger.error(f"[DeepFace] Detection error: {e}")
            return None
