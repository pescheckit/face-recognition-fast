import logging
import cv2
import numpy as np
import os
import tempfile

logger = logging.getLogger(__name__)

def detect_face_opencv(image_path):
    """
    Standard OpenCV face detection using Haar Cascades.
    Returns a BGR face image or None if no face is detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error("[OpenCV] Could not load image: %s", image_path)
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return image[y:y+h, x:x+w]  # Returns in BGR format
    return None

def detect_face_mtcnn(image_path):
    """
    MTCNN-based face detection.
    Returns a BGR face image or None if no face is detected.
    """
    from mtcnn import MTCNN

    image = cv2.imread(image_path)
    if image is None:
        logger.error("[MTCNN] Could not load image: %s", image_path)
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)
    if faces:
        x, y, width, height = faces[0]['box']
        return image[y:y+height, x:x+width]  # Returns in BGR format
    return None

def detect_face_facerecognition(image_path):
    """
    Use face_recognition library for face detection.
    Returns a properly cropped and resized RGB face image.
    """
    import face_recognition
    import cv2

    try:
        image = face_recognition.load_image_file(image_path)  # Loads as RGB
    except Exception as e:
        logger.error("[FaceRecognition] Error loading image %s: %s", image_path, e)
        return None

    face_locations = face_recognition.face_locations(image)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        face = image[top:bottom, left:right]  # Extract face

        # Ensure minimum size for embedding models
        if face.shape[0] < 112 or face.shape[1] < 112:
            face = cv2.resize(face, (112, 112))  # Resize to 112x112 if too small

        return face  # Returns RGB format
    return None

def detect_face_arcface(image_path):
    """
    Detect face using InsightFace's FaceAnalysis (RetinaFace).
    Return a cropped face (in RGB format) or None if detection fails.
    """
    import insightface

    image = cv2.imread(image_path)
    if image is None:
        logger.error("[ArcFace] Could not load image: %s", image_path)
        return None

    try:
        # Prepare FaceAnalysis object with detection module
        model = insightface.app.FaceAnalysis(allowed_modules=['detection'])
        
        # Try different detection sizes if necessary
        det_sizes = [(640, 640), (1280, 1280)]
        faces = None
        
        for det_size in det_sizes:
            try:
                model.prepare(ctx_id=0, det_size=det_size)
                
                # Convert to RGB because FaceAnalysis expects RGB
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = model.get(img_rgb)
                
                if len(faces) > 0:
                    break  # If faces found, exit the loop
            except Exception as e:
                logger.warning(f"[ArcFace] Detection failed with det_size={det_size}: {e}")
                continue
        
        if faces is None or len(faces) == 0:
            # Try with CPU if GPU failed
            logger.warning("[ArcFace] No faces detected with GPU. Trying CPU...")
            model = insightface.app.FaceAnalysis(allowed_modules=['detection'])
            model.prepare(ctx_id=-1, det_size=(640, 640))
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = model.get(img_rgb)

        if len(faces) == 0:
            logger.warning("[ArcFace] No faces detected.")
            # Fall back to face_recognition
            return detect_face_facerecognition(image_path)

        # Sort faces by area (largest first)
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        
        # Each face has bbox=[left, top, right, bottom], etc.
        face = faces[0]
        box = face.bbox.astype(int)  # (left, top, right, bottom)
        x1, y1, x2, y2 = box
        
        # Add a little padding around the face
        padding = int(((x2 - x1) + (y2 - y1)) / 20)  # 5% padding
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        cropped = image[y1:y2, x1:x2]  # BGR
        
        # Ensure cropped face is at least 112x112 (minimum for most face recognition models)
        h, w = cropped.shape[:2]
        if h < 112 or w < 112:
            scale = max(112 / h, 112 / w)
            cropped = cv2.resize(cropped, (int(w * scale), int(h * scale)))
            
        # Convert to RGB for consistency with other methods
        return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        logger.error(f"[ArcFace] Detection error: {e}")
        # Fall back to face_recognition
        logger.warning("[ArcFace] Falling back to face_recognition")
        return detect_face_facerecognition(image_path)

def detect_face_deepface(image_path):
    """
    DeepFace-based face detection with multiple fallbacks.
    Returns an RGB face image or None if no face is detected.
    """
    from deepface import DeepFace

    try:
        # First try: Use face_recognition for detection (most reliable)
        face_image = detect_face_facerecognition(image_path)
        if face_image is not None:
            return face_image  # Already in RGB format
            
        # If face_recognition fails, try DeepFace
        logger.info("[DeepFace] face_recognition detection failed, trying DeepFace")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error("[DeepFace] Could not load image: %s", image_path)
            return None
        
        # Check if image is valid
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            logger.error("[DeepFace] Invalid image dimensions: %s", image.shape)
            return None
        
        # Fix the CV_64F issue by converting to uint8 if needed
        if image.dtype != np.uint8:
            logger.info(f"[DeepFace] Converting image from {image.dtype} to uint8")
            
            # Normalize and convert to uint8 safely
            if np.max(image) == np.min(image):  # Handle constant value images
                image = np.ones(image.shape, dtype=np.uint8) * 128  # Use mid-gray
            else:
                image = ((image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))).astype(np.uint8)
            
            # Save to a temporary file with the corrected format
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                temp_path = temp.name
                cv2.imwrite(temp_path, image)
                
            use_path = temp_path  # Use this temporary path instead
        else:
            use_path = image_path
        
        # Try extract_faces
        try:
            faces = DeepFace.extract_faces(
                img_path=use_path,
                detector_backend="opencv",
                enforce_detection=False,
                align=True
            )
            
            # Clean up temp file if we created one
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            if faces and len(faces) > 0 and 'face' in faces[0] and faces[0]['face'] is not None:
                face = faces[0]['face']
                # Check if the extracted face has valid dimensions
                if face.shape[0] > 0 and face.shape[1] > 0:
                    return face  # DeepFace returns in RGB format
        except Exception as e:
            logger.warning(f"[DeepFace] extract_faces failed: {e}")
            
            # Clean up temp file if we created one
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        # Last resort, try OpenCV's built-in face detector
        logger.info("[DeepFace] Falling back to OpenCV Haar Cascade")
        face = detect_face_opencv(use_path)
        
        # Clean up temp file if we created one
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        if face is not None:
            # Convert from BGR to RGB
            return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
        return None
        
    except Exception as e:
        logger.error(f"[DeepFace] Error in face detection: {e}")
        
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
            
        return None
