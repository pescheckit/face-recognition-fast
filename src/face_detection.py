import logging

import cv2

logger = logging.getLogger(__name__)

def detect_face_opencv(image_path):
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
        return image[y:y+h, x:x+w]
    # Optionally, you could log a warning if no face is found
    # logger.warning("[OpenCV] No faces found in %s", image_path)
    return None

def detect_face_mtcnn(image_path):
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
        return image[y:y+height, x:x+width]
    # logger.warning("[MTCNN] No faces found in %s", image_path)
    return None

def detect_face_facerecognition(image_path):
    import face_recognition

    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        logger.error("[FaceRecognition] Error loading image %s: %s", image_path, e)
        return None

    face_locations = face_recognition.face_locations(image)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        return image[top:bottom, left:right]
    # logger.warning("[FaceRecognition] No faces found in %s", image_path)
    return None
