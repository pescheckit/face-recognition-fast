import cv2

# We do NOT import MTCNN or face_recognition globally.
# Instead, we do local imports inside each function.

def detect_face_opencv(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[OpenCV] Error: Could not load image {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return image[y:y+h, x:x+w]
    return None

def detect_face_mtcnn(image_path):
    # MTCNN -> local import so TensorFlow does NOT load 
    # if you never call this function.
    from mtcnn import MTCNN

    image = cv2.imread(image_path)
    if image is None:
        print(f"[MTCNN] Error: Could not load image {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)
    if faces:
        x, y, width, height = faces[0]['box']
        return image[y:y+height, x:x+width]
    return None

def detect_face_facerecognition(image_path):
    # face_recognition -> local import
    import face_recognition

    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"[FaceRecognition] Error loading image {image_path}: {e}")
        return None

    face_locations = face_recognition.face_locations(image)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        # face_recognition loads images in RGB
        return image[top:bottom, left:right]
    return None
