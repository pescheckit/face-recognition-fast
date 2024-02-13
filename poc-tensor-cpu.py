import numpy as np
import argparse
from mtcnn import MTCNN
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import resource

# Limited at 2GB
import resource
def set_memory_limit(limit_in_bytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limit_in_bytes, hard))
set_memory_limit(2048 * 2048 * 2048)  # Corrected to use 2048 for KB to MB to GB conversion

def detect_and_extract_face(image_path, detector):
    image = Image.open(image_path)
    image = image.convert('RGB')

    def detect_face(img):
        pixels = np.asarray(img)
        results = detector.detect_faces(pixels)
        if results:
            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            face_image = Image.fromarray(face)
            face_image = face_image.resize((224, 224))
            return np.asarray(face_image)
        return None

    # First attempt at face detection
    face_array = detect_face(image)
    if face_array is not None:
        return face_array

    # Rotate image 180 degrees if no face detected initially
    rotated_image = image.rotate(180)
    face_array = detect_face(rotated_image)
    return face_array

def get_feature_extractor():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    return model

def preprocess_image(image):
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def get_face_embedding(model, face_pixels):
    face_pixels = preprocess_image(face_pixels)
    embedding = model.predict(face_pixels)
    return embedding.flatten()

def compare_faces(embedding1, embedding2):
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance

def main():
    parser = argparse.ArgumentParser(description='Compare two faces in images.')
    parser.add_argument('id_image_path', type=str, help='Path to ID image')
    parser.add_argument('webcam_image_path', type=str, help='Path to webcam or other image')
    
    args = parser.parse_args()
    detector = MTCNN()
    model = get_feature_extractor()
    
    id_face = detect_and_extract_face(args.id_image_path, detector)
    webcam_face = detect_and_extract_face(args.webcam_image_path, detector)
    
    if id_face is not None and webcam_face is not None:
        id_embedding = get_face_embedding(model, id_face)
        webcam_embedding = get_face_embedding(model, webcam_face)
        distance = compare_faces(id_embedding, webcam_embedding)
        print(f"Faces distance: {distance}")
    else:
        print("Face not detected in one or both images")

if __name__ == '__main__':
    main()
