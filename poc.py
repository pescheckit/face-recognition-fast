import cv2
import face_recognition
import argparse
import numpy as np
import gc  # Garbage Collector interface

# Limited at 2GB
import resource
def set_memory_limit(limit_in_bytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limit_in_bytes, hard))
set_memory_limit(2048 * 2048 * 2048)  # Corrected to use 2048 for KB to MB to GB conversion

def detect_and_extract_face(image_path):
    # Load image
    image = face_recognition.load_image_file(image_path)

    for _ in range(4):  # Rotate image up to 4 times to find a face
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            del image  # Delete the original image to free its memory
            gc.collect()  # Explicitly call garbage collection
            return face_image
        else:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    del image  # Make sure to delete and collect even if no face is found
    gc.collect()
    return None

def compare_faces(face_image1, face_image2):
    face_encoding1 = face_recognition.face_encodings(face_image1)
    face_encoding2 = face_recognition.face_encodings(face_image2)

    if face_encoding1 and face_encoding2:
        matches = face_recognition.compare_faces(face_encoding1, face_encoding2[0])
        face_distance = face_recognition.face_distance(face_encoding1, face_encoding2[0])
        return matches[0], (1 - face_distance[0]) * 100
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Compare two faces in images.')
    parser.add_argument('id_image_path', type=str, help='Path to ID image')
    parser.add_argument('webcam_image_path', type=str, help='Path to webcam or other image')

    args = parser.parse_args()

    id_face = detect_and_extract_face(args.id_image_path)
    webcam_face = detect_and_extract_face(args.webcam_image_path)

    if id_face is not None and webcam_face is not None:
        match, similarity_percentage = compare_faces(id_face, webcam_face)
        print(f"Faces match: {match}, Similarity: {similarity_percentage:.2f}%")
    else:
        print("Face not detected in one or both images")

    # After processing, explicitly delete large objects and collect garbage
    del id_face, webcam_face
    gc.collect()

if __name__ == '__main__':
    main()
