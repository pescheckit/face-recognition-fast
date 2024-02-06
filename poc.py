import cv2
import face_recognition
import argparse
import numpy as np

def detect_and_extract_face(image_path):
    # Load image
    image = face_recognition.load_image_file(image_path)

    # Try detecting face in different orientations
    for _ in range(4):
        # Detect faces
        face_locations = face_recognition.face_locations(image)

        if face_locations:
            # Extracting the first face
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            # Ensure the face image is in the correct format
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            return face_image
        else:
            # Rotate image by 90 degrees
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return None

def compare_faces(face_image1, face_image2):
    # Compute face encodings
    face_encoding1 = face_recognition.face_encodings(face_image1)
    face_encoding2 = face_recognition.face_encodings(face_image2)

    # Check if encodings are available
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

    # Detect and extract faces
    id_face = detect_and_extract_face(args.id_image_path)
    webcam_face = detect_and_extract_face(args.webcam_image_path)

    if id_face is not None and webcam_face is not None:
        # Compare faces
        match, similarity_percentage = compare_faces(id_face, webcam_face)
        print(f"Faces match: {match}, Similarity: {similarity_percentage:.2f}%")
    else:
        print("Face not detected in one or both images")

if __name__ == '__main__':
    main()
