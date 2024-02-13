import cv2
from scipy.spatial import distance
import argparse
import numpy as np
import resource

# Limited at 2GB
import resource
def set_memory_limit(limit_in_bytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limit_in_bytes, hard))
set_memory_limit(2048 * 2048 * 2048)  # Corrected to use 2048 for KB to MB to GB conversion

def detect_and_extract_face(image_path):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Attempt to load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Proceed with face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    
    if len(faces) > 0:
        # Assuming the first detected face is the target
        x, y, w, h = faces[0]
        face_image = image[y:y+h, x:x+w]
        return face_image
    else:
        return None

def compare_faces(embedding1, embedding2):
    # Ensure embeddings are 1-D
    embedding1 = np.asarray(embedding1).flatten()
    embedding2 = np.asarray(embedding2).flatten()

    # Calculate cosine similarity
    similarity = 1 - distance.cosine(embedding1, embedding2)
    return similarity >= 0.5, similarity * 100  # Adjust threshold as necessary

def get_face_embedding(face_image):
    # This function should extract the face embedding using a pre-trained model.
    # Placeholder for actual embedding extraction code.
    # You would need to integrate with OpenFace or another model here.
    return np.random.rand(128)  # Example: Return a dummy 128-dimension embedding

def main():
    parser = argparse.ArgumentParser(description='Compare two faces in images.')
    parser.add_argument('id_image_path', type=str, help='Path to ID image')
    parser.add_argument('webcam_image_path', type=str, help='Path to webcam or other image')

    args = parser.parse_args()

    # Detect and extract faces
    id_face = detect_and_extract_face(args.id_image_path)
    webcam_face = detect_and_extract_face(args.webcam_image_path)

    if id_face is not None and webcam_face is not None:
        # Extract embeddings for each face
        id_embedding = get_face_embedding(id_face)
        webcam_embedding = get_face_embedding(webcam_face)
        
        # Compare faces based on embeddings
        match, similarity_percentage = compare_faces(id_embedding, webcam_embedding)
        print(f"Faces match: {match}, Similarity: {similarity_percentage:.2f}%")
    else:
        print("Face not detected in one or both images")


if __name__ == '__main__':
    main()
