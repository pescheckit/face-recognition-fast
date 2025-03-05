import argparse
import time

import cv2
import psutil

from src.compare_faces import compare_embeddings, cosine_similarity
from src.face_detection import (detect_face_facerecognition, detect_face_mtcnn,
                                detect_face_opencv)
from src.face_embedding import (get_face_embedding_facerecognition,
                                get_face_embedding_mobilenet,
                                get_feature_extractor)
from src.utils import cleanup


def run_method(method, id_image_path, webcam_image_path):
    """
    Helper function to:
      1) Detect faces using the chosen method.
      2) Embed the faces using the chosen method.
      3) Compare the embeddings.
    
    Returns:
      (distance, similarity, total_time_seconds, peak_memory_diff_bytes)
    """
    # Start timing and memory usage measurement
    start_time = time.perf_counter()
    process = psutil.Process()
    mem_info_start = process.memory_info().rss  # in bytes

    # ------------------ Face Detection ------------------
    if method == "opencv":
        id_face = detect_face_opencv(id_image_path)
        webcam_face = detect_face_opencv(webcam_image_path)
    elif method == "mtcnn":
        id_face = detect_face_mtcnn(id_image_path)
        webcam_face = detect_face_mtcnn(webcam_image_path)
    elif method == "facerecognition":
        id_face = detect_face_facerecognition(id_image_path)
        webcam_face = detect_face_facerecognition(webcam_image_path)
    elif method == "mobilenet":
        # By default, re-use face_recognition for detection
        id_face = detect_face_facerecognition(id_image_path)
        webcam_face = detect_face_facerecognition(webcam_image_path)
    else:
        raise ValueError(f"Unknown method '{method}'")

    if id_face is None or webcam_face is None:
        # Return a sentinel if detection failed
        return None, None, 0, 0

    # Convert to RGB for face_recognition or consistency
    id_face = cv2.cvtColor(id_face, cv2.COLOR_BGR2RGB)
    webcam_face = cv2.cvtColor(webcam_face, cv2.COLOR_BGR2RGB)

    # ------------------ Face Embedding ------------------
    if method in ["opencv", "mtcnn", "facerecognition"]:
        # Use face_recognition embeddings
        id_embedding = get_face_embedding_facerecognition(id_face)
        webcam_embedding = get_face_embedding_facerecognition(webcam_face)
    elif method == "mobilenet":
        model = get_feature_extractor()
        id_embedding = get_face_embedding_mobilenet(model, id_face)
        webcam_embedding = get_face_embedding_mobilenet(model, webcam_face)
    else:
        raise ValueError(f"Unknown method '{method}'")

    if id_embedding is None or webcam_embedding is None:
        # Return a sentinel if embedding failed
        return None, None, 0, 0

    # ------------------ Compare Embeddings ------------------
    distance = compare_embeddings(id_embedding, webcam_embedding)
    similarity = cosine_similarity(id_embedding, webcam_embedding)

    # End timing and memory usage measurement
    end_time = time.perf_counter()
    total_time = end_time - start_time

    mem_info_end = process.memory_info().rss
    mem_diff = mem_info_end - mem_info_start  # how many bytes we grew

    # Cleanup any leftover memory (if needed)
    cleanup()

    return distance, similarity, total_time, mem_diff

def main():
    parser = argparse.ArgumentParser(description="Face Recognition using different techniques")
    parser.add_argument("id_image_path", type=str, help="Path to the ID image")
    parser.add_argument("webcam_image_path", type=str, help="Path to the webcam image")
    parser.add_argument(
        "--method",
        type=str,
        choices=["opencv", "mtcnn", "facerecognition", "mobilenet"],
        default=None,
        help="Face recognition method to use (ignored if --benchmark is set)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="If set, run all methods (opencv, mtcnn, facerecognition, mobilenet) and compare performance."
    )

    args = parser.parse_args()

    if args.benchmark:
        # Run ALL methods in a loop
        methods = ["opencv", "mtcnn", "facerecognition", "mobilenet"]

        # A place to store results
        results = []

        for m in methods:
            distance, similarity, total_time, mem_diff = run_method(
                m, args.id_image_path, args.webcam_image_path
            )

            # If detection/embedding failed, distance/similarity will be None
            # We handle that gracefully here:
            if distance is None:
                results.append({
                    "method": m,
                    "distance": None,
                    "similarity": None,
                    "time_s": total_time,
                    "mem_diff_bytes": mem_diff,
                    "status": "FAILED"
                })
            else:
                results.append({
                    "method": m,
                    "distance": f"{distance:.4f}",
                    "similarity": f"{similarity:.2f}%",
                    "time_s": f"{total_time:.4f}",
                    "mem_diff_bytes": mem_diff,
                    "status": "OK"
                })

        # Print summary
        print("\n===== BENCHMARK RESULTS =====")
        for r in results:
            print(
                f"Method: {r['method']:>15} | "
                f"Status: {r['status']:>7} | "
                f"Time: {r['time_s']}s | "
                f"MemΔ: {r['mem_diff_bytes']} bytes | "
                f"Distance: {r['distance']} | "
                f"Similarity: {r['similarity']}"
            )

    else:
        # NORMAL flow: run just one method
        if not args.method:
            print("Error: You must specify --method if not using --benchmark.")
            return

        distance, similarity, _, _ = run_method(
            args.method, args.id_image_path, args.webcam_image_path
        )

        if distance is None:
            print("Face not detected or embedding failed for one or both images.")
            return

        print(f"Face similarity (cosine similarity): {similarity:.2f}%")
        print(f"Face distance (L2 norm): {distance:.4f}")

if __name__ == "__main__":
    main()
