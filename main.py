import argparse
import logging
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

def print_method_info():
    """
    Logs a summary of each method's detection & embedding approach.
    """
    info_text = r"""
====================== METHOD INFORMATION ======================

1) OpenCV (Haar Cascade) + face_recognition
   - Detection: Uses classical Haar features (frontalface_default.xml) in OpenCV.
   - Embedding: A 128-d vector from face_recognition (dlib).
   - Pros: Fast on CPU, minimal dependencies, good for frontal faces.
   - Cons: Less accurate for angled/occluded faces than modern CNN-based approaches.

2) MTCNN + face_recognition
   - Detection: MTCNN (deep-learning-based) requires TensorFlow.
   - Embedding: The same 128-d face_recognition embedding.
   - Pros: More robust detection than Haar cascades, better for multiple/angled faces.
   - Cons: Slower and heavier due to TensorFlow.

3) face_recognition (Detection + Embedding)
   - Both detection and embedding come from face_recognition (dlib).
   - Pros: Strong all-in-one library for faces, typically very good accuracy on CPU.
   - Cons: Need to compile/install dlib, can be tricky on some platforms.

4) face_recognition (Detection) + MobileNet (Embedding)
   - Detection: face_recognition (dlib) bounding box.
   - Embedding: General-purpose MobileNetV2 from TensorFlow/Keras, ~1280-d vector.
   - Pros: Flexible, easy to swap or fine-tune.
   - Cons: Not specialized for face tasks; out-of-the-box accuracy may be lower.

Choose the method that best suits your needs based on speed, memory usage, accuracy, and library preferences.
    """

    # Log each line
    for line in info_text.split('\n'):
        logger.info(line)

def run_method(method, id_image_path, webcam_image_path):
    """
    1) Detect faces using the chosen method.
    2) Embed the faces.
    3) Compare embeddings.

    Returns: (distance, similarity, total_time_seconds, memory_diff)
             or (None, None, 0, 0) if detection/embedding fails.
    """
    start_time = time.perf_counter()
    process = psutil.Process()
    mem_info_start = process.memory_info().rss

    # Face Detection
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
        # Reuse face_recognition for detection
        id_face = detect_face_facerecognition(id_image_path)
        webcam_face = detect_face_facerecognition(webcam_image_path)
    else:
        logger.info(f"Unknown method '{method}'")
        return None, None, 0, 0

    if id_face is None or webcam_face is None:
        logger.info(f"Face not detected in one or both images (method: {method})")
        return None, None, 0, 0

    # Convert to RGB
    id_face = cv2.cvtColor(id_face, cv2.COLOR_BGR2RGB)
    webcam_face = cv2.cvtColor(webcam_face, cv2.COLOR_BGR2RGB)

    # Embedding
    if method in ["opencv", "mtcnn", "facerecognition"]:
        id_embedding = get_face_embedding_facerecognition(id_face)
        webcam_embedding = get_face_embedding_facerecognition(webcam_face)
    elif method == "mobilenet":
        model = get_feature_extractor()
        id_embedding = get_face_embedding_mobilenet(model, id_face)
        webcam_embedding = get_face_embedding_mobilenet(model, webcam_face)
    else:
        logger.info(f"Unknown method '{method}' for embedding.")
        return None, None, 0, 0

    if id_embedding is None or webcam_embedding is None:
        logger.info(f"Embedding failed for one or both images (method: {method})")
        return None, None, 0, 0

    distance = compare_embeddings(id_embedding, webcam_embedding)
    similarity = cosine_similarity(id_embedding, webcam_embedding)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    mem_info_end = process.memory_info().rss
    mem_diff = mem_info_end - mem_info_start

    cleanup()
    return distance, similarity, total_time, mem_diff

def main():
    usage_examples = """
Examples:
  1) Run a single method (e.g., facerecognition):
     python main.py images/id_image.png images/test_image.png --method facerecognition

  2) Benchmark all methods:
     python main.py images/id_image.png images/test_image.png --benchmark

  3) Show info about all methods:
     python main.py --info
"""

    parser = argparse.ArgumentParser(
        description="Face Recognition using different techniques (OpenCV, MTCNN, face_recognition, MobileNet)",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "id_image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the ID/reference image."
    )
    parser.add_argument(
        "webcam_image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the webcam/test image."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["opencv", "mtcnn", "facerecognition", "mobilenet"],
        help="Which method to run (ignored if --benchmark is set)."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare performance across all methods on the given images."
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show a summary of each method (OpenCV, MTCNN, face_recognition, MobileNet)."
    )

    args = parser.parse_args()

    # If user wants method info, print it and return
    if args.info:
        print_method_info()
        return

    # If the user calls python main.py with no arguments, let's show help
    if len(vars(args)) == 0 or (
        not args.method and 
        not args.benchmark and 
        not args.id_image_path and 
        not args.webcam_image_path
    ):
        parser.print_help()
        return

    # If benchmark is set, we must have two image paths
    if args.benchmark:
        if not args.id_image_path or not args.webcam_image_path:
            logger.info("Error: Please provide two image paths for the benchmark.\n")
            parser.print_help()
            return

        methods = ["opencv", "mtcnn", "facerecognition", "mobilenet"]
        results = []

        for m in methods:
            distance, similarity, total_time, mem_diff = run_method(
                m, args.id_image_path, args.webcam_image_path
            )
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

        logger.info("\n===== BENCHMARK RESULTS =====")
        for r in results:
            logger.info(
                "Method: %15s | Status: %7s | Time: %ss | MemΔ: %d bytes | Distance: %s | Similarity: %s",
                r["method"], r["status"], r["time_s"], r["mem_diff_bytes"], r["distance"], r["similarity"]
            )
    else:
        # Single method usage
        if not args.method:
            logger.info("Error: Please specify --method or use --benchmark.\n")
            parser.print_help()
            return

        if not args.id_image_path or not args.webcam_image_path:
            logger.info("Error: Please provide two image paths (id_image_path, webcam_image_path).\n")
            parser.print_help()
            return

        distance, similarity, _, _ = run_method(
            args.method, args.id_image_path, args.webcam_image_path
        )
        if distance is None:
            logger.info("Face not detected or embedding failed for one or both images.")
            return

        logger.info(f"Face similarity (cosine similarity): {similarity:.2f}%")
        logger.info(f"Face distance (L2 norm): {distance:.4f}")

if __name__ == "__main__":
    main()
