#!/usr/bin/env python3
"""
Face Recognition System

A modular face recognition system that supports multiple detection and embedding methods.
Can be used for single face comparisons or comprehensive benchmarks.

Usage:
  1) Run a single method:
     python main.py images/id_image.png images/test_image.png --method facerecognition

  2) Benchmark all methods:
     python main.py images/id_image.png images/test_image.png --benchmark

  3) Show info about all methods:
     python main.py --info
"""

import argparse
import logging
import sys
import os
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

# Add src directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from src.core.factory import RecognitionMethodFactory
from src.pipeline import FaceRecognizer, Benchmarker
from src.benchmark import run_simple_benchmark

# Import all detectors and embedding models to register them with the factory
import src.detectors
import src.embedding


def print_method_info():
    """Print detailed information about available recognition methods"""
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
   - Detection: Uses `face_recognition.face_locations()` (HOG or CNN-based).
   - Embedding: 128-d face_recognition embedding from dlib.
   - Pros: Strong all-in-one library for faces, typically very good accuracy on CPU.
   - Cons: Need to compile/install dlib, can be tricky on some platforms.

4) MTCNN (Detection) + MobileNetV2 (Embedding)
   - Detection: MTCNN (deep-learning-based) requires TensorFlow.
   - Embedding: General-purpose MobileNetV2 from TensorFlow/Keras, ~1280-d vector.
   - Pros: Flexible, good for feature extraction, easy to swap or fine-tune.
   - Cons: Not specialized for face tasks; out-of-the-box accuracy may be lower.

5) ArcFace (Detection + Embedding)
   - Detection: Uses RetinaFace (deep-learning-based).
   - Embedding: ArcFace model (InsightFace framework).
   - Pros: High accuracy, commonly used in industry benchmarks.
   - Cons: Requires additional frameworks, more complex setup.

6) DeepFace (Detection + Embedding)
   - A wrapper library supporting multiple backends (Facenet, VGG-Face, ArcFace, etc.).
   - Detection: Uses `DeepFace.extract_faces()` (configurable backend).
   - Pros: Quick integration with multiple models, easy to use.
   - Cons: Requires additional dependencies, heavier than other methods.

Choose the method that best suits your needs based on speed, memory usage, accuracy, and library preferences.
    """
    for line in info_text.split('\n'):
        logger.info(line)


def main():
    """Main entry point for the application"""
    
    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Face Recognition System using multiple methods",
        epilog=__doc__,
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
        "test_image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the test/query image."
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=RecognitionMethodFactory.get_available_methods(),
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
        help="Show a summary of each method."
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom similarity threshold (0-1) for face matching."
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show method info if requested
    if args.info:
        print_method_info()
        return 0
    
    # Handle case with no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    # Run simple benchmark if requested
    if args.benchmark:
        if not args.id_image_path or not args.test_image_path:
            logger.error("Error: Please provide two image paths for the benchmark.")
            parser.print_help()
            return 1
        
        return run_simple_benchmark(args.id_image_path, args.test_image_path)
    
    # Run single method comparison
    else:
        if not args.method:
            logger.error("Error: Please specify --method or use --benchmark.")
            parser.print_help()
            return 1
        
        if not args.id_image_path or not args.test_image_path:
            logger.error("Error: Please provide two image paths (id_image_path, test_image_path).")
            parser.print_help()
            return 1
        
        # Create face recognizer and compare faces
        recognizer = FaceRecognizer(method=args.method, similarity_threshold=args.threshold)
        result = recognizer.compare_faces(args.id_image_path, args.test_image_path)
        
        if result is None:
            logger.error("Face detection or embedding failed for one or both images.")
            return 1
        
        # Print results
        match_status = "MATCH" if result.is_match else "NO MATCH"
        logger.info("\n===== COMPARISON RESULTS =====")
        logger.info(f"Method:           {args.method}")
        logger.info(f"Status:           {match_status}")
        logger.info(f"Similarity:       {result.similarity:.2f}%")
        logger.info(f"Distance:         {result.distance:.4f}")
        logger.info(f"Processing Time:  {result.processing_time:.4f} seconds")
        logger.info(f"Memory Usage:     {result.memory_usage/(1024*1024):.2f} MB")
        logger.info(f"CPU Usage:        {result.cpu_percent:.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
