"""
High-level face recognition pipeline components.
"""

import logging
import time
import psutil
import numpy as np
from typing import List, Optional, Tuple
import gc

from ..core.factory import RecognitionMethodFactory
from ..core.data import ComparisonResult, BenchmarkResult
from ..comparison.metrics import compare_embeddings

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """High-level face recognition manager"""
    
    def __init__(self, method: str = "facerecognition", similarity_threshold: float = None):
        """
        Initialize the face recognizer.
        
        Args:
            method: Recognition method to use (one of the supported methods)
            similarity_threshold: Custom similarity threshold (0-1), overrides default if provided
        """
        self.method = method
        self.pipeline = RecognitionMethodFactory.create_pipeline(method)
        
        if self.pipeline is None:
            logger.error(f"Failed to create pipeline for method '{method}'")
            raise ValueError(f"Invalid method: {method}")
            
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized FaceRecognizer with method: {method}")
    
    def compare_faces(self, id_image_path: str, test_image_path: str) -> Optional[ComparisonResult]:
        """
        Compare faces in two images and return detailed results.
        
        Args:
            id_image_path: Path to the reference/ID image
            test_image_path: Path to the test/query image
            
        Returns:
            ComparisonResult object or None if comparison fails
        """
        import gc
        process = psutil.Process()
        
        # Force garbage collection before measurement
        gc.collect()
        time.sleep(0.1)  # Small delay to let system stabilize
        
        # Get baseline memory and CPU usage
        mem_info_start = process.memory_info().rss
        process.cpu_percent(interval=None)  # First call to cpu_percent always returns 0.0
        
        start_time = time.perf_counter()
        
        try:
            # Process reference image
            id_detection, id_embedding = self.pipeline.process_image(id_image_path)
            if id_detection is None or id_embedding is None:
                logger.warning(f"Face detection or embedding failed for reference image: {id_image_path}")
                return None
            
            # Process test image
            test_detection, test_embedding = self.pipeline.process_image(test_image_path)
            if test_detection is None or test_embedding is None:
                logger.warning(f"Face detection or embedding failed for test image: {test_image_path}")
                return None
            
            # Convert embeddings to numpy arrays if needed
            id_emb = np.array(id_embedding.embedding, dtype=np.float32)
            test_emb = np.array(test_embedding.embedding, dtype=np.float32)
            
            # If embeddings are different lengths, we can't compare them
            if id_emb.shape != test_emb.shape:
                logger.warning(f"Embedding shapes don't match: {id_emb.shape} vs {test_emb.shape}")
                return None
            
            # Compare embeddings with comprehensive metrics
            comparison = compare_embeddings(id_emb, test_emb, id_embedding.model_name)
            
            # Override threshold if specified
            if self.similarity_threshold is not None:
                is_match = comparison["cosine_similarity"] > self.similarity_threshold
            else:
                is_match = comparison["is_match"]
            
            # Calculate processing metrics
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            # Get memory and CPU usage
            mem_info_end = process.memory_info().rss
            memory_diff = max(0, mem_info_end - mem_info_start)  # Ensure non-negative
            
            # Normalize CPU percentage by number of cores
            cpu_percent = process.cpu_percent(interval=None)
            num_cores = psutil.cpu_count(logical=True)
            normalized_cpu_percent = min(100.0, cpu_percent / num_cores)
            
            # Create and return comparison result
            return ComparisonResult(
                distance=comparison["euclidean_distance"],
                similarity=comparison["cosine_similarity_percent"],
                is_match=is_match,
                processing_time=processing_time,
                memory_usage=memory_diff,
                cpu_percent=normalized_cpu_percent
            )
            
        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return None
        finally:
            # Ensure resources are cleaned up
            self.pipeline.cleanup()
    
    def verify(self, id_image_path: str, test_image_path: str) -> Tuple[bool, float]:
        """
        Simple verification API - returns match status and similarity.
        
        Args:
            id_image_path: Path to the reference/ID image
            test_image_path: Path to the test/query image
            
        Returns:
            Tuple of (is_match, similarity_percentage)
        """
        result = self.compare_faces(id_image_path, test_image_path)
        if result is None:
            return False, 0.0
            
        return result.is_match, result.similarity


class Benchmarker:
    """Benchmark different face recognition methods"""
    
    def __init__(self):
        """Initialize the benchmarker"""
        self.available_methods = RecognitionMethodFactory.get_available_methods()
        
    def run_benchmark(self, id_image_path: str, test_image_path: str, 
                     methods: List[str] = None) -> List[BenchmarkResult]:
        """
        Benchmark multiple face recognition methods on a pair of images.
        
        Args:
            id_image_path: Path to the reference/ID image
            test_image_path: Path to the test/query image
            methods: List of methods to benchmark (defaults to all available methods)
            
        Returns:
            List of BenchmarkResult objects with performance metrics
        """
        if methods is None:
            methods = self.available_methods
        
        results = []
        
        for method in methods:
            logger.info(f"Benchmarking method: {method}")
            
            try:
                # Create recognizer for this method
                recognizer = FaceRecognizer(method=method)
                
                # Run comparison
                comparison = recognizer.compare_faces(id_image_path, test_image_path)
                
                if comparison is None:
                    # Failed case
                    results.append(BenchmarkResult(
                        method=method,
                        status="FAILED",
                        error="Face detection or embedding failed"
                    ))
                else:
                    # Success case
                    results.append(BenchmarkResult(
                        method=method,
                        status="OK",
                        comparison=comparison
                    ))
                    
            except Exception as e:
                # Error case
                logger.error(f"Error benchmarking method {method}: {e}")
                results.append(BenchmarkResult(
                    method=method,
                    status="FAILED",
                    error=str(e)
                ))
        
        return results
