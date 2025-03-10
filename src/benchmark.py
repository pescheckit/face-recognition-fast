"""
Simplified benchmark module for the face recognition system.
"""

import logging
import argparse
from tabulate import tabulate

from src.pipeline import Benchmarker

logger = logging.getLogger(__name__)

def run_simple_benchmark(id_image_path: str, test_image_path: str) -> int:
    """
    Run a simple benchmark comparing all methods on a pair of images.
    
    Args:
        id_image_path: Path to the ID/reference image
        test_image_path: Path to the test/query image
        
    Returns:
        Exit code (0 for success)
    """
    # Create benchmarker and run all methods
    benchmarker = Benchmarker()
    results = benchmarker.run_benchmark(id_image_path, test_image_path)
    
    # Format and print results using tabulate
    table_data = []
    headers = ["Method", "Status", "Time(s)", "Memory(MB)", "CPU(%)", "Distance", "Similarity"]
    
    for r in results:
        if r.status == "OK":
            c = r.comparison
            table_data.append([
                r.method,
                r.status,
                f"{c.processing_time:.4f}",
                f"{c.memory_usage/(1024*1024):.2f}",
                f"{c.cpu_percent:.2f}",
                f"{c.distance:.4f}",
                f"{c.similarity:.2f}%"
            ])
        else:
            table_data.append([
                r.method,
                r.status,
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            ])
    
    logger.info("\n===== BENCHMARK RESULTS =====")
    logger.info(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Recognition Benchmark Tool"
    )
    
    parser.add_argument(
        'id_image_path',
        type=str,
        help='Path to the reference/ID image'
    )
    
    parser.add_argument(
        'test_image_path',
        type=str,
        help='Path to the test image'
    )
    
    args = parser.parse_args()
    run_simple_benchmark(args.id_image_path, args.test_image_path)
