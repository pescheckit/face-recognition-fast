"""
Comparison metrics for face recognition embeddings.
"""

from .metrics import (
    euclidean_distance,
    cosine_similarity,
    cosine_distance,
    compare_embeddings,
    is_match
)

__all__ = [
    'euclidean_distance',
    'cosine_similarity',
    'cosine_distance',
    'compare_embeddings',
    'is_match'
]
