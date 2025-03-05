import numpy as np
from scipy.spatial import distance


def compare_embeddings(embedding1, embedding2):
    """
    Returns the L2 distance between two embeddings.
    Lower = more similar.
    """
    distance_score = np.linalg.norm(embedding1 - embedding2)
    return distance_score

def cosine_similarity(embedding1, embedding2):
    """
    Returns the cosine similarity as a percentage [0..100].
    """
    similarity = 1 - distance.cosine(embedding1, embedding2)
    return similarity * 100
