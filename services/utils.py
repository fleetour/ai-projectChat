import numpy as np
from typing import List

def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize vector for cosine similarity"""
    vector_array = np.array(vector)
    norm = np.linalg.norm(vector_array)
    if norm == 0:
        return vector  # Return as-is if zero vector
    normalized = vector_array / norm
    return normalized.tolist()