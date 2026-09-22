import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_cosine(histograms):
    client_ids = sorted(histograms)
    matrix = {}
    for i, cid_a in enumerate(client_ids):
        matrix[cid_a] = {}
        for j, cid_b in enumerate(client_ids):
            if i == j:
                matrix[cid_a][cid_b] = 1.0
            elif cid_b in matrix and cid_a in matrix[cid_b]:
                matrix[cid_a][cid_b] = matrix[cid_b][cid_a]
            else:
                matrix[cid_a][cid_b] = cosine_similarity(histograms[cid_a], histograms[cid_b])
    return matrix
