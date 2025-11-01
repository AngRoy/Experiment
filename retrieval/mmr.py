from typing import List
import numpy as np

def mmr_select(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    k: int = 20,
    lambda_: float = 0.6,
) -> List[int]:
    """
    Select k indices using Maximal Marginal Relevance.
    embeddings: (N, D) float array for each candidate
    relevance: (N,) normalized relevance score in [0,1]
    Returns: list of indices selected in order.
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    selected: List[int] = []
    # start with the most relevant item
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining = set(range(n))
    remaining.remove(first)

    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    emb_norm = embeddings / norm

    while len(selected) < k and remaining:
        best_idx = None
        best_score = -1e9
        # precompute similarity to any selected as max cosine
        selected_mat = emb_norm[selected]
        for i in list(remaining):
            # cosine to selected
            sims = selected_mat @ emb_norm[i].reshape(-1,1)
            max_sim = float(np.max(sims)) if sims.size else 0.0
            score = lambda_ * float(relevance[i]) - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected
