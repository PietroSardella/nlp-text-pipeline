import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def hybrid_search(
    query_vector: np.ndarray,
    document_embeddings: np.ndarray,
    tfidf_matrix: np.ndarray,
    tfidf_query_vector: np.ndarray,
    documents: list[str],
    lambda_weight: float = 0.7,
    k: int = 2,
) -> tuple[
    list[str],
    list[float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    if not 0 <= lambda_weight <= 1:
        raise ValueError("lambda_weight deve estar entre 0 e 1")

    query_vector = query_vector.reshape(1, -1)

    # Similaridade semântica
    semantic_scores = cosine_similarity(
        query_vector,
        document_embeddings
    )[0]

    # Similaridade lexical
    lexical_scores = cosine_similarity(
        tfidf_query_vector,
        tfidf_matrix
    )[0]

    # Combinação híbrida
    final_scores = (
        lambda_weight * semantic_scores
        + (1 - lambda_weight) * lexical_scores
    )

    sorted_indices = np.argsort(final_scores)[::-1][:k]

    results = [documents[i] for i in sorted_indices]
    scores = [final_scores[i] for i in sorted_indices]

    return results, scores, semantic_scores, lexical_scores, final_scores


def debug_scores(
    semantic_scores: np.ndarray,
    lexical_scores: np.ndarray,
    final_scores: np.ndarray,
    documents: list[str],
    top_k: int = 5,
) -> None:

    print("\n=== DEBUG SIMILARIDADE HÍBRIDA ===")

    sorted_indices = np.argsort(final_scores)[::-1][:top_k]

    for idx in sorted_indices:
        print(f"\nDocumento: {documents[idx]}")
        print(f"Semantic: {semantic_scores[idx]:.4f}")
        print(f"Lexical : {lexical_scores[idx]:.4f}")
        print(f"Final   : {final_scores[idx]:.4f}")