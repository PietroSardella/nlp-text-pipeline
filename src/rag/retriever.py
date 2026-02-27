import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def search(query_vector, document_embeddings, documents,
           k=2,
           threshold=0.40,
           gap_threshold=0.25):

    # 1️⃣ Calcular similaridades
    similarities = cosine_similarity(
        [query_vector], document_embeddings
    )[0]

    # 2️⃣ Ordenar índices por score (decrescente)
    sorted_indices = np.argsort(similarities)[::-1]

    # 3️⃣ Aplicar threshold
    filtered_indices = [
        idx for idx in sorted_indices
        if similarities[idx] >= threshold
    ]

    # Se nenhum documento passar do threshold
    if not filtered_indices:
        return [], []

    # 4️⃣ Aplicar gap (se houver pelo menos 2 documentos)
    if len(filtered_indices) > 1:
        first_score = similarities[filtered_indices[0]]
        second_score = similarities[filtered_indices[1]]
        gap = first_score - second_score

        if gap > gap_threshold:
            # Retorna apenas o mais relevante
            idx = filtered_indices[0]
            return [documents[idx]], [first_score]

    # 5️⃣ Aplicar Top-k
    top_k_indices = filtered_indices[:k]

    results = [documents[idx] for idx in top_k_indices]
    scores = [similarities[idx] for idx in top_k_indices]

    return results, scores