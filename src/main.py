import argparse

from src.core.loader import load_documents
from src.core.cleaner import clean_text
from src.core.vectorizer import generate_tfidf, transform_query

from src.rag.embedder import generate_embeddings, embed_query
from src.rag.retriever import search
from src.rag.hybrid_retriever import hybrid_search, debug_scores


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline RAG experimental"
    )

    parser.add_argument(
        "--mode",
        choices=["semantic", "hybrid"],
        default="semantic",
        help="Modo de busca: semantic ou hybrid"
    )

    parser.add_argument(
        "--lambda_weight",
        type=float,
        default=0.7,
        help="Peso do embedding no modo híbrido (0 a 1)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Exibir debug de similaridade (apenas modo híbrido)"
    )

    args = parser.parse_args()

    # -------------------------
    # 1. Carregar documentos
    # -------------------------
    documents = load_documents("data")
    cleaned_docs = [clean_text(doc) for doc in documents]

    # -------------------------
    # 2. Embeddings
    # -------------------------
    print("Gerando embeddings...")
    doc_embeddings = generate_embeddings(cleaned_docs)

    query = input("Digite sua pergunta: ")
    cleaned_query = clean_text(query)
    query_vector = embed_query(cleaned_query)

    # -------------------------
    # 3. Execução por modo
    # -------------------------
    if args.mode == "semantic":
        results, scores = search(
            query_vector,
            doc_embeddings,
            documents
        )

    else:  # hybrid
        vectorizer, tfidf_matrix = generate_tfidf(cleaned_docs)
        tfidf_query_vector = transform_query(vectorizer, cleaned_query)

        results, scores, semantic_scores, lexical_scores, final_scores = hybrid_search(
            query_vector,
            doc_embeddings,
            tfidf_matrix,
            tfidf_query_vector,
            documents,
            lambda_weight=args.lambda_weight
        )

        if args.debug:
            debug_scores(
                semantic_scores,
                lexical_scores,
                final_scores,
                documents
            )

    # -------------------------
    # 4. Exibir resultados
    # -------------------------
    if not results:
        print("\nNenhum documento relevante encontrado.")
    else:
        print("\nDocumentos retornados:")
        for doc, score in zip(results, scores):
            print(f"- {doc}")
            print(f"  Score: {score:.4f}")


if __name__ == "__main__":
    main()