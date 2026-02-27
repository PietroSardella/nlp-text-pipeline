from src.core.loader import load_documents
from src.core.cleaner import clean_text
from src.rag.embedder import generate_embeddings, embed_query
from src.rag.retriever import search


def main():
    documents = load_documents("data/sample.txt")

    cleaned_docs = [clean_text(doc) for doc in documents]

    print("Gerando embeddings...")
    doc_embeddings = generate_embeddings(cleaned_docs)

    print("Shape dos embeddings:")
    print(doc_embeddings.shape)
    print()

    query = input("Digite sua pergunta: ")
    cleaned_query = clean_text(query)

    query_vector = embed_query(cleaned_query)

    results, scores = search(query_vector, doc_embeddings, documents)

    if not results:
        print("\nNenhum documento relevante encontrado.")
    else:
        print("\nDocumentos retornados:")
        for doc, score in zip(results, scores):
            print(f"- {doc}")
            print(f"  Score: {score:.4f}")


if __name__ == "__main__":
    main()