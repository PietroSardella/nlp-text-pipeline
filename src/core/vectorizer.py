from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def generate_tfidf(documents: list[str]) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Gera matriz TF-IDF a partir dos documentos.
    """
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def transform_query(
    vectorizer: TfidfVectorizer,
    query: str
) -> csr_matrix:
    """
    Transforma a query usando o mesmo vocabulário do corpus.
    """
    return vectorizer.transform([query])