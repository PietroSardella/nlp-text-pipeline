from sklearn.feature_extraction.text import TfidfVectorizer

def generate_tfidf(text: str):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([text])
    return vectorizer, matrix