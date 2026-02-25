from loader import load_text
from cleaner import clean_text
from tokenizer import tokenize
from vectorizer import generate_tfidf


def main():
    text = load_text("data/sample.txt")
    cleaned = clean_text(text)

    tokens = tokenize(cleaned)
    print("Tokens:")
    print(tokens)
    print()

    vectorizer, matrix = generate_tfidf(cleaned)

    print("Features (vocabulário):")
    print(vectorizer.get_feature_names_out())
    print()

    print("Shape da matriz TF-IDF:")
    print(matrix.shape)


if __name__ == "__main__":
    main()