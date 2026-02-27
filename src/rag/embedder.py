from sentence_transformers import SentenceTransformer
import numpy as np

# Modelo leve e eficiente para CPU
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(documents: list[str]):
    embeddings = model.encode(documents, convert_to_numpy=True)
    return embeddings

def embed_query(query: str):
    return model.encode([query], convert_to_numpy=True)[0]