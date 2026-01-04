import sys
import json
import pandas as pd
import numpy as np
import ollama


def embed_query(text, model="embeddinggemma"):
    response = ollama.embeddings(model=model, prompt=text)
    return np.array(response["embedding"], dtype=float)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main(model_csv, query):
    df = pd.read_csv(model_csv)
    query_emb = embed_query(query)

    best_score = -1
    best_row = None

    for _, row in df.iterrows():
        emb = np.array(json.loads(row["embedding"]), dtype=float)
        score = cosine_similarity(query_emb, emb)

        if score > best_score:
            best_score = score
            best_row = row

    print("Query:", query)
    print("Best match:", best_row["filename"])
    print("Score:", best_score)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python search.py model.csv \"query text\"")
        sys.exit(1)

    main(sys.argv[1], " ".join(sys.argv[2:]))
