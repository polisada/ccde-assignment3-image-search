import sys
import json
import pandas as pd
from pathlib import Path
import ollama


def load_image_list(images_txt):
    with open(images_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def describe_image(image_path, model="ministral-3:3b"):
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Describe this image in one sentence.",
                "images": [str(image_path)],
            }
        ],
    )
    return response["message"]["content"].strip()


def embed_text(text, model="embeddinggemma"):
    response = ollama.embeddings(
        model=model,
        prompt=text,
    )
    return response["embedding"]


def main(images_txt, output_csv):
    images = load_image_list(images_txt)

    records = []
    for img in images:
        desc = describe_image(img)
        emb = embed_text(desc)

        records.append(
            {
                "filename": img,
                "description": desc,
                "embedding": json.dumps(emb),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved model with {len(df)} images to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py images.txt model.csv")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
