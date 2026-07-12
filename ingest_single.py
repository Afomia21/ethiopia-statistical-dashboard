from pathlib import Path
import argparse
import shutil
import pandas as pd
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def load_csv_to_chunks(csv_path: Path):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False, na_filter=False)
    texts, metadatas, ids = [], [], []
    for row_index, row in enumerate(df.itertuples(index=False, name=None)):
        values = [f"{column}: {value}" for column, value in zip(df.columns, row)]
        text = "\n".join(values)
        texts.append(text)
        metadatas.append({"source": csv_path.name, "row_index": row_index})
        ids.append(f"{csv_path.name}-{row_index}")
    return texts, metadatas, ids


def main():
    parser = argparse.ArgumentParser(description="Ingest a single merged CSV file into Chroma DB.")
    parser.add_argument("--file", required=True, help="Path to the merged CSV file")
    parser.add_argument("--output-dir", default="db/chroma_db", help="Chroma DB output directory")
    parser.add_argument("--clear-output", action="store_true", help="Delete the existing vector store before ingesting")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_dir = Path(args.output_dir)
    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    texts, metadatas, ids = load_csv_to_chunks(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(is_persistent=True, persist_directory=str(output_dir))
    client = Client(settings)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        normalize_embeddings=True,
    )
    collection = client.get_or_create_collection(
        name="ethiopia_stats",
        embedding_function=embedding_function,
    )
    collection.add(documents=texts, metadatas=metadatas, ids=ids)

    print(f"Ingested {len(texts)} rows from {csv_path} into {output_dir}")


if __name__ == "__main__":
    main()
