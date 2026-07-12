from pathlib import Path

import pandas as pd
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

STATS_FILE = Path("data set") / "preprocessed" / "aggregate_stats.csv"
DB_DIR = Path("db") / "chroma_db"
COLLECTION_NAME = "esps_stats"


def load_documents() -> pd.DataFrame:
    df = pd.read_csv(STATS_FILE)
    if df.empty:
        raise ValueError(f"No rows found in {STATS_FILE} — did you run build_aggregate_stats.py first?")
    return df


def main():
    print(f"Loading statistics from {STATS_FILE} ...")
    df = load_documents()
    print(f"Loaded {len(df)} statistics.")

    print(f"Connecting to ChromaDB at {DB_DIR} ...")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(DB_DIR))

    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    ids = [f"stat_{i}" for i in range(len(df))]
    documents = df["text"].tolist()
    metadatas = df[["group", "indicator", "value"]].astype(str).to_dict(orient="records")

    print("Embedding and storing documents (this may take a little while on first run)...")
    batch_size = 50
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  stored {end if end < len(documents) else len(documents)}/{len(documents)}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} documents.")

    print("\nTest query: 'literacy rate in Amhara'")
    result = collection.query(query_texts=["literacy rate in Amhara"], n_results=3)
    for doc in result["documents"][0]:
        print(" -", doc)


if __name__ == "__main__":
    main()