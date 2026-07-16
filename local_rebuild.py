"""
Builds BOTH ChromaDB collections locally (stats + all PDFs in ./pdf),
using the SAME multilingual embedding model as the deployed app, so the
committed database matches exactly what production expects.

Run once locally:
    pip install sentence-transformers pymupdf
    python local_rebuild.py

Then upload the resulting db/chroma_db folder to GitHub (same way you
uploaded the PDFs) so the deployed app starts up with everything already
built in - no runtime rebuild needed, nothing lost on server restart.
"""

from pathlib import Path

import pandas as pd
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF

DB_DIR = Path("db") / "chroma_db"
STATS_COLLECTION = "esps_stats"
PDF_COLLECTION = "ess_pdf_docs"
STATS_FILE = Path("data set") / "preprocessed" / "aggregate_stats.csv"
PDF_DIR = Path("data") / "pdf"
MULTILINGUAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def split_pdf_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def extract_pdf_chunks(pdf_path: Path) -> list:
    doc = fitz.open(pdf_path)
    chunks = []
    for page_number in range(len(doc)):
        page_text = doc[page_number].get_text().strip()
        if not page_text or len(page_text) < 20:
            continue
        for i, chunk in enumerate(split_pdf_text(page_text)):
            chunks.append({"text": chunk, "source": pdf_path.name, "page": page_number + 1, "chunk_index": i})
    doc.close()
    return chunks


def main():
    print("Loading multilingual embedding model (first run downloads it, may take a minute)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MULTILINGUAL_MODEL_NAME)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(DB_DIR))

    # --- Stats collection ---
    print(f"\nBuilding '{STATS_COLLECTION}' from {STATS_FILE} ...")
    try:
        client.delete_collection(STATS_COLLECTION)
    except Exception:
        pass
    stats_collection = client.create_collection(name=STATS_COLLECTION, embedding_function=embed_fn)

    df = pd.read_csv(STATS_FILE)
    docs, ids, metas = [], [], []
    for i, row in df.iterrows():
        docs.append(str(row.get("text", "")))
        ids.append(f"stat_{i}")
        metas.append({"group": str(row.get("group", "")), "indicator": str(row.get("indicator", ""))})
        if len(docs) >= 100:
            stats_collection.add(documents=docs, ids=ids, metadatas=metas)
            docs, ids, metas = [], [], []
    if docs:
        stats_collection.add(documents=docs, ids=ids, metadatas=metas)
    print(f"  {len(df)} statistics indexed.")

    # --- PDF collection ---
    print(f"\nBuilding '{PDF_COLLECTION}' from PDFs in {PDF_DIR} ...")
    try:
        client.delete_collection(PDF_COLLECTION)
    except Exception:
        pass
    pdf_collection = client.create_collection(name=PDF_COLLECTION, embedding_function=embed_fn)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIR}. Make sure all 12 PDFs are in a folder named 'pdf' next to app.py."
        )

    total_chunks = 0
    for pdf_path in pdf_files:
        print(f"  Extracting {pdf_path.name} ...")
        chunks = extract_pdf_chunks(pdf_path)
        safe_name = "".join(c if c.isalnum() else "_" for c in pdf_path.name)
        ids = [f"pdf_{safe_name}_{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]} for c in chunks]

        batch_size = 50
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            pdf_collection.add(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])
        print(f"    {len(chunks)} chunks added.")
        total_chunks += len(chunks)

    print(f"\nDone. Stats: {len(df)} rows. PDFs: {len(pdf_files)} files, {total_chunks} chunks total.")
    print(f"\nNow upload the '{DB_DIR}' folder to GitHub, then reboot the Streamlit Cloud app.")


if __name__ == "__main__":
    main()