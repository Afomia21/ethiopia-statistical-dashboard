import os
from pathlib import Path

from dotenv import load_dotenv
from chromadb import Client
from chromadb.config import Settings
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VECTOR_DIR = Path("db") / "chroma_db"
COLLECTION_NAME = "ethiopia_stats"  # same name your DB was already built with
MODEL = "gemini-2.5-flash"  # fast + free-tier friendly


def get_collection(vector_dir: Path):
    if not vector_dir.exists():
        raise FileNotFoundError(f"Vector database not found at {vector_dir}. Run process_dataset.py first.")
    settings = Settings(is_persistent=True, persist_directory=str(vector_dir))
    client = Client(settings)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def query_collection(collection, query: str, n_results: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return documents, metadatas


def build_prompt(query: str, documents: list, metadatas: list) -> str:
    context = []
    for doc, metadata in zip(documents, metadatas):
        source = metadata.get("source", "unknown")
        row_index = metadata.get("row_index", "unknown")
        context.append(f"Source: {source}, Row: {row_index}\n{doc}")
    return (
        "You are a helpful assistant answering questions about Ethiopian household survey statistics "
        "(ESPS-5, 2021/22). Answer the user's question using ONLY the data provided below. "
        "If the answer is not contained in the data, say you don't have that data.\n\n"
        "Context:\n" + "\n\n".join(context) + "\n\nQuestion: " + query
    )


def ask_gemini(model, query: str, documents: list, metadatas: list) -> str:
    prompt = build_prompt(query, documents, metadatas)
    response = model.generate_content(prompt)
    return response.text.strip()


def main():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found.")
        print("Add a line like this to your .env file: GEMINI_API_KEY=AIza-xxxxx")
        return

    collection = get_collection(VECTOR_DIR)
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL)

    print("Ethiopian Statistical Service Chatbot")
    print("Type a question about the dataset. Type 'exit' or 'quit' to end.\n")

    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        documents, metadatas = query_collection(collection, query)
        if not documents:
            print("No relevant results found.")
            continue

        try:
            answer = ask_gemini(model, query, documents, metadatas)
            print("\nAnswer:\n", answer)
        except Exception as e:
            print("Error calling Gemini API:", e)
            print("Showing relevant data instead:")
            for idx, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
                print(f"\n--- Result {idx} ---")
                print(f"Source: {metadata.get('source')} | Row: {metadata.get('row_index')}")
                print(doc)


if __name__ == "__main__":
    main()