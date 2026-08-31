import hashlib
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from groq import Groq
from deep_translator import GoogleTranslator
import psycopg2
import fitz  # PyMuPDF

load_dotenv()


def get_secret(key: str, default=None):
    """Reads a credential from a local .env file first (for local development),
    then falls back to Streamlit Cloud's secrets manager (Settings -> Secrets),
    since .env files are never deployed to Streamlit Cloud."""
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return default


GROQ_API_KEY = get_secret("GROQ_API_KEY")

DB_DIR = Path("chroma_db")  # pre-built locally with local_rebuild.py, committed to the repo at the root - read-only at runtime
STATS_COLLECTION = "esps_stats"
PDF_COLLECTION = "ess_pdf_docs"
STATS_FILE = Path("data set") / "preprocessed" / "aggregate_stats.csv"
MODEL = "openai/gpt-oss-20b"


st.set_page_config(page_title="ESS Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- Small embeddable widget mode -----------------------------------------
# Visiting the app with ?widget=1 in the URL shows ONLY the chatbot - no
# tabs, no login sidebar, no header - so it can be dropped into another
# website inside a small floating iframe.
WIDGET_MODE = st.query_params.get("widget") == "1"

if WIDGET_MODE:
    st.markdown(
        """
        <style>
        #MainMenu, header, footer {visibility: hidden;}
        .block-container {padding: 8px 10px 0 10px !important; max-width: 100% !important;}
        section[data-testid="stSidebar"] {display: none !important;}

        /* Match the ESS site's own deep-blue + gold branding instead of the
           purple/pink dashboard theme, so the widget looks native when
           embedded on ess.gov.et rather than mismatched. */
        .stApp {
            background: #f4f6fa !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #1a3e72, #2c5aa0) !important;
        }
        div[data-testid="stChatMessage"], .st-key-ess_input_card {
            border-color: #1a3e72 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

PG_CONFIG = {
    "dbname": get_secret("PG_DBNAME", "ethiopia_stats"),
    "user": get_secret("PG_USER", "postgres"),
    "password": get_secret("PG_PASSWORD"),
    "host": get_secret("PG_HOST", "localhost"),
    "port": get_secret("PG_PORT", "5432"),
    "sslmode": get_secret("PG_SSLMODE", "prefer"),  # hosted DBs (Neon, Supabase) require "require"
}

if not GROQ_API_KEY:
    st.sidebar.error(
        "⚠️ GROQ_API_KEY is not set. On Streamlit Cloud, add it under "
        "your app's Settings -> Secrets (not just your local .env file), "
        "or answers will silently fall back to raw, unformatted document text."
    )

RANK_WORDS = ["highest", "lowest", "most", "least", "best", "worst", "top", "compare", "rank", "maximum", "minimum"]
PDF_STRONG_WORDS = ["definition", "explain", "describe", "define", "chapter", "meaning of"]
CSV_SIGNAL_WORDS = [
    "literacy", "household size", "read and write",
    "consumption", "average age", "attended school", "illness", "female",
]


def route_question(question: str) -> str:
    q = question.lower()
    if any(w in q for w in RANK_WORDS):
        return "sql"
    if any(w in q for w in PDF_STRONG_WORDS):
        return "pdf"
    if any(w in q for w in CSV_SIGNAL_WORDS):
        return "csv"
    return "pdf"


def generate_sql_query(client: Groq, question: str) -> str:
    schema_info = (
        "Table: aggregate_stats\n"
        "Columns: group_name (Ethiopian region name, e.g. 'AMHARA', 'TIGRAY', 'OROMIA'), "
        "indicator (statistic type, one of: pct_literate, avg_household_size, avg_age, pct_female, "
        "avg_total_consumption_per_adult_equiv, avg_total_consumption_per_adult_equiv_by_area, "
        "pct_ever_attended_school, pct_illness_4wk), "
        "value (numeric), text (full readable sentence describing the stat)\n"
    )
    prompt = (
        f"{schema_info}\n"
        "Write a single PostgreSQL SELECT query to answer the question below. "
        "Reply with ONLY the raw SQL query - no explanation, no markdown code fences, no semicolon.\n\n"
        f"Question: {question}"
    )
    response = call_groq_with_backoff(
        client,
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    sql = response.choices[0].message.content.strip()
    sql = sql.strip("`").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


FORBIDDEN_SQL_KEYWORDS = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke", "--", ";"]


def is_safe_select(sql: str) -> bool:
    s = sql.lower().strip()
    if not s.startswith("select"):
        return False
    return not any(k in s for k in FORBIDDEN_SQL_KEYWORDS)


def execute_dynamic_sql(sql: str, limit: int = 10):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            colnames = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(limit)
        return colnames, rows
    finally:
        conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def ensure_tables():
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(64);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100),
                    question TEXT,
                    answer TEXT,
                    route VARCHAR(20),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS source_doc VARCHAR(255);")
            cur.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS source_page VARCHAR(50);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255),
                    indexed_at TIMESTAMP DEFAULT NOW(),
                    chunk_count INTEGER
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    id SERIAL PRIMARY KEY,
                    question_hash VARCHAR(64) UNIQUE NOT NULL,
                    question TEXT,
                    answer TEXT,
                    route VARCHAR(20),
                    source_doc VARCHAR(255),
                    source_page VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
        conn.commit()
    finally:
        conn.close()


def save_user(username: str):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username) VALUES (%s) ON CONFLICT (username) DO NOTHING",
                (username,),
            )
        conn.commit()
    finally:
        conn.close()


def get_user_password_hash(username: str):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def create_user_with_password(username: str, password: str):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) "
                "ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash "
                "WHERE users.password_hash IS NULL",
                (username, hash_password(password)),
            )
        conn.commit()
    finally:
        conn.close()


def login_user(username: str, password: str):
    """Returns (success: bool, message: str). Auto-registers brand new usernames."""
    existing_hash = get_user_password_hash(username)
    if existing_hash is None:
        save_user(username)
        create_user_with_password(username, password)
        return True, f"Account created. Logged in as {username}."
    if existing_hash == hash_password(password):
        return True, f"Logged in as {username}."
    return False, "Incorrect password for that username."


def save_chat(username: str, question: str, answer: str, route: str, source_doc: str = "", source_page: str = ""):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (username, question, answer, route, source_doc, source_page) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (username, question, answer, route, source_doc, source_page),
            )
        conn.commit()
    finally:
        conn.close()


def load_chat_history(username: str):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT question, answer, route, source_doc, source_page FROM chat_history "
                "WHERE username = %s ORDER BY created_at ASC",
                (username,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def hash_question(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()


def get_cached_answer(question: str):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT answer, route, source_doc, source_page FROM query_cache WHERE question_hash = %s",
                (hash_question(question),),
            )
            return cur.fetchone()
    finally:
        conn.close()


def save_to_cache(question: str, answer: str, route: str, source_doc: str, source_page: str):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO query_cache (question_hash, question, answer, route, source_doc, source_page) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (question_hash) DO NOTHING",
                (hash_question(question), question, answer, route, source_doc, source_page),
            )
        conn.commit()
    finally:
        conn.close()


def clear_query_cache():
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM query_cache")
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def get_admin_stats():
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chat_history")
            total = cur.fetchone()[0]
            cur.execute("SELECT route, COUNT(*) FROM chat_history GROUP BY route ORDER BY COUNT(*) DESC")
            by_route = cur.fetchall()
            cur.execute("SELECT COUNT(DISTINCT username) FROM chat_history")
            users_count = cur.fetchone()[0]
            cur.execute("SELECT username, question, route, created_at FROM chat_history ORDER BY created_at DESC LIMIT 20")
            recent = cur.fetchall()
        return total, by_route, users_count, recent
    finally:
        conn.close()


def save_document_record(filename: str, chunk_count: int):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename, chunk_count) VALUES (%s, %s)",
                (filename, chunk_count),
            )
        conn.commit()
    finally:
        conn.close()


def get_indexed_documents():
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT filename, indexed_at, chunk_count FROM documents ORDER BY indexed_at DESC")
            return cur.fetchall()
    finally:
        conn.close()


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


def extract_tables_as_text(page) -> list:
    """Detects tables on a page and turns each into clean pipe-separated rows,
    so table values become their own searchable chunks instead of being lost
    or mashed into paragraph text."""
    table_texts = []
    try:
        found = page.find_tables()
    except Exception:
        return table_texts

    for table_index, table in enumerate(found.tables):
        try:
            rows = table.extract()
        except Exception:
            continue
        if not rows:
            continue

        lines = []
        header = rows[0]
        header_line = " | ".join(str(c).strip() if c is not None else "" for c in header)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        for row in rows[1:]:
            cleaned = [str(c).strip() if c is not None else "" for c in row]
            if any(cleaned):
                lines.append(" | ".join(cleaned))

        table_text = "\n".join(lines).strip()
        if table_text and len(table_text) > 15:
            table_texts.append((table_index, table_text))

    return table_texts


def extract_pdf_chunks_from_bytes(pdf_bytes: bytes, filename: str) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        page_text = page.get_text().strip()
        if page_text and len(page_text) >= 20:
            for i, chunk in enumerate(split_pdf_text(page_text)):
                chunks.append({
                    "text": chunk, "source": filename, "page": page_number + 1,
                    "chunk_index": i, "content_type": "text",
                })

        for table_index, table_text in extract_tables_as_text(page):
            labeled = f"[Table on page {page_number + 1}]\n{table_text}"
            for i, chunk in enumerate(split_pdf_text(labeled, chunk_size=1500)):
                chunks.append({
                    "text": chunk, "source": filename, "page": page_number + 1,
                    "chunk_index": f"table{table_index}_{i}", "content_type": "table",
                })
    doc.close()
    return chunks


def index_uploaded_pdf(pdf_bytes: bytes, filename: str, client=None, collection=None) -> int:
    chunks = extract_pdf_chunks_from_bytes(pdf_bytes, filename)
    if not chunks:
        return 0

    if collection is None:
        if client is None:
            client = get_chroma_client()
        embed_fn = get_embed_fn()
        try:
            collection = client.get_collection(name=PDF_COLLECTION, embedding_function=embed_fn)
        except Exception:
            collection = client.create_collection(name=PDF_COLLECTION, embedding_function=embed_fn)

    safe_name = "".join(c if c.isalnum() else "_" for c in filename)
    ids = [f"pdf_{safe_name}_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"], "page": c["page"],
            "chunk_index": str(c["chunk_index"]), "content_type": c.get("content_type", "text"),
        }
        for c in chunks
    ]

    batch_size = 50
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])

    return len(chunks)


MULTILINGUAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # supports English + Amharic


@st.cache_resource
def get_embed_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MULTILINGUAL_MODEL_NAME
    )


@st.cache_resource
def get_chroma_client():
    """Single shared ChromaDB connection for the whole app - reads the
    pre-built database committed to the repo (built locally with
    local_rebuild.py). Read-only access works fine even though the
    filesystem itself can't be written to at runtime on Streamlit Cloud."""
    return PersistentClient(path=str(DB_DIR))


# Kept for the manual admin "Rebuild" button and PDF-upload feature only -
# NOT run automatically on startup anymore (that caused CPU throttling,
# since it reloaded the multilingual model and re-processed 4500+ PDF
# chunks on every single app restart).
PDF_SEARCH_DIRS = [Path("pdf"), Path("data") / "pdf", Path("data set") / "pdf"]


def find_all_pdfs():
    seen_names = set()
    found = []
    for d in PDF_SEARCH_DIRS:
        if not d.exists():
            continue
        for pdf_path in sorted(d.glob("*.pdf")):
            if pdf_path.name not in seen_names:
                seen_names.add(pdf_path.name)
                found.append(pdf_path)
    return found


def rebuild_all_collections_multilingual():
    """Manual admin action only - rebuilds both collections in memory for this
    session. Since Streamlit Cloud's filesystem is read-only at runtime, this
    does NOT persist across restarts; use local_rebuild.py + upload to GitHub
    for the permanent fix."""
    client = get_chroma_client()
    embed_fn = get_embed_fn()

    def delete_if_exists(name):
        try:
            existing_names = [c.name for c in client.list_collections()]
        except Exception:
            existing_names = []
        if name in existing_names:
            client.delete_collection(name=name)

    delete_if_exists(STATS_COLLECTION)
    stats_collection = client.get_or_create_collection(name=STATS_COLLECTION, embedding_function=embed_fn)

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
    stats_count = len(df)

    delete_if_exists(PDF_COLLECTION)
    pdf_collection = client.get_or_create_collection(name=PDF_COLLECTION, embedding_function=embed_fn)

    pdf_files = find_all_pdfs()
    pdf_chunk_total = 0
    pdf_files_done = []
    for pdf_path in pdf_files:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        chunk_count = index_uploaded_pdf(pdf_bytes, pdf_path.name, client=client, collection=pdf_collection)
        pdf_chunk_total += chunk_count
        pdf_files_done.append(f"{pdf_path.name} (from {pdf_path.parent}/)")

    debug_lines = [f"Working directory: {Path('.').resolve()}"]
    for d in PDF_SEARCH_DIRS:
        debug_lines.append(f"'{d}' exists: {d.exists()}" + (f", {len(list(d.glob('*.pdf')))} PDF(s)" if d.exists() else ""))
    debug_info = "\n".join(debug_lines)

    files_list = "\n".join(f"  - {name}" for name in pdf_files_done) if pdf_files_done else "(none found)"
    return (
        f"Rebuilt stats collection: {stats_count} rows.\n"
        f"Rebuilt PDF collection: {pdf_chunk_total} chunks from {len(pdf_files_done)} file(s):\n{files_list}"
        f"\n\n--- Debug info ---\n{debug_info}"
    )


def get_stats_collection():
    client = get_chroma_client()
    embed_fn = get_embed_fn()
    return client.get_collection(name=STATS_COLLECTION, embedding_function=embed_fn)


def get_pdf_collection():
    client = get_chroma_client()
    embed_fn = get_embed_fn()
    try:
        return client.get_collection(name=PDF_COLLECTION, embedding_function=embed_fn)
    except Exception:
        return None


def diagnose_chroma_db() -> str:
    """Inspects the actual chroma_db on disk and reports what's really in it -
    used to debug 'I don't have the data' issues by revealing collection
    name/embedding mismatches between local_rebuild.py and this app."""
    lines = [f"DB_DIR the app is reading from: {DB_DIR.resolve()}"]
    lines.append(f"DB_DIR exists on disk: {DB_DIR.exists()}")

    try:
        client = get_chroma_client()
    except Exception as e:
        lines.append(f"Could not open PersistentClient at that path: {e}")
        return "\n".join(lines)

    try:
        collections = client.list_collections()
    except Exception as e:
        lines.append(f"Could not list collections: {e}")
        return "\n".join(lines)

    if not collections:
        lines.append("No collections found at all in this chroma_db - it looks empty or was built at a different path/name.")
        return "\n".join(lines)

    lines.append(f"Found {len(collections)} collection(s):")
    for c in collections:
        name = c.name
        try:
            col = client.get_collection(name=name)
            count = col.count()
        except Exception as e:
            count = f"error reading count: {e}"
        expected_marker = ""
        if name == PDF_COLLECTION:
            expected_marker = "  <-- this is what the app expects for PDFs"
        elif name == STATS_COLLECTION:
            expected_marker = "  <-- this is what the app expects for CSV stats"
        lines.append(f"  - '{name}': {count} item(s){expected_marker}")

    if not any(c.name == PDF_COLLECTION for c in collections):
        lines.append(
            f"\n⚠️ No collection named '{PDF_COLLECTION}' was found. "
            "Your local_rebuild.py likely uses a different collection name for PDFs - "
            "check it uses exactly this name, or update PDF_COLLECTION in app.py to match."
        )
    if not any(c.name == STATS_COLLECTION for c in collections):
        lines.append(
            f"\n⚠️ No collection named '{STATS_COLLECTION}' was found. "
            "Your local_rebuild.py likely uses a different collection name for stats - "
            "check it uses exactly this name, or update STATS_COLLECTION in app.py to match."
        )

    return "\n".join(lines)


@st.cache_data
def load_stats_df():
    return pd.read_csv(STATS_FILE)


def query_collection(collection, query: str, n_results: int = 8):
    results = collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    return results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]


def build_prompt(query: str, documents: list, max_chars_per_doc: int = 800) -> str:
    # Truncate each retrieved chunk - Groq's free tier caps requests at 6000 tokens/minute,
    # and sending too many full-length chunks was blowing past that limit (causing 413 errors
    # that fell back to an unformatted raw dump instead of a real answer).
    trimmed = [d[:max_chars_per_doc] for d in documents]
    context = "\n".join(f"- {d}" for d in trimmed)
    return (
        "You are the official ESS (Ethiopian Statistical Service) AI Assistant, answering questions "
        "using ONLY the context provided below - never invent numbers or facts not present in the context.\n\n"
        "Guidelines for your answer:\n"
        "- If the question asks for ONE specific fact/number: answer in 1-2 short sentences, lead with "
        "the number/fact directly, no preamble like 'According to the documents'.\n"
        "- If the question asks about MULTIPLE things (e.g. 'list the reports', 'what does each dataset "
        "cover', 'tell me about X and Y'): answer with a short bullet list, one line per item, still no "
        "filler sentences before or after the list.\n"
        "- Never pad the answer or repeat the same figure twice.\n"
        "- If the exact figure requested isn't in the context but a closely related one is, say so in a "
        "short clause, then give the closely related figure and what it actually covers.\n"
        "- If truly nothing relevant is in the context, say so in one sentence.\n"
        "- Do NOT include a 'Source:' line yourself - that is added separately.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


def call_groq_with_backoff(client: Groq, **kwargs):
    """Calls Groq's chat completion with exponential backoff on rate-limit errors."""
    max_retries = 4
    delay = 1.0
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            message = str(e).lower()
            is_rate_limit = "rate" in message or "429" in message or "overloaded" in message
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise last_error


def ask_groq(client: Groq, query: str, documents: list) -> str:
    prompt = build_prompt(query, documents)
    response = call_groq_with_backoff(
        client,
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        reasoning_effort="low",
    )
    return response.choices[0].message.content.strip()


def stream_groq_answer(client: Groq, query: str, documents: list):
    """Generator that yields text chunks as they arrive, for word-by-word display."""
    prompt = build_prompt(query, documents)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def translate_to_amharic(text: str) -> str:
    return GoogleTranslator(source="en", target="am").translate(text)


def transcribe_audio(client: Groq, audio_bytes: bytes) -> str:
    transcription = client.audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model="whisper-large-v3",
    )
    return transcription.text.strip()


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Quicksand', sans-serif;
    }

    /* Soft pastel page background */
    .stApp {
        background: linear-gradient(180deg, #f5f7ff 0%, #fdf6f9 100%);
    }

    /* Cute bot header */
    .ess-bot-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .ess-bot-avatar {
        font-size: 42px;
        background: linear-gradient(135deg, #a78bfa, #f9a8d4);
        border-radius: 50%;
        width: 64px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 10px rgba(167,139,250,0.35);
    }
    .ess-bot-title {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .ess-bot-subtitle {
        font-size: 14px;
        color: #7c7c8a;
        margin-top: -4px;
    }

    /* Rounded, softer buttons and inputs everywhere */
    .stButton>button, .stTextInput>div>div>input, .stCheckbox, div[data-baseweb="input"] {
        border-radius: 14px !important;
    }
    .stButton>button {
        border: none;
        background: linear-gradient(90deg, #a78bfa, #f472b6);
        color: white;
        font-weight: 600;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        color: white;
    }

    /* Cute sidebar login card */
    .ess-login-card {
        background: white;
        border-radius: 18px;
        padding: 18px 16px 8px 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .ess-login-title {
        font-size: 18px;
        font-weight: 700;
        color: #7c3aed;
        margin-bottom: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if WIDGET_MODE:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <div style="font-size:22px;">🤖</div>
            <div style="font-weight:700; color:#1a3e72; font-size:16px;">ESS AI Buddy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="ess-bot-header">
            <div class="ess-bot-avatar">🤖</div>
            <div>
                <p class="ess-bot-title">ESS AI Buddy</p>
                <p class="ess-bot-subtitle">Ethiopia Statistical Service · your friendly stats assistant</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def ensure_tables_once():
    """Runs the CREATE TABLE checks only once for the whole app's lifetime,
    instead of on every rerun (Streamlit reruns this script on every click,
    so without caching this was hitting Postgres on every single interaction)."""
    ensure_tables()
    return True


try:
    ensure_tables_once()
    DB_READY = True
except Exception as e:
    DB_READY = False
    st.sidebar.warning(f"Database features unavailable: {e}")

if not WIDGET_MODE:
    with st.sidebar:
        st.markdown(
            """
            <div class="ess-login-card">
                <div class="ess-login-title">🔐 Login</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "username" not in st.session_state:
            st.session_state.username = ""
        if "history_loaded_for" not in st.session_state:
            st.session_state.history_loaded_for = None

        username_input = st.text_input("👤 Enter your name", value=st.session_state.username)
        password_input = st.text_input("🔑 Password", type="password")
        st.caption("New username? Just pick a password - it creates your account automatically.")

        if st.button("✨ Set username") and username_input.strip():
            if not password_input:
                st.sidebar.error("Please also enter a password.")
            elif not DB_READY:
                st.sidebar.error("Cannot log in - database unavailable.")
            else:
                ok, message = login_user(username_input.strip(), password_input)
                if ok:
                    st.session_state.username = username_input.strip()
                    st.sidebar.success(message)
                else:
                    st.sidebar.error(message)
else:
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "history_loaded_for" not in st.session_state:
        st.session_state.history_loaded_for = None

current_user = st.session_state.get("username") or "anonymous"

if "messages" not in st.session_state:
    st.session_state.messages = []

if DB_READY and current_user != "anonymous" and st.session_state.history_loaded_for != current_user:
    try:
        past = load_chat_history(current_user)
        st.session_state.messages = list(past)
        st.session_state.history_loaded_for = current_user
    except Exception as e:
        st.sidebar.warning(f"Could not load chat history: {e}")


def process_question(question: str, amharic_mode: bool):
    """Runs the full routing + retrieval + answer pipeline for one question.
    Returns (answer, route, source_doc, source_page)."""
    cached = get_cached_answer(question) if DB_READY else None

    if cached:
        cached_answer, cached_route, cached_source_doc, cached_source_page = cached
        route = cached_route
        source_doc, source_page = cached_source_doc or "", cached_source_page or ""
        answer = f"{cached_answer}\n\n_⚡ instant repeat answer from cache_"
        return answer, route, source_doc, source_page

    route = route_question(question)
    source_doc, source_page = "", ""

    if route == "sql":
        if not GROQ_API_KEY:
            answer = "This type of question needs the Groq API key to generate a query. Please add GROQ_API_KEY to .env."
        else:
            sql_client = Groq(api_key=GROQ_API_KEY)
            try:
                generated_sql = generate_sql_query(sql_client, question)
                if not is_safe_select(generated_sql):
                    answer = f"I couldn't safely run a query for that question. Try rephrasing.\n\n(Generated: {generated_sql})"
                else:
                    colnames, rows = execute_dynamic_sql(generated_sql)
                    if not rows:
                        answer = "No results found for that question."
                    else:
                        lines = [", ".join(f"{c}: {v}" for c, v in zip(colnames, row)) for row in rows]
                        answer = "\n".join(lines)
                    source_doc = "PostgreSQL - ESPS-5 aggregate_stats table (AI-generated SQL)"
                    answer += f"\n\n_Query used: `{generated_sql}`_"
            except Exception as e:
                answer = f"Error running dynamic SQL query: {e}"

    else:
        if route == "csv":
            collection = get_stats_collection()
            documents, metadatas = query_collection(collection, question)
            source_doc = "ESPS-5 Socioeconomic Survey, 2021/22"
            source_page = ""
        else:
            pdf_collection = get_pdf_collection()
            if pdf_collection is None:
                documents, metadatas = [], []
                source_doc = "ESS PDF report"
                source_page = ""
            else:
                documents, metadatas = query_collection(pdf_collection, question, n_results=12)
                safe_metas = [m for m in metadatas if isinstance(m, dict)]
                sources = sorted({m.get("source") for m in safe_metas if m.get("source")})
                pages = sorted({m.get("page") for m in safe_metas if m.get("page") is not None})
                source_doc = ", ".join(sources) if sources else "ESS PDF report"
                source_page = ", ".join(str(p) for p in pages[:3]) if pages else ""

        if not documents:
            if route == "pdf":
                answer = "No PDF has been indexed yet, so I can't answer document-based questions. Ask a statistics question instead (e.g. 'What is the literacy rate in Amhara?'), or index a PDF first."
            else:
                answer = "No relevant statistics found for that question."
        elif GROQ_API_KEY:
            client = Groq(api_key=GROQ_API_KEY)
            try:
                answer = ask_groq(client, question, documents)
                if amharic_mode:
                    amharic_answer = translate_to_amharic(answer)
                    answer += f"\n\n🇪🇹 **አማርኛ:** {amharic_answer}"
            except Exception as e:
                error_text = str(e).lower()
                too_large = "413" in error_text or "too large" in error_text or "rate_limit_exceeded" in error_text
                if too_large:
                    try:
                        small_prompt = build_prompt(question, documents[:2], max_chars_per_doc=200)
                        response = call_groq_with_backoff(
                            client,
                            model=MODEL,
                            messages=[{"role": "user", "content": small_prompt}],
                            max_tokens=200,
                        )
                        answer = response.choices[0].message.content.strip()
                        if amharic_mode:
                            amharic_answer = translate_to_amharic(answer)
                            answer += f"\n\n🇪🇹 **አማርኛ:** {amharic_answer}"
                    except Exception:
                        answer = (
                            "⚠️ That question needed more data than the AI service allows right now "
                            "(free-tier rate limit). Try asking something more specific, or try again "
                            "in a minute."
                        )
                else:
                    answer = "⚠️ Something went wrong generating an answer. Please try rephrasing your question."
        else:
            answer = (
                "⚠️ I found relevant data, but can't summarize it right now because "
                "GROQ_API_KEY isn't configured for this app - ask the site owner to set it "
                "under Settings -> Secrets on Streamlit Cloud."
            )

    if DB_READY:
        try:
            save_to_cache(question, answer, route, source_doc, source_page)
        except Exception:
            pass

    return answer, route, source_doc, source_page


def render_chatbot():
    st.header("Ask the Chatbot")

    if "prefill" not in st.session_state:
        st.session_state.prefill = ""
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None
    if "chat_input_key" not in st.session_state:
        st.session_state.chat_input_key = 0

    ROUTE_ICONS = {"pdf": "📘", "csv": "📊", "sql": "🧮"}
    ROUTE_LABELS = {"pdf": "DHS Final Report", "csv": "ESPS-5 Survey Data", "sql": "Database Query"}
    ROUTE_COLORS = {"pdf": "#1b3a2b", "csv": "#1b2a3a", "sql": "#3a2f1b"}

    for q, a, route, s_doc, s_page in st.session_state.messages:
        st.markdown(f"**You:** {q}")
        icon = ROUTE_ICONS.get(route, "💬")
        label = ROUTE_LABELS.get(route, route)
        color = ROUTE_COLORS.get(route, "#222222")
        source_line = ""
        if s_doc:
            page_part = f", page(s) {s_page}" if s_page else ""
            source_line = f"<div style='margin-top:10px; font-size:0.85em; opacity:0.75;'>📌 Source: {s_doc}{page_part}</div>"
        answer_html = a.replace("\n", "<br>")
        st.markdown(
            f"""
            <div style="background-color:{color}; color:#e6e6e6; padding:16px 18px;
                        border-radius:10px; margin:8px 0 18px 0; line-height:1.5;">
                <div style="font-weight:600; margin-bottom:6px;">{icon} {label}</div>
                <div>{answer_html}</div>
                {source_line}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.messages:
        chat_text = "\n\n".join(
            f"You: {q}\n(routed to: {route})\nChatbot: {a}"
            for q, a, route, s_doc, s_page in st.session_state.messages
        )
        st.download_button("Download chat as text", chat_text, file_name="chat_history.txt")

    st.markdown(
        """
        <style>
        /* Vertically + horizontally centers the input card when there's no chat yet */
        .st-key-ess_center_wrapper {
            min-height: 52vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* The rounded card itself, Claude/Gemini style */
        .st-key-ess_input_card {
            width: 100%;
            max-width: 700px;
            margin: 0 auto;
            background: white;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 24px;
            padding: 18px 22px 10px 22px;
            box-shadow: 0 6px 24px rgba(124,58,237,0.10);
        }
        /* Make the text input blend into the card - no visible border of its own */
        .st-key-ess_input_card input[type="text"] {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            font-size: 16px;
            padding-left: 2px !important;
        }
        /* Small icon-style controls row inside the card */
        .ess-card-icons .stCheckbox label p {
            font-size: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    chat_is_empty = len(st.session_state.messages) == 0
    outer = st.container(key="ess_center_wrapper") if chat_is_empty else st.container()

    with outer:
        with st.container(key="ess_input_card"):
            typed_question = st.text_input(
                "Ask",
                placeholder="Ask ESS AI Assistant...",
                label_visibility="collapsed",
                key=f"chat_text_{st.session_state.chat_input_key}",
            )
            icon_row = st.container()
            icon_row.markdown('<div class="ess-card-icons">', unsafe_allow_html=True)
            with icon_row:
                c_amharic, c_mic, c_spacer, c_send = st.columns([1, 1, 4, 1.3])
                with c_amharic:
                    amharic_mode = st.checkbox("🇪🇹", value=False, help="Also answer in Amharic")
                with c_mic:
                    audio_value = None
                    with st.popover("🎤"):
                        st.caption("Record a question")
                        audio_value = st.audio_input("Recorder", label_visibility="collapsed")
                with c_send:
                    send_clicked = st.button("Send ➤", use_container_width=True)
            icon_row.markdown('</div>', unsafe_allow_html=True)

    voice_question = None
    if audio_value is not None and GROQ_API_KEY:
        audio_bytes = audio_value.read()
        audio_id = hashlib.md5(audio_bytes).hexdigest()
        if st.session_state.last_audio_id != audio_id:
            st.session_state.last_audio_id = audio_id
            voice_client = Groq(api_key=GROQ_API_KEY)
            try:
                transcribed = transcribe_audio(voice_client, audio_bytes)
                if transcribed:
                    st.info(f"🎤 Heard: \"{transcribed}\"")
                    voice_question = transcribed
            except Exception as e:
                st.warning(f"Could not transcribe audio: {e}")

    final_question = None
    if send_clicked and typed_question.strip():
        final_question = typed_question.strip()
    elif voice_question:
        final_question = voice_question

    if final_question and final_question.strip():
        final_question = final_question.strip()
        answer, route, source_doc, source_page = process_question(final_question, amharic_mode)
        st.session_state.messages.append((final_question, answer, route, source_doc, source_page))
        if DB_READY:
            try:
                save_chat(current_user, final_question, answer, route, source_doc, source_page)
            except Exception as e:
                st.warning(f"Could not save chat history: {e}")
        st.session_state.chat_input_key += 1  # resets the text box for the next question
        st.rerun()


if WIDGET_MODE:
    render_chatbot()
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Chatbot", "Data Explorer", "Visualizations", "Admin"])

    with tab1:
        render_chatbot()

    with tab2:
        st.header("Data Explorer")
        df = load_stats_df()
        region_filter = st.selectbox("Filter by region (optional)", ["All"] + sorted(df["group"].unique().tolist()))
        if region_filter != "All":
            st.dataframe(df[df["group"] == region_filter])
        else:
            st.dataframe(df)

    with tab3:
        st.header("Visualizations")
        df = load_stats_df()

        st.subheader("All regions")
        indicator = st.selectbox("Choose an indicator", sorted(df["indicator"].unique()))
        chart_df = df[df["indicator"] == indicator][["group", "value"]].set_index("group")
        st.bar_chart(chart_df)

        st.divider()
        st.subheader("Compare two regions")
        regions = sorted(df["group"].unique().tolist())
        col_a, col_b = st.columns(2)
        region_a = col_a.selectbox("Region A", regions, index=0)
        region_b = col_b.selectbox("Region B", regions, index=min(1, len(regions) - 1))

        compare_indicator = st.selectbox("Indicator to compare", sorted(df["indicator"].unique()), key="compare_indicator")
        row_a = df[(df["group"] == region_a) & (df["indicator"] == compare_indicator)]
        row_b = df[(df["group"] == region_b) & (df["indicator"] == compare_indicator)]

        if not row_a.empty and not row_b.empty:
            val_a = row_a["value"].iloc[0]
            val_b = row_b["value"].iloc[0]
            compare_df = pd.DataFrame({"value": [val_a, val_b]}, index=[region_a, region_b])
            st.bar_chart(compare_df)
            diff = val_a - val_b
            st.caption(f"{region_a} vs {region_b}: difference of {diff:.2f}")
        else:
            st.info("No data available for one of the selected regions for this indicator.")

    with tab4:
        st.header("Admin / Usage Stats")

        st.caption("🔧 Search index storage: pre-built locally, read from the committed 'chroma_db' folder at the repo root")
        st.caption(f"🔧 PDF folders checked by manual rebuild: {[str(p) for p in PDF_SEARCH_DIRS]}")

        st.subheader("🔍 Diagnose 'I don't have the data' issues")
        st.caption(
            "Run this first if questions keep saying there's no data. It shows exactly what's inside "
            "chroma_db right now - collection names and item counts - so we can see if it matches what "
            "the app expects, or if local_rebuild.py used different names."
        )
        if st.button("🔍 Check chroma_db contents"):
            with st.spinner("Inspecting chroma_db..."):
                try:
                    st.code(diagnose_chroma_db())
                except Exception as e:
                    st.error(f"Diagnostic failed: {e}")

        st.divider()
        st.subheader("Rebuild search index (this session only)")
        st.caption(
            "⚠️ This only rebuilds the index in memory for THIS running session - it does NOT save "
            "permanently, since Streamlit Cloud's filesystem is read-only. For a permanent update, run "
            "local_rebuild.py on your own computer and upload the resulting chroma_db folder to GitHub."
        )
        if st.button("🔄 Rebuild search index (this session only)"):
            with st.spinner("Rebuilding... this can take a few minutes."):
                try:
                    summary = rebuild_all_collections_multilingual()
                    st.success("Rebuild complete for this session!")
                    st.text(summary)
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

        st.divider()
        st.subheader("Clear cached answers")
        st.caption(
            "If you just rebuilt the search index or uploaded a new PDF, old cached answers "
            "may still show up for questions asked before the update. Clear the cache to force "
            "fresh answers for every question."
        )
        if st.button("🗑️ Clear all cached answers"):
            try:
                deleted = clear_query_cache()
                st.success(f"Cleared {deleted} cached answer(s).")
            except Exception as e:
                st.error(f"Could not clear cache: {e}")

        st.divider()
        st.subheader("Upload a new PDF report (this session only)")
        st.caption("⚠️ Same limitation as above - won't persist across restarts unless you also rebuild locally and re-upload the database folder.")
        uploaded_pdf = st.file_uploader("Choose a PDF file to index", type="pdf")
        if uploaded_pdf is not None and st.button("Index this PDF"):
            with st.spinner("Extracting and indexing... this may take a few minutes for large PDFs"):
                try:
                    pdf_bytes = uploaded_pdf.read()
                    chunk_count = index_uploaded_pdf(pdf_bytes, uploaded_pdf.name)
                    if DB_READY:
                        save_document_record(uploaded_pdf.name, chunk_count)
                    st.success(f"Indexed '{uploaded_pdf.name}': {chunk_count} chunks added to the PDF search database.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

        st.divider()

        if not DB_READY:
            st.warning("Database not connected - admin stats unavailable.")
        else:
            try:
                total, by_route, users_count, recent = get_admin_stats()
                col1, col2 = st.columns(2)
                col1.metric("Total questions asked", total)
                col2.metric("Unique users", users_count)

                if by_route:
                    st.subheader("Questions by route")
                    route_df = pd.DataFrame(by_route, columns=["route", "count"]).set_index("route")
                    st.bar_chart(route_df)

                try:
                    docs = get_indexed_documents()
                    if docs:
                        st.subheader("Indexed documents")
                        docs_df = pd.DataFrame(docs, columns=["filename", "indexed_at", "chunk_count"])
                        st.dataframe(docs_df)
                except Exception as e:
                    st.warning(f"Could not load documents list: {e}")

                if recent:
                    st.subheader("Recent activity")
                    recent_df = pd.DataFrame(recent, columns=["username", "question", "route", "time"])
                    st.dataframe(recent_df)
                else:
                    st.info("No questions asked yet.")
            except Exception as e:
                st.error(f"Could not load admin stats: {e}")
