import hashlib
import tempfile
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_DIR = Path(tempfile.gettempdir()) / "ess_chroma_db"  # always-writable location; rebuilt fresh on every app start anyway
STATS_COLLECTION = "esps_stats"
PDF_COLLECTION = "ess_pdf_docs"
STATS_FILE = Path("data set") / "preprocessed" / "aggregate_stats.csv"
MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="ESS Dashboard", layout="wide")

PG_CONFIG = {
    "dbname": os.getenv("PG_DBNAME", "ethiopia_stats"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD"),
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "sslmode": os.getenv("PG_SSLMODE", "prefer"),  # hosted DBs (Neon, Supabase) require "require"
}

RANK_WORDS = ["highest", "lowest", "most", "least", "best", "worst", "top", "compare", "rank", "maximum", "minimum"]
PDF_STRONG_WORDS = ["definition", "explain", "describe", "define", "chapter", "meaning of"]
# Only words tied to what aggregate_stats.csv actually contains (literacy, household size,
# age, female %, consumption, school attendance, illness). Removed overly generic words like
# "rate", "total", "population", "inflation" which caused unrelated PDF-only topics (like
# inflation reports) to get wrongly routed to the CSV path instead of PDF search.
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
            # Added: password support on top of the existing users table
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
            # Added: separate source columns on top of the existing chat_history table
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

            # Added: query_cache table (new, does not touch anything existing)
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
        save_user(username)  # keeps existing save_user behavior intact
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


def extract_pdf_chunks_from_bytes(pdf_bytes: bytes, filename: str) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    for page_number in range(len(doc)):
        page_text = doc[page_number].get_text().strip()
        if not page_text or len(page_text) < 20:
            continue
        for i, chunk in enumerate(split_pdf_text(page_text)):
            chunks.append({"text": chunk, "source": filename, "page": page_number + 1, "chunk_index": i})
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
    metadatas = [{"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]} for c in chunks]

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
    """Single shared ChromaDB connection for the whole app - avoids the
    'Collection does not exist' errors caused by opening multiple separate
    PersistentClient connections to the same database in one process."""
    return PersistentClient(path=str(DB_DIR))


def rebuild_all_collections_multilingual():
    """
    Rebuilds BOTH ChromaDB collections (stats + PDFs) using the multilingual
    embedding function, directly on the server, from files already in the repo.
    Returns a summary string.
    """
    client = get_chroma_client()
    embed_fn = get_embed_fn()

    def delete_if_exists(name):
        try:
            existing_names = [c.name for c in client.list_collections()]
        except Exception:
            existing_names = []
        if name in existing_names:
            client.delete_collection(name=name)

    # --- Rebuild stats collection from the CSV ---
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

    # --- Rebuild PDF collection from whatever PDFs are already in the repo ---
    delete_if_exists(PDF_COLLECTION)
    pdf_collection = client.get_or_create_collection(name=PDF_COLLECTION, embedding_function=embed_fn)

    pdf_dir = Path("pdf")
    pdf_chunk_total = 0
    pdf_files_done = []
    if pdf_dir.exists():
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            chunk_count = index_uploaded_pdf(pdf_bytes, pdf_path.name, client=client, collection=pdf_collection)
            pdf_chunk_total += chunk_count
            pdf_files_done.append(pdf_path.name)

    # Diagnostic info in case the PDF still isn't found, so we don't have to guess blindly
    debug_lines = [f"Working directory: {Path('.').resolve()}"]
    try:
        top_level = sorted(p.name + ("/" if p.is_dir() else "") for p in Path(".").iterdir())
        debug_lines.append(f"Top-level contents: {top_level}")
    except Exception as e:
        debug_lines.append(f"Could not list top level: {e}")
    for candidate in ["pdf", "data set/pdf", "data set", "data"]:
        cpath = Path(candidate)
        debug_lines.append(f"'{candidate}' exists: {cpath.exists()}" + (f", contents: {list(cpath.glob('*'))}" if cpath.exists() else ""))
    debug_info = "\n".join(debug_lines)

    return (
        f"Rebuilt stats collection: {stats_count} rows.\n"
        f"Rebuilt PDF collection: {pdf_chunk_total} chunks from {len(pdf_files_done)} file(s): "
        f"{', '.join(pdf_files_done) if pdf_files_done else '(none found)'}.\n\n"
        f"--- Debug info ---\n{debug_info}"
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


@st.cache_data
def load_stats_df():
    return pd.read_csv(STATS_FILE)


def query_collection(collection, query: str, n_results: int = 5):
    results = collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    return results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]


def build_prompt(query: str, documents: list) -> str:
    context = "\n".join(f"- {d}" for d in documents)
    return (
        "You are a helpful assistant answering questions about Ethiopian statistics. "
        "Answer using ONLY the context below. If the answer isn't covered, say you don't have that data.\n\n"
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
        max_tokens=512,
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


st.title("Ethiopia Statistical Service Dashboard")

try:
    ensure_tables()
    DB_READY = True
except Exception as e:
    DB_READY = False
    st.sidebar.warning(f"Database features unavailable: {e}")

# Auto-heal: on every fresh app start (including waking up from sleep), check
# whether the PDF search index is actually populated. If it's missing or empty
# (e.g. because the server restarted and lost the previous session's index),
# rebuild it automatically so the app doesn't need a manual admin click.
if "startup_check_done" not in st.session_state:
    st.session_state.startup_check_done = False

if not st.session_state.startup_check_done:
    st.session_state.startup_check_done = True
    try:
        _client = get_chroma_client()
        _embed_fn = get_embed_fn()
        _needs_rebuild = False
        try:
            _pdf_coll = _client.get_collection(name=PDF_COLLECTION, embedding_function=_embed_fn)
            if _pdf_coll.count() == 0:
                _needs_rebuild = True
        except Exception:
            _needs_rebuild = True

        if _needs_rebuild:
            with st.spinner("First load after waking up: rebuilding the search index, please wait (one-time, takes a couple of minutes)..."):
                rebuild_all_collections_multilingual()
    except Exception as _e:
        st.sidebar.warning(f"Auto-rebuild check skipped: {_e}")

with st.sidebar:
    st.header("Login")
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "history_loaded_for" not in st.session_state:
        st.session_state.history_loaded_for = None

    username_input = st.text_input("Enter your name", value=st.session_state.username)
    password_input = st.text_input("Password", type="password")
    st.caption("New username? Just pick a password - it creates your account automatically.")

    if st.button("Set username") and username_input.strip():
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

current_user = st.session_state.get("username") or "anonymous"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Reload past chat history once per login, so it doesn't wipe out new messages on every rerun
if DB_READY and current_user != "anonymous" and st.session_state.history_loaded_for != current_user:
    try:
        past = load_chat_history(current_user)
        st.session_state.messages = list(past)
        st.session_state.history_loaded_for = current_user
    except Exception as e:
        st.sidebar.warning(f"Could not load chat history: {e}")

tab1, tab2, tab3, tab4 = st.tabs(["Chatbot", "Data Explorer", "Visualizations", "Admin"])

with tab1:
    st.header("Ask the Chatbot")

    if "prefill" not in st.session_state:
        st.session_state.prefill = ""

    question = st.text_input("Type your question:", value=st.session_state.prefill)

    audio_value = st.audio_input("Or record your question")
    if audio_value is not None and GROQ_API_KEY:
        voice_client = Groq(api_key=GROQ_API_KEY)
        try:
            transcribed = transcribe_audio(voice_client, audio_value.read())
            if transcribed:
                st.session_state.prefill = transcribed
                st.info(f"Heard: \"{transcribed}\" — click Ask to submit, or edit the text above first.")
        except Exception as e:
            st.warning(f"Could not transcribe audio: {e}")

    amharic_mode = st.checkbox("Also answer in Amharic")
    ask_clicked = st.button("Ask")

    if ask_clicked and question.strip():
        cached = get_cached_answer(question) if DB_READY else None

        if cached:
            cached_answer, cached_route, cached_source_doc, cached_source_page = cached
            route = cached_route
            source_doc, source_page = cached_source_doc or "", cached_source_page or ""
            answer = f"{cached_answer}\n\n_⚡ instant repeat answer from cache_"

        else:
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
                            answer += f"\n\n[Source: {source_doc}]"
                            answer += f"\n\n_Query used: `{generated_sql}`_"
                    except Exception as e:
                        answer = f"Error running dynamic SQL query: {e}"

            else:
                if route == "csv":
                    collection = get_stats_collection()
                    documents, metadatas = query_collection(collection, question)
                    source_doc = "ESPS-5 Socioeconomic Survey, 2021/22"
                    source_page = ""
                    source_note = f"[Source: {source_doc}]"
                else:
                    pdf_collection = get_pdf_collection()
                    if pdf_collection is None:
                        documents, metadatas = [], []
                        source_doc = "ESS PDF report"
                        source_page = ""
                        source_note = f"[Source: {source_doc}]"
                    else:
                        documents, metadatas = query_collection(pdf_collection, question)
                        safe_metas = [m for m in metadatas if isinstance(m, dict)]
                        sources = sorted({m.get("source") for m in safe_metas if m.get("source")})
                        pages = sorted({m.get("page") for m in safe_metas if m.get("page") is not None})
                        source_doc = ", ".join(sources) if sources else "ESS PDF report"
                        source_page = ", ".join(str(p) for p in pages[:3]) if pages else ""
                        if source_page:
                            source_note = f"[Source: {source_doc}, page(s) {source_page}]"
                        else:
                            source_note = f"[Source: {source_doc}]"

                if not documents:
                    if route == "pdf":
                        answer = "No PDF has been indexed yet, so I can't answer document-based questions. Ask a statistics question instead (e.g. 'What is the literacy rate in Amhara?'), or index a PDF first."
                    else:
                        answer = "No relevant statistics found for that question."
                elif GROQ_API_KEY:
                    client = Groq(api_key=GROQ_API_KEY)
                    try:
                        if amharic_mode:
                            answer = ask_groq(client, question, documents)
                            amharic_answer = translate_to_amharic(answer)
                            answer += f"\n\n🇪🇹 **አማርኛ:** {amharic_answer}"
                        else:
                            st.markdown("**Chatbot (streaming):**")
                            answer = st.write_stream(stream_groq_answer(client, question, documents))
                        answer += f"\n\n{source_note}"
                    except Exception as e:
                        answer = f"Error calling Groq API: {e}\n\nRaw matches:\n" + "\n".join(f"- {d}" for d in documents)
                else:
                    answer = documents[0] + f"\n\n{source_note}"

            if DB_READY:
                try:
                    save_to_cache(question, answer, route, source_doc, source_page)
                except Exception:
                    pass

        st.session_state.messages.append((question, answer, route, source_doc, source_page))
        st.session_state.prefill = ""
        if DB_READY:
            try:
                save_chat(current_user, question, answer, route, source_doc, source_page)
            except Exception as e:
                st.warning(f"Could not save chat history: {e}")

    if st.session_state.messages:
        chat_text = "\n\n".join(
            f"You: {q}\n(routed to: {route})\nChatbot: {a}"
            for q, a, route, s_doc, s_page in st.session_state.messages
        )
        st.download_button("Download chat as text", chat_text, file_name="chat_history.txt")

    ROUTE_ICONS = {"pdf": "📘", "csv": "📊", "sql": "🧮"}
    ROUTE_LABELS = {"pdf": "DHS Final Report", "csv": "ESPS-5 Survey Data", "sql": "Database Query"}
    ROUTE_COLORS = {"pdf": "#1b3a2b", "csv": "#1b2a3a", "sql": "#3a2f1b"}

    for q, a, route, s_doc, s_page in reversed(st.session_state.messages):
        st.markdown(f"**You:** {q}")

        icon = ROUTE_ICONS.get(route, "💬")
        label = ROUTE_LABELS.get(route, route)
        color = ROUTE_COLORS.get(route, "#222222")
        source_line = ""
        if s_doc:
            page_part = f", page(s) {s_page}" if s_page else ""
            source_line = f"<div style='margin-top:10px; font-size:0.85em; opacity:0.75;'>Source: {s_doc}{page_part}</div>"

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
        st.divider()

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

    st.subheader("Rebuild search index for Amharic support")
    st.caption(
        "One-time step: rebuilds the search database using a multilingual model, "
        "so questions typed in Amharic can be matched correctly (not just English). "
        "Uses the CSV and PDF files already in this repo — no upload needed."
    )
    if st.button("🔄 Rebuild search index (multilingual)"):
        with st.spinner("Rebuilding... this can take a few minutes (downloads the multilingual model on first run)."):
            try:
                summary = rebuild_all_collections_multilingual()
                st.success("Rebuild complete!")
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
    st.subheader("Upload a new PDF report")
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
