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


st.set_page_config(page_title="ESS Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling for Image Layout ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Quicksand', sans-serif;
    }

    .stApp {
        background: #f8f8fd;
    }

    /* Header styling */
    .ess-bot-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 20px;
    }
    .ess-bot-avatar {
        font-size: 32px;
        background: linear-gradient(135deg, #a78bfa, #f9a8d4);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 10px rgba(167,139,250,0.35);
    }
    .ess-bot-title {
        font-size: 22px;
        font-weight: 700;
        color: #7c3aed;
        margin: 0;
    }
    .ess-bot-subtitle {
        font-size: 13px;
        color: #7c7c8a;
        margin-top: -2px;
    }

    /* Thin Question Box */
div[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 2px solid #222222 !important;
    background-color: #ffffff !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05) !important;
}

div[data-testid="stChatInput"] textarea {
    height: 42px !important;
    font-size: 14px !important;
}

    /* Floating container for bottom-right buttons */
    .floating-button-wrapper {
        position: fixed;
        bottom: 25px;
        right: 25px;
        z-index: 999999;
        display: flex;
        flex-direction: row;
        gap: 8px;
    }

    .custom-icon-btn {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        border: 1px solid #222222;
        background-color: #ffffff;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        cursor: pointer;
        text-decoration: none;
        color: #000000;
    }

    .custom-icon-btn:hover {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# --- Small embeddable widget mode -----------------------------------------
WIDGET_MODE = st.query_params.get("widget") == "1"

if WIDGET_MODE:
    st.markdown(
        """
        <style>
        #MainMenu, header, footer {visibility: hidden;}
        .block-container {padding: 8px 10px 0 10px !important; max-width: 100% !important;}
        section[data-testid="stSidebar"] {display: none !important;}
        .stApp { background: #f4f6fa !important; }
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
    "sslmode": get_secret("PG_SSLMODE", "prefer"),
}

if not GROQ_API_KEY:
    st.sidebar.error(
        "⚠️ GROQ_API_KEY is not set. On Streamlit Cloud, add it under "
        "your app's Settings -> Secrets (not just your local .env file)."
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


MULTILINGUAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_resource
def get_embed_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MULTILINGUAL_MODEL_NAME
    )


@st.cache_resource
def get_chroma_client():
    return PersistentClient(path=str(DB_DIR))


def get_pdf_collection():
    client = get_chroma_client()
    embed_fn = get_embed_fn()
    try:
        return client.get_collection(name=PDF_COLLECTION, embedding_function=embed_fn)
    except Exception:
        return None


def build_prompt(query: str, documents: list, max_chars_per_doc: int = 1000) -> str:
    trimmed = [d[:max_chars_per_doc] for d in documents]
    context = "\n".join(f"- {d}" for d in trimmed)
    return (
        "You are the official ESS (Ethiopian Statistical Service) AI Assistant, answering questions "
        "using ONLY the context provided below.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


def call_groq_with_backoff(client: Groq, **kwargs):
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
    )
    return response.choices[0].message.content.strip()


def translate_to_amharic(text: str) -> str:
    return GoogleTranslator(source="en", target="am").translate(text)


@st.cache_resource
def ensure_tables_once():
    ensure_tables()
    return True


try:
    ensure_tables_once()
    DB_READY = True
except Exception as e:
    DB_READY = False
    st.sidebar.warning(f"Database features unavailable: {e}")

# --- Sidebar Login preserved ---
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

        if st.button("Login"):
            if username_input and password_input:
                ok, msg = login_user(username_input, password_input)
                if ok:
                    st.session_state.username = username_input
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("Please enter username and password.")

# --- Header Section ---
if not WIDGET_MODE:
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

# --- Thin Centered Question Box ---
left_pad, center_col, right_pad = st.columns([1, 2, 1])

# --- Question Box Section ---
left_pad, center_col, right_pad = st.columns([1, 2, 1])

with center_col:
    # Native chat input places the submit arrow icon directly inside the input box
    user_query = st.chat_input("Ask ESS AI Assistant...")

    if user_query:
        # Check cache first for instant responses
        cached_res = get_cached_answer(user_query)
        if cached_res:
            answer, route, src_doc, src_page = cached_res
            st.markdown(f"**Answer:** {answer}")
        else:
            client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
            pdf_col = get_pdf_collection()
            
            # Retrieve top 5 matches to include both text and table chunks
            docs = []
            if pdf_col:
                res = pdf_col.query(query_texts=[user_query], n_results=5)
                docs = res.get("documents", [[]])[0]

            if client and docs:
                # Fast model response generation
                answer = ask_groq(client, user_query, docs)
            else:
                answer = "I couldn't locate specific information on that in the available documents or tables."

            st.markdown(f"**Answer:** {answer}")
            
            # Save to PostgreSQL and cache for instant future loads
            save_chat(st.session_state.get("username", "guest"), user_query, answer, "pdf")
            save_to_cache(user_query, answer, "pdf", "", "")
# --- Bottom Floating Buttons ---
st.markdown(
    """
    <div class="floating-button-wrapper">
        <button class="custom-icon-btn" onclick="alert('Mic clicked')">🎙️</button>
        <button class="custom-icon-btn" onclick="alert('Amharic clicked')">አ</button>
    </div>
    """,
    unsafe_allow_html=True,
)
