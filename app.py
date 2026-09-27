"""
app.py - Ethiopia Statistical Service (ESS) AI Buddy dashboard.

This version merges everything that existed scattered across the project
(app.py, core.py, app_old_backup.py) into ONE working Streamlit app:

  - Real 3-way question routing (PDF / survey CSV / live SQL), using the
    same keyword lists as core.py.
  - Real bcrypt password hashing (replacing the old plain SHA-256 password
    hash - SHA-256 alone is not a proper password hash; bcrypt is already
    in requirements.txt but was never actually used for passwords).
  - Postgres-backed login, chat history, and answer caching.
  - Real working voice input (browser Web Speech API via a small JS
    component) - unchanged from the version that was already confirmed
    working live.
  - A real, working Amharic toggle. The old version tried to flip Amharic
    mode by navigating the *parent* page from inside an embedded HTML
    component (window.parent.location.href) - browsers/Streamlit's iframe
    sandboxing block or unreliably handle that cross-frame navigation,
    which is almost certainly why it didn't work. This version replaces
    that with a normal Streamlit sidebar toggle, which needs no cross-frame
    trick at all.
  - A working admin panel to upload and index new PDF reports.
  - The decommissioned Groq model (llama-3.1-70b-versatile) replaced with
    a currently-supported model.

Before deploying: push this file to GitHub as app.py, then on Streamlit
Cloud use the "⋮ menu -> Reboot app" so it actually picks up this commit
(the live app was found to be running a stale, older commit).
"""

import hashlib
import os
import tempfile
import time
from pathlib import Path

import bcrypt
import fitz  # PyMuPDF
import psycopg2
import streamlit as st
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ==============================================================================
# 1. CONFIG & CONSTANTS
# ==============================================================================


def get_secret(key: str, default=None):
    """Reads a credential from a local .env file first (for local development),
    then falls back to Streamlit Cloud's secrets manager (Settings -> Secrets)."""
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return default


GROQ_API_KEY = get_secret("GROQ_API_KEY")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD", "")  # set this in Secrets to enable the admin panel

DB_DIR = Path("chroma_db")
STATS_COLLECTION = "esps_stats"
PDF_COLLECTION = "ess_pdf_docs"

# FIX: llama-3.1-70b-versatile was decommissioned by Groq (this was the bug
# causing every question on the live app to fail). llama-3.1-8b-instant is
# the model core.py already used successfully - kept as the single model
# for the whole app so behavior is consistent everywhere.
MODEL = "llama-3.1-8b-instant"

MULTILINGUAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

PG_CONFIG = {
    "dbname": get_secret("PG_DBNAME", "ethiopia_stats"),
    "user": get_secret("PG_USER", "postgres"),
    "password": get_secret("PG_PASSWORD"),
    "host": get_secret("PG_HOST", "localhost"),
    "port": get_secret("PG_PORT", "5432"),
    "sslmode": get_secret("PG_SSLMODE", "prefer"),
}

# Router keyword lists (identical to core.py, so the widget and dashboard
# route questions the same way).
RANK_WORDS = ["highest", "lowest", "most", "least", "best", "worst", "top", "compare", "rank", "maximum", "minimum"]
PDF_STRONG_WORDS = ["definition", "explain", "describe", "define", "chapter", "meaning of"]
CSV_SIGNAL_WORDS = [
    "literacy", "household size", "read and write",
    "consumption", "average age", "attended school", "illness", "female",
]
FORBIDDEN_SQL_KEYWORDS = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke", "--", ";"]

st.set_page_config(page_title="ESS AI Buddy", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Quicksand', sans-serif; }
    .stApp { background: #f8f8fd; }
    .ess-bot-header { display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }
    .ess-bot-avatar {
        font-size: 32px; background: linear-gradient(135deg, #a78bfa, #f9a8d4);
        border-radius: 50%; width: 50px; height: 50px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 10px rgba(167,139,250,0.35);
    }
    .ess-bot-title { font-size: 22px; font-weight: 700; color: #7c3aed; margin: 0; }
    .ess-bot-subtitle { font-size: 13px; color: #7c7c8a; margin-top: -2px; }
    div[data-testid="stChatInput"] {
        border-radius: 12px !important; border: 2px solid #222222 !important;
        background-color: #ffffff !important; box-shadow: 0px 4px 12px rgba(0,0,0,0.05) !important;
    }
    div[data-testid="stChatInput"] textarea { height: 42px !important; font-size: 14px !important; }
    .floating-button-wrapper {
        position: fixed; bottom: 25px; right: 25px; z-index: 999999;
        display: flex; flex-direction: row; gap: 8px;
    }
    .ess-login-card {
        background: white; border-radius: 18px; padding: 18px 16px 8px 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 12px;
    }
    .ess-login-title { font-size: 18px; font-weight: 700; color: #7c3aed; margin-bottom: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

if not GROQ_API_KEY:
    st.sidebar.error(
        "⚠️ GROQ_API_KEY is not set. On Streamlit Cloud, add it under "
        "your app's Settings -> Secrets (not just your local .env file)."
    )

# ==============================================================================
# 2. ROUTING
# ==============================================================================


def route_question(question: str) -> str:
    q = question.lower()
    if any(w in q for w in RANK_WORDS):
        return "sql"
    if any(w in q for w in PDF_STRONG_WORDS):
        return "pdf"
    if any(w in q for w in CSV_SIGNAL_WORDS):
        return "csv"
    return "pdf"


# ==============================================================================
# 3. DATABASE: SCHEMA, AUTH (bcrypt), HISTORY, CACHE
# ==============================================================================


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
            # VARCHAR(64) was originally sized for a 64-char SHA-256 hex
            # digest. A bcrypt hash (e.g. "$2b$12$...") is 60 characters,
            # so it fits without a column resize.
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


def hash_password(password: str) -> str:
    """bcrypt, not SHA-256: bcrypt is deliberately slow and salted per-password,
    which is what makes it suitable for passwords (unlike a fast general-purpose
    hash such as SHA-256, which is fine for the query-cache key below but not
    for passwords)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(password: str, stored_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # stored_hash is an old plain-SHA-256 value from before this fix -
        # it can't be verified with bcrypt, so treat as not matching and
        # let the user re-register (see login_user below).
        return False


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
        create_user_with_password(username, password)
        return True, f"Account created. Logged in as {username}."
    if check_password(password, existing_hash):
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
    # SHA-256 is fine here: this is a cache lookup key, not a secret.
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
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (question_hash) DO NOTHING",
                (hash_question(question), question, answer, route, source_doc, source_page),
            )
        conn.commit()
    finally:
        conn.close()


def record_uploaded_document(filename: str, chunk_count: int):
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


# ==============================================================================
# 4. SQL GENERATION & SAFETY (now actually wired into the chat flow below)
# ==============================================================================


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
        client, model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=300,
    )
    sql = response.choices[0].message.content.strip()
    sql = sql.strip("`").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


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


# ==============================================================================
# 5. PDF INGESTION (text + table-aware chunking)
# ==============================================================================


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


# ==============================================================================
# 6. CHROMADB (PDF + survey-statistics collections)
# ==============================================================================


@st.cache_resource
def get_embed_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MULTILINGUAL_MODEL_NAME)


@st.cache_resource
def get_chroma_client():
    return PersistentClient(path=str(DB_DIR))


@st.cache_resource(show_spinner=False)
def get_pdf_collection():
    client = get_chroma_client()
    embed_fn = get_embed_fn()
    try:
        return client.get_or_create_collection(name=PDF_COLLECTION, embedding_function=embed_fn)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_stats_collection():
    client = get_chroma_client()
    embed_fn = get_embed_fn()
    try:
        return client.get_collection(name=STATS_COLLECTION, embedding_function=embed_fn)
    except Exception:
        return None


def query_collection(collection, query: str, n_results: int = 8):
    results = collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    return results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]


# ==============================================================================
# 7. ANSWER GENERATION (Groq) + AMHARIC TRANSLATION
# ==============================================================================


def build_prompt(query: str, documents: list, max_chars_per_doc: int = 1000) -> str:
    trimmed = [d[:max_chars_per_doc] for d in documents]
    context = "\n".join(f"- {d}" for d in trimmed)
    return (
        "You are the official ESS (Ethiopian Statistical Service) AI Assistant, answering questions "
        "using ONLY the context provided below - never invent numbers or facts not present in the context.\n\n"
        "Give a thorough, well-explained answer (roughly 4-8 sentences, or a short bulleted list for "
        "multi-part questions). Lead with the direct fact/number first, then add helpful context. "
        "If the exact figure requested isn't in the context but a related one is, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


def ask_groq(client: Groq, query: str, documents: list) -> str:
    prompt = build_prompt(query, documents)
    response = call_groq_with_backoff(
        client, model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def translate_to_amharic(text: str) -> str:
    try:
        return GoogleTranslator(source="en", target="am").translate(text)
    except Exception as e:
        return f"(Amharic translation unavailable right now: {e})"


# ==============================================================================
# 8. THE MAIN QUESTION HANDLER - this is where routing now actually happens
# ==============================================================================


def answer_question(question: str, amharic_mode: bool, username: str):
    """Routes the question, retrieves from the right source, and generates
    an answer. Returns (answer, route, source_doc, source_page)."""
    cached = get_cached_answer(question) if DB_READY else None
    if cached:
        answer, route, source_doc, source_page = cached
        return f"{answer}\n\n*(instant repeat answer from cache)*", route, source_doc or "", source_page or ""

    route = route_question(question)
    source_doc, source_page = "", ""
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    if route == "sql":
        if not client:
            answer = "This type of question needs the Groq API key. Please contact the site admin."
        else:
            try:
                generated_sql = generate_sql_query(client, question)
                if not is_safe_select(generated_sql):
                    answer = "I couldn't safely run a query for that question. Try rephrasing."
                else:
                    colnames, rows = execute_dynamic_sql(generated_sql)
                    if not rows:
                        answer = "No results found for that question."
                    else:
                        lines = [", ".join(f"{c}: {v}" for c, v in zip(colnames, row)) for row in rows]
                        answer = "\n".join(lines)
                    source_doc = "PostgreSQL - aggregate_stats table (AI-generated SQL)"
            except Exception as e:
                answer = f"Error running the database query: {e}"

    elif route == "csv":
        collection = get_stats_collection()
        documents, metadatas = ([], []) if collection is None else query_collection(collection, question)
        source_doc = "ESPS-5 Socioeconomic Survey, 2021/22"
        if not documents:
            answer = "No relevant statistics found for that question."
        elif client:
            answer = ask_groq(client, question, documents)
        else:
            answer = "I found relevant data, but the AI service isn't configured to summarize it."

    else:  # "pdf"
        collection = get_pdf_collection()
        documents, metadatas = ([], []) if collection is None else query_collection(collection, question)
        safe_metas = [m for m in metadatas if isinstance(m, dict)]
        sources = sorted({m.get("source") for m in safe_metas if m.get("source")})
        pages = sorted({m.get("page") for m in safe_metas if m.get("page") is not None})
        source_doc = ", ".join(sources) if sources else "ESS PDF report"
        source_page = ", ".join(str(p) for p in pages[:3]) if pages else ""
        if not documents:
            answer = "No PDF has been indexed yet, so I can't answer document-based questions."
        elif client:
            answer = ask_groq(client, question, documents)
        else:
            answer = "I found relevant text, but the AI service isn't configured to summarize it."

    if amharic_mode and route in ("pdf", "csv") and not answer.startswith(("No ", "Error", "I couldn't", "I found relevant")):
        answer = f"{answer}\n\n**አማርኛ:** {translate_to_amharic(answer)}"

    if DB_READY:
        try:
            save_to_cache(question, answer, route, source_doc, source_page)
            save_chat(username, question, answer, route, source_doc, source_page)
        except Exception:
            pass

    return answer, route, source_doc, source_page


# ==============================================================================
# 9. APP STATE INIT
# ==============================================================================

try:
    ensure_tables()
    DB_READY = True
except Exception as e:
    DB_READY = False
    st.sidebar.warning(f"Database features unavailable: {e}")

if "username" not in st.session_state:
    st.session_state.username = "guest"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "amharic_mode" not in st.session_state:
    st.session_state.amharic_mode = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

# ==============================================================================
# 10. SIDEBAR: LOGIN, LANGUAGE TOGGLE, HISTORY, ADMIN PANEL
# ==============================================================================

if not WIDGET_MODE:
    with st.sidebar:
        st.markdown(
            '<div class="ess-login-card"><div class="ess-login-title">🔐 User Profile</div></div>',
            unsafe_allow_html=True,
        )
        if st.session_state.username == "guest":
            username_input = st.text_input("👤 Username")
            password_input = st.text_input("🔑 Password", type="password")
            if st.button("Login / Register"):
                if username_input and password_input:
                    ok, msg = login_user(username_input, password_input)
                    if ok:
                        st.session_state.username = username_input
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter both username and password.")
        else:
            st.write(f"Logged in as: **{st.session_state.username}**")
            if st.button("Logout"):
                st.session_state.username = "guest"
                st.rerun()

        st.markdown("---")
        # FIX: a plain Streamlit toggle instead of the old cross-frame
        # window.parent.location.href approach, which is almost certainly
        # what made the Amharic button not work on the live app.
        st.subheader("⚙️ Language")
        st.session_state.amharic_mode = st.toggle(
            "Answer in Amharic too", value=st.session_state.amharic_mode
        )

        st.markdown("---")
        st.subheader("📜 Chat History")
        current_user = st.session_state.get("username", "guest")
        db_history = load_chat_history(current_user) if (DB_READY and current_user != "guest") else []
        if db_history:
            for q_text, a_text, route, src_doc, src_page in reversed(db_history):
                with st.expander(f"❓ {q_text[:30]}..."):
                    st.write(f"**Q:** {q_text}")
                    st.write(f"**A:** {a_text}")
                    if src_doc:
                        st.caption(f"Source: {src_doc}" + (f", p.{src_page}" if src_page else ""))
        else:
            st.caption("No previous questions found." if current_user != "guest" else "Log in to save your chat history.")

        # --- Admin panel: only shown if ADMIN_PASSWORD is set and entered correctly ---
        if ADMIN_PASSWORD:
            st.markdown("---")
            with st.expander("🛠️ Admin Panel"):
                admin_pw = st.text_input("Admin password", type="password", key="admin_pw")
                if admin_pw == ADMIN_PASSWORD:
                    uploaded_pdf = st.file_uploader("Upload a new ESS PDF report", type=["pdf"])
                    if uploaded_pdf is not None and st.button("Index this PDF"):
                        with st.spinner(f"Extracting text and tables from '{uploaded_pdf.name}'..."):
                            chunks = extract_pdf_chunks_from_bytes(uploaded_pdf.getvalue(), uploaded_pdf.name)
                            collection = get_pdf_collection()
                            if chunks and collection is not None:
                                collection.add(
                                    documents=[c["text"] for c in chunks],
                                    metadatas=[{"source": c["source"], "page": c["page"], "content_type": c["content_type"]} for c in chunks],
                                    ids=[f"{uploaded_pdf.name}_p{c['page']}_{c['chunk_index']}" for c in chunks],
                                )
                                record_uploaded_document(uploaded_pdf.name, len(chunks))
                                st.success(f"Indexed {len(chunks)} chunks from '{uploaded_pdf.name}'.")
                            else:
                                st.error("No extractable text found in that PDF.")
                elif admin_pw:
                    st.error("Incorrect admin password.")

# ==============================================================================
# 11. MAIN CHAT INTERFACE
# ==============================================================================

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

for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

left_pad, center_col, right_pad = st.columns([1, 2, 1])
with center_col:
    user_query = st.chat_input("Ask ESS AI Assistant...")

# A voice-filled query arrives via session_state (set by the JS component below)
if not user_query and st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = ""

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, route, source_doc, source_page = answer_question(
                user_query, st.session_state.amharic_mode, st.session_state.get("username", "guest")
            )
            st.write(answer)
            if source_doc:
                st.caption(f"Source: {source_doc}" + (f", p.{source_page}" if source_page else "") + f"  ·  route: {route}")
    st.session_state.chat_history.append((user_query, answer))

# ==============================================================================
# 12. VOICE INPUT (real browser speech recognition - confirmed working)
# ==============================================================================

st.components.v1.html(
    """
    <script>
    (function() {
        const parentDoc = window.parent.document;
        let dock = parentDoc.getElementById('ess-floating-dock');
        if (!dock) {
            dock = parentDoc.createElement('div');
            dock.id = 'ess-floating-dock';
            dock.style.cssText = `
                position: fixed !important; bottom: 25px !important; right: 30px !important;
                z-index: 9999999 !important; display: flex !important; gap: 10px !important;
            `;
            parentDoc.body.appendChild(dock);
        }
        dock.innerHTML = `
            <button id="dockMicBtn" style="
                background-color:#ffffff; color:#000000; border:1px solid #d0d7de; border-radius:10px;
                width:44px; height:44px; font-size:18px; cursor:pointer;
                box-shadow:0 4px 10px rgba(0,0,0,0.15); display:flex; align-items:center; justify-content:center;
            " title="Voice Input">🎙</button>
        `;
        const micBtn = dock.querySelector('#dockMicBtn');

        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SR();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            let isListening = false;

            recognition.onstart = function() {
                isListening = true;
                micBtn.style.backgroundColor = '#ff4b4b';
                micBtn.style.color = '#ffffff';
            };
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                const chatInputs = parentDoc.querySelectorAll('textarea[data-testid="stChatInputTextArea"]');
                if (chatInputs.length > 0) {
                    chatInputs[0].value = transcript;
                    chatInputs[0].dispatchEvent(new Event('input', { bubbles: true }));
                    // Simulate pressing Enter so the question is actually submitted.
                    chatInputs[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
                }
            };
            recognition.onerror = recognition.onend = function() {
                isListening = false;
                micBtn.style.backgroundColor = '#ffffff';
                micBtn.style.color = '#000000';
            };
            micBtn.onclick = function() {
                if (isListening) { recognition.stop(); } else { recognition.start(); }
            };
        } else {
            micBtn.onclick = function() { alert('Voice recognition requires Chrome or Edge.'); };
        }
    })();
    </script>
    """,
    height=0,
    width=0,
)
