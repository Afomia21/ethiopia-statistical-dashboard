"""
core.py - the chatbot's "brain", extracted from app.py so it can run
without Streamlit. This does NOT change app.py in any way - it's a
separate copy of the same logic (routing, ChromaDB search, table
extraction results, AI answer generation) so the small website widget
can call it directly through api.py.

If you ever change the logic in app.py's process_question() function,
make sure to mirror the change here too so the widget stays in sync.
"""

import hashlib
import os
import time
from pathlib import Path
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from groq import Groq
from deep_translator import GoogleTranslator
import psycopg2

load_dotenv()


def get_secret(key: str, default=None):
    return os.getenv(key, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")

DB_DIR = Path("chroma_db")
STATS_COLLECTION = "esps_stats"
PDF_COLLECTION = "ess_pdf_docs"
STATS_FILE = Path("data set") / "preprocessed" / "aggregate_stats.csv"
MODEL = "llama-3.1-8b-instant"

PG_CONFIG = {
    "dbname": get_secret("PG_DBNAME", "ethiopia_stats"),
    "user": get_secret("PG_USER", "postgres"),
    "password": get_secret("PG_PASSWORD"),
    "host": get_secret("PG_HOST", "localhost"),
    "port": get_secret("PG_PORT", "5432"),
    "sslmode": get_secret("PG_SSLMODE", "prefer"),
}

RANK_WORDS = ["highest", "lowest", "most", "least", "best", "worst", "top", "compare", "rank", "maximum", "minimum"]
PDF_STRONG_WORDS = ["definition", "explain", "describe", "define", "chapter", "meaning of"]
CSV_SIGNAL_WORDS = [
    "literacy", "household size", "read and write",
    "consumption", "average age", "attended school", "illness", "female",
]

MULTILINGUAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

FORBIDDEN_SQL_KEYWORDS = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke", "--", ";"]


def route_question(question: str) -> str:
    q = question.lower()
    if any(w in q for w in RANK_WORDS):
        return "sql"
    if any(w in q for w in PDF_STRONG_WORDS):
        return "pdf"
    if any(w in q for w in CSV_SIGNAL_WORDS):
        return "csv"
    return "pdf"


@lru_cache(maxsize=1)
def get_embed_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MULTILINGUAL_MODEL_NAME)


@lru_cache(maxsize=1)
def get_chroma_client():
    return PersistentClient(path=str(DB_DIR))


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


def query_collection(collection, query: str, n_results: int = 8):
    results = collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    return results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]


def build_prompt(query: str, documents: list, max_chars_per_doc: int = 800) -> str:
    trimmed = [d[:max_chars_per_doc] for d in documents]
    context = "\n".join(f"- {d}" for d in trimmed)
    return (
        "You are the official ESS (Ethiopian Statistical Service) AI Assistant, answering questions "
        "using ONLY the context provided below - never invent numbers or facts not present in the context.\n\n"
        "Guidelines for your answer:\n"
        "- Give a thorough, well-explained answer (roughly 4-8 sentences, or a short bulleted "
        "list with a sentence of explanation per point when the question covers multiple things).\n"
        "- Lead with the direct fact/number/answer first, then add helpful context: what the "
        "figure means, how it compares to related figures in the context, or any relevant detail "
        "the context provides.\n"
        "- If the question asks about MULTIPLE things (e.g. 'list the reports', 'what does each dataset "
        "cover', 'tell me about X and Y'): use a bullet list, one line per item, with a short "
        "explanation on each line rather than just a bare fact.\n"
        "- Don't repeat the same figure twice, and don't pad with empty filler sentences that add no "
        "information - every sentence should add something useful.\n"
        "- If the exact figure requested isn't in the context but a closely related one is, say so "
        "clearly, then give the closely related figure and explain what it actually covers.\n"
        "- If truly nothing relevant is in the context, say so in one or two sentences.\n"
        "- Do NOT include a 'Source:' line yourself - that is added separately.\n\n"
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


def hash_question(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()


def get_cached_answer(question: str):
    try:
        conn = psycopg2.connect(**PG_CONFIG)
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT answer, route, source_doc, source_page FROM query_cache WHERE question_hash = %s",
                (hash_question(question),),
            )
            return cur.fetchone()
    except Exception:
        return None
    finally:
        conn.close()


def save_to_cache(question: str, answer: str, route: str, source_doc: str, source_page: str):
    try:
        conn = psycopg2.connect(**PG_CONFIG)
    except Exception:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO query_cache (question_hash, question, answer, route, source_doc, source_page) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (question_hash) DO NOTHING",
                (hash_question(question), question, answer, route, source_doc, source_page),
            )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def save_chat(username: str, question: str, answer: str, route: str, source_doc: str = "", source_page: str = ""):
    try:
        conn = psycopg2.connect(**PG_CONFIG)
    except Exception:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (username, question, answer, route, source_doc, source_page) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (username, question, answer, route, source_doc, source_page),
            )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def translate_to_amharic(text: str) -> str:
    return GoogleTranslator(source="en", target="am").translate(text)


def ask_groq(client: Groq, query: str, documents: list) -> str:
    prompt = build_prompt(query, documents)
    response = call_groq_with_backoff(
        client, model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def process_question(question: str, amharic_mode: bool = False, username: str = "widget_visitor"):
    """Same behavior as app.py's process_question(): routes the question,
    retrieves from ChromaDB (text AND table chunks - table extraction
    already happens at indexing time in app.py), and generates an answer.
    Returns (answer, route, source_doc, source_page)."""
    cached = get_cached_answer(question)
    if cached:
        cached_answer, cached_route, cached_source_doc, cached_source_page = cached
        answer = f"{cached_answer}\n\n(instant repeat answer from cache)"
        return answer, cached_route, cached_source_doc or "", cached_source_page or ""

    route = route_question(question)
    source_doc, source_page = "", ""

    if route == "sql":
        if not GROQ_API_KEY:
            answer = "This type of question needs the Groq API key. Please contact the site admin."
        else:
            sql_client = Groq(api_key=GROQ_API_KEY)
            try:
                generated_sql = generate_sql_query(sql_client, question)
                if not is_safe_select(generated_sql):
                    answer = "I couldn't safely run a query for that question. Try rephrasing."
                else:
                    colnames, rows = execute_dynamic_sql(generated_sql)
                    if not rows:
                        answer = "No results found for that question."
                    else:
                        lines = [", ".join(f"{c}: {v}" for c, v in zip(colnames, row)) for row in rows]
                        answer = "\n".join(lines)
                    source_doc = "PostgreSQL - ESPS-5 aggregate_stats table (AI-generated SQL)"
            except Exception as e:
                answer = f"Error running dynamic SQL query: {e}"
    else:
        if route == "csv":
            collection = get_stats_collection()
            documents, metadatas = query_collection(collection, question)
            source_doc = "ESPS-5 Socioeconomic Survey, 2021/22"
        else:
            pdf_collection = get_pdf_collection()
            if pdf_collection is None:
                documents, metadatas = [], []
                source_doc = "ESS PDF report"
            else:
                documents, metadatas = query_collection(pdf_collection, question, n_results=8)
                safe_metas = [m for m in metadatas if isinstance(m, dict)]
                sources = sorted({m.get("source") for m in safe_metas if m.get("source")})
                pages = sorted({m.get("page") for m in safe_metas if m.get("page") is not None})
                source_doc = ", ".join(sources) if sources else "ESS PDF report"
                source_page = ", ".join(str(p) for p in pages[:3]) if pages else ""

        if not documents:
            answer = (
                "No PDF has been indexed yet, so I can't answer document-based questions."
                if route == "pdf" else "No relevant statistics found for that question."
            )
        elif GROQ_API_KEY:
            client = Groq(api_key=GROQ_API_KEY)
            try:
                answer = ask_groq(client, question, documents)
                if amharic_mode:
                    answer += f"\n\nAmharic: {translate_to_amharic(answer)}"
            except Exception:
                answer = "Something went wrong generating an answer. Please try rephrasing your question."
        else:
            answer = "I found relevant data, but can't summarize it right now - the AI service isn't configured."

    try:
        save_to_cache(question, answer, route, source_doc, source_page)
        save_chat(username, question, answer, route, source_doc, source_page)
    except Exception:
        pass

    return answer, route, source_doc, source_page
