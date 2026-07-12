import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from groq import Groq
from deep_translator import GoogleTranslator
import psycopg2
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_DIR = Path("db") / "chroma_db"
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
}

RANK_WORDS = ["highest", "lowest", "most", "least", "best", "worst", "top", "compare", "rank", "maximum", "minimum"]
PDF_STRONG_WORDS = ["definition", "explain", "describe", "define", "chapter", "meaning of"]
CSV_SIGNAL_WORDS = [
    "population", "inflation", "rate", "total", "average", "percentage",
    "how many", "literacy", "household size", "consumption", "age", "female",
]
INDICATOR_KEYWORDS = {
    "literacy": "pct_literate",
    "read and write": "pct_literate",
    "school": "pct_ever_attended_school",
    "household size": "avg_household_size",
    "consumption": "avg_total_consumption_per_adult_equiv",
    "age": "avg_age",
    "female": "pct_female",
    "illness": "pct_illness_4wk",
}


def route_question(question: str) -> str:
    q = question.lower()
    if any(w in q for w in RANK_WORDS):
        return "sql"
    if any(w in q for w in PDF_STRONG_WORDS):
        return "pdf"
    if any(w in q for w in CSV_SIGNAL_WORDS):
        return "csv"
    return "pdf"


def detect_indicator(question: str):
    q = question.lower()
    for keyword, indicator in INDICATOR_KEYWORDS.items():
        if keyword in q:
            return indicator
    return None


def detect_direction(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["lowest", "least", "worst", "minimum"]):
        return "ASC"
    return "DESC"


def query_postgres_ranking(indicator: str, direction: str, limit: int = 5):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT group_name, value FROM aggregate_stats WHERE indicator = %s ORDER BY value {direction} LIMIT %s",
                (indicator, limit),
            )
            return cur.fetchall()
    finally:
        conn.close()


def format_ranking_answer(indicator: str, direction: str, rows: list) -> str:
    if not rows:
        return "No data found for that indicator."
    label = indicator.replace("pct_", "").replace("avg_", "").replace("_", " ")
    lines = [f"{i + 1}. {region}: {value:.1f}" for i, (region, value) in enumerate(rows)]
    order_word = "highest" if direction == "DESC" else "lowest"
    return f"Regions ranked by {label} ({order_word} first):\n" + "\n".join(lines)


def get_stats_collection():
    client = PersistentClient(path=str(DB_DIR))
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(name=STATS_COLLECTION, embedding_function=embed_fn)


def get_pdf_collection():
    client = PersistentClient(path=str(DB_DIR))
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
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


def ask_groq(client: Groq, query: str, documents: list) -> str:
    prompt = build_prompt(query, documents)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def translate_to_amharic(text: str) -> str:
    return GoogleTranslator(source="en", target="am").translate(text)


def transcribe_audio(audio_bytes: bytes) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
        return ""


st.title("Ethiopia Statistical Service Dashboard")

tab1, tab2, tab3 = st.tabs(["Chatbot", "Data Explorer", "Visualizations"])

with tab1:
    st.header("Ask the Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "prefill" not in st.session_state:
        st.session_state.prefill = ""

    st.write("🎤 Click to record your question, or just type below:")
    audio_bytes = audio_recorder(text="", icon_size="2x", key="voice_recorder")
    if audio_bytes:
        with st.spinner("Transcribing..."):
            transcribed = transcribe_audio(audio_bytes)
        if transcribed:
            st.session_state.prefill = transcribed
            st.success(f"Heard: \"{transcribed}\"")
        else:
            st.warning("Couldn't understand the audio. Please try again or type your question.")

    question = st.text_input("Type your question:", value=st.session_state.prefill)
    amharic_mode = st.checkbox("Also answer in Amharic")
    ask_clicked = st.button("Ask")

    if ask_clicked and question.strip():
        route = route_question(question)

        if route == "sql":
            indicator = detect_indicator(question)
            if not indicator:
                answer = "I couldn't tell which statistic you're asking about. Try mentioning literacy, school, household size, consumption, age, female, or illness."
            else:
                direction = detect_direction(question)
                try:
                    rows = query_postgres_ranking(indicator, direction)
                    answer = format_ranking_answer(indicator, direction, rows)
                    answer += "\n\n[Source: PostgreSQL - ESPS-5 aggregate_stats table]"
                except Exception as e:
                    answer = f"Error querying PostgreSQL: {e}"

        else:
            if route == "csv":
                collection = get_stats_collection()
                documents, metadatas = query_collection(collection, question)
                source_note = "[Source: ESPS-5 Socioeconomic Survey, 2021/22]"
            else:
                pdf_collection = get_pdf_collection()
                if pdf_collection is None:
                    documents, metadatas = [], []
                else:
                    documents, metadatas = query_collection(pdf_collection, question)
                source_note = "[Source: ESS PDF report]"

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
                    answer += f"\n\n{source_note}"
                except Exception as e:
                    answer = f"Error calling Groq API: {e}\n\nRaw matches:\n" + "\n".join(f"- {d}" for d in documents)
            else:
                answer = documents[0] + f"\n\n{source_note}"

        st.session_state.messages.append((question, answer, route))
        st.session_state.prefill = ""

    for q, a, route in reversed(st.session_state.messages):
        st.markdown(f"**You:** {q}  \n*(routed to: {route})*")
        st.markdown(f"**Chatbot:** {a}")
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