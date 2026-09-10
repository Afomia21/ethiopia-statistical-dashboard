import os
import tempfile
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF for handling uploaded PDFs
from pathlib import Path
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from groq import Groq

# ==============================================================================
# 1. INITIAL CONFIGURATION & CUSTOM CSS FOR FLOATING BUTTONS
# ==============================================================================
st.set_page_config(
    page_title="ESS AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to pin the Mic and Amharic buttons cleanly to the bottom-left corner
st.markdown(
    """
    <style>
    .floating-button-wrapper {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 99999;
        display: flex;
        gap: 10px;
    }
    .custom-icon-btn {
        background-color: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 16px;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.2s ease-in-out;
    }
    .custom-icon-btn:hover {
        background-color: #f3f4f6;
        border-color: #000000;
        transform: translateY(-2px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Retrieve keys from Streamlit Cloud Secrets or local environment
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
WIDGET_MODE = st.query_params.get("embed", "false").lower() == "true"

# Define database directory paths
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "ess_pdf_docs"

# Global flags
DB_READY = False

# ==============================================================================
# 2. CHROMADB & HELPER FUNCTIONS
# ==============================================================================
@st.cache_resource
def get_pdf_collection():
    """Initializes and returns the persistent ChromaDB collection."""
    global DB_READY
    try:
        if DB_DIR.exists():
            client = PersistentClient(path=str(DB_DIR))
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu"
            )
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn
            )
            DB_READY = True
            return collection
    except Exception as e:
        st.warning(f"ChromaDB connection note: {e}")
    DB_READY = False
    return None

def ask_groq(client: Groq, query: str, context_chunks: list) -> str:
    """Generates a grounded response using the Groq LLM API."""
    context_text = "\n\n".join(context_chunks)
    system_prompt = (
        "You are an expert AI assistant for the Ethiopia Statistical Service (ESS).\n"
        "Answer the user query strictly based on the provided context below.\n"
        "If the answer cannot be determined from the context, state that clearly."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuery: {query}"

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as err:
        return f"Error communicating with Groq API: {err}"

def load_chat_history(username: str):
    return st.session_state.get(f"history_{username}", [])

def save_chat(username: str, query: str, answer: str, source_type: str):
    key = f"history_{username}"
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append((query, answer, source_type, "", ""))

def get_cached_answer(query: str):
    cache = st.session_state.get("query_cache", {})
    return cache.get(query)

def save_to_cache(query: str, answer: str, route: str, src_doc: str, src_page: str):
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    st.session_state.query_cache[query] = (answer, route, src_doc, src_page)

# Initialize ChromaDB connection
pdf_collection = get_pdf_collection()

# ==============================================================================
# 3. SIDEBAR: PDF FILE UPLOADER & HISTORY
# ==============================================================================
with st.sidebar:
    st.subheader("📄 Document Repository")
    uploaded_pdfs = st.file_uploader(
        "Upload new PDF reports",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_pdfs:
        if pdf_collection is not None:
            for uploaded_pdf in uploaded_pdfs:
                session_key = f"uploaded_{uploaded_pdf.name}"
                if session_key not in st.session_state:
                    with st.spinner(f"Ingesting {uploaded_pdf.name}..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_pdf.getvalue())
                            tmp_path = tmp_file.name

                        doc = fitz.open(tmp_path)
                        chunks = []
                        metadatas = []
                        ids = []

                        for page_num in range(len(doc)):
                            text = doc[page_num].get_text("text").strip()
                            if text:
                                chunks.append(text)
                                metadatas.append({
                                    "source": uploaded_pdf.name,
                                    "page": page_num + 1
                                })
                                ids.append(f"upload_{uploaded_pdf.name}_p{page_num + 1}")

                        doc.close()
                        os.remove(tmp_path)

                        if chunks:
                            pdf_collection.add(
                                documents=chunks,
                                metadatas=metadatas,
                                ids=ids
                            )
                            st.session_state[session_key] = True
                            st.success(f"✅ Indexed '{uploaded_pdf.name}'!")
        else:
            st.error("ChromaDB vector collection is unavailable.")

    st.markdown("---")
    st.subheader("📜 Chat History")
    current_user = st.session_state.get("username", "guest")
    db_history = []
    if DB_READY and current_user:
        try:
            db_history = load_chat_history(current_user)
        except Exception:
            pass

    all_history = db_history + [
        (q, a, "pdf", "", "") for q, a in st.session_state.get("chat_history", [])
        if (q, a) not in [(item[0], item[1]) for item in db_history]
    ]

    if all_history:
        for item in reversed(all_history):
            q_text = item[0]
            a_text = item[1]
            with st.expander(f"❓ {q_text[:30]}..."):
                st.write(f"**Q:** {q_text}")
                st.write(f"**A:** {a_text}")
    else:
        st.caption("No previous questions found.")

# ==============================================================================
# 4. MAIN INTERFACE & CHAT CONSOLE
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

left_pad, center_col, right_pad = st.columns([1, 2, 1])

with center_col:
    # Native chat input bar with built-in attachment upload icon
    chat_input_response = st.chat_input("Ask ESS AI Assistant...", accept_file=True)

    if chat_input_response:
        if isinstance(chat_input_response, str):
            user_query = chat_input_response
            uploaded_files = []
        else:
            user_query = getattr(chat_input_response, "text", "")
            uploaded_files = getattr(chat_input_response, "files", [])

        if user_query:
            cached_res = get_cached_answer(user_query) if DB_READY else None

            if cached_res:
                answer, route, src_doc, src_page = cached_res
            else:
                client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
                docs = []

                if pdf_collection:
                    res = pdf_collection.query(query_texts=[user_query], n_results=5)
                    docs = res.get("documents", [[]])[0]

                if client and docs:
                    answer = ask_groq(client, user_query, docs)
                elif not GROQ_API_KEY:
                    answer = "Error: GROQ_API_KEY is not configured in secrets."
                else:
                    answer = "I couldn't locate specific information on that in the documents or tables."

                if DB_READY:
                    current_user = st.session_state.get("username", "guest")
                    save_chat(current_user, user_query, answer, "pdf")
                    save_to_cache(user_query, answer, "pdf", "", "")

            st.session_state.chat_history.append((user_query, answer))
            st.markdown(f"**Answer:** {answer}")

# ==============================================================================
# 5. FIXED BOTTOM-LEFT FLOATING BUTTONS
# ==============================================================================
st.markdown(
    """
    <div class="floating-button-wrapper">
        <button class="custom-icon-btn" onclick="alert('Mic clicked')">🎙</button>
        <button class="custom-icon-btn" onclick="alert('Amharic clicked')">አ</button>
    </div>
    """,
    unsafe_allow_html=True,
)
