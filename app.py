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
# 1. PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="ESS AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for header styling
st.markdown(
    """
    <style>
    .ess-bot-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 25px;
    }
    .ess-bot-avatar {
        font-size: 40px;
    }
    .ess-bot-title {
        font-size: 28px;
        font-weight: bold;
        margin: 0;
    }
    .ess-bot-subtitle {
        font-size: 14px;
        color: #666;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Environment setup
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
WIDGET_MODE = st.query_params.get("embed", "false").lower() == "true"
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "ess_pdf_docs"
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
    """Generates a grounded response using active Groq models with fallback support."""
    context_text = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."
    system_prompt = (
        "You are an expert AI assistant for the Ethiopia Statistical Service (ESS).\n"
        "Answer the user query strictly based on the provided context below.\n"
        "If the answer cannot be determined from the context, state that clearly."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuery: {query}"

    # Use active, current Groq model endpoints
    candidate_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]

    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_name,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception:
            # Catch all model-level failures (decommissioned 400, not found 404, rate limits) 
            # and gracefully fall back to the next model in candidate_models
            continue

    return "Error: None of the candidate Groq models were accessible with your current API Key."

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

def process_and_index_pdf(uploaded_file, collection):
    """Processes dynamic PDF uploads directly from chat_input."""
    session_key = f"uploaded_{uploaded_file.name}"
    if session_key not in st.session_state:
        with st.spinner(f"Ingesting uploaded document '{uploaded_file.name}'..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            doc = fitz.open(tmp_path)
            chunks, metadatas, ids = [], [], []

            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text").strip()
                if text:
                    chunks.append(text)
                    metadatas.append({"source": uploaded_file.name, "page": page_num + 1})
                    ids.append(f"upload_{uploaded_file.name}_p{page_num + 1}")

            doc.close()
            os.remove(tmp_path)

            if chunks and collection:
                collection.add(documents=chunks, metadatas=metadatas, ids=ids)
                st.session_state[session_key] = True
                st.toast(f"✅ Indexed '{uploaded_file.name}' successfully!")

# Initialize ChromaDB
pdf_collection = get_pdf_collection()

# ==============================================================================
# 3. SIDEBAR: AUTH & HISTORY
# ==============================================================================
with st.sidebar:
    st.subheader("👤 User Authentication")
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = "guest"

    if not st.session_state.authenticated:
        with st.form("login_form"):
            user_input = st.text_input("Username")
            password_input = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")

            if login_btn:
                if user_input.strip() and password_input.strip():
                    st.session_state.authenticated = True
                    st.session_state.username = user_input.strip()
                    st.success(f"Welcome, {st.session_state.username}!")
                    st.rerun()
                else:
                    st.error("Please enter both username and password.")
    else:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = "guest"
            st.rerun()

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
# 4. MAIN INTERFACE & CHAT DISPLAY
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

# Display current session messages
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

chat_input_response = st.chat_input("Ask ESS AI Assistant...", accept_file=True)

if chat_input_response:
    if isinstance(chat_input_response, str):
        user_query = chat_input_response
        uploaded_files = []
    else:
        user_query = getattr(chat_input_response, "text", "")
        uploaded_files = getattr(chat_input_response, "files", [])

    if uploaded_files and pdf_collection:
        for file_obj in uploaded_files:
            if file_obj.name.lower().endswith(".pdf"):
                process_and_index_pdf(file_obj, pdf_collection)

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

            if client:
                answer = ask_groq(client, user_query, docs)
            elif not GROQ_API_KEY:
                answer = "Error: GROQ_API_KEY is missing in secrets or environment variables."
            else:
                answer = "I couldn't locate specific information on that in the documents or tables."

            if DB_READY:
                current_user = st.session_state.get("username", "guest")
                save_chat(current_user, user_query, answer, "pdf")
                save_to_cache(user_query, answer, "pdf", "", "")

        st.session_state.chat_history.append((user_query, answer))
        st.rerun()
