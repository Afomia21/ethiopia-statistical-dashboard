import os
import tempfile
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF for handling uploaded PDFs
from pathlib import Path
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from groq import Groq
import streamlit.components.v1 as components

# ==============================================================================
# 1. PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="ESS AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polish and clean action button layout
st.markdown(
    """
    <style>
    /* Styling for the header container */
    .ess-bot-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    .ess-bot-avatar { font-size: 36px; }
    .ess-bot-title { font-size: 24px; font-weight: bold; margin: 0; }
    .ess-bot-subtitle { font-size: 14px; color: #666; margin: 0; }

    /* Custom compact styling for action buttons directly below chat bar */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Environment & Globals Setup
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
WIDGET_MODE = st.query_params.get("embed", "false").lower() == "true"
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "ess_pdf_docs"
DB_READY = False

if "amharic_mode" not in st.session_state:
    st.session_state.amharic_mode = False
if "show_mic" not in st.session_state:
    st.session_state.show_mic = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================================================================
# 2. CHROMADB & HELPER FUNCTIONS
# ==============================================================================
@st.cache_resource
def get_pdf_collection():
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
    except Exception:
        pass
    DB_READY = False
    return None

def ask_groq(client: Groq, query: str, context_chunks: list = None, amharic: bool = False) -> str:
    """Generates a response using Groq LLM API with or without PDF context."""
    if context_chunks:
        context_text = "\n\n".join(context_chunks)
        user_prompt = f"Context from documents:\n{context_text}\n\nUser Question: {query}"
        instructions = "Answer the user question based on the context above if applicable."
    else:
        user_prompt = query
        instructions = "Answer the user question as an expert assistant on Ethiopian statistics and general information."

    lang_instruction = "Respond in clear Amharic (አማርኛ)." if amharic else "Respond in clear English."
    
    system_prompt = (
        "You are an expert AI assistant for the Ethiopia Statistical Service (ESS).\n"
        f"{instructions}\n"
        f"{lang_instruction}"
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b",
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

def process_and_index_pdf(uploaded_file, collection):
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

pdf_collection = get_pdf_collection()

# ==============================================================================
# 3. SIDEBAR: AUTHENTICATION & HISTORY
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
# 4. MAIN INTERFACE & CHAT LOGIC
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

left_pad, center_col, right_pad = st.columns([1, 2, 1])

with center_col:
    if st.session_state.show_mic:
        components.html(
            """
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; font-family: sans-serif; text-align: center;">
                <p style="margin: 0 0 10px 0; font-weight: bold; color: #1f2937;">🎙️ Direct Voice Recorder</p>
                <button id="recBtn" style="background-color: #e11d48; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold;">
                    🔴 Start Recording
                </button>
                <p id="statusMsg" style="margin-top: 8px; font-size: 13px; color: #4b5563;">Click button to speak...</p>
            </div>
            <script>
                let btn = document.getElementById("recBtn");
                let statusMsg = document.getElementById("statusMsg");
                let recording = false;
                let mediaRecorder;
                let audioChunks = [];

                btn.onclick = async () => {
                    if (!recording) {
                        try {
                            let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                            mediaRecorder = new MediaRecorder(stream);
                            audioChunks = [];
                            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                            mediaRecorder.onstop = () => {
                                statusMsg.innerText = "Recording finished! Type or submit your question.";
                            };
                            mediaRecorder.start();
                            recording = true;
                            btn.innerText = "⏹️ Stop Recording";
                            btn.style.backgroundColor = "#2563eb";
                            statusMsg.innerText = "Listening...";
                        } catch (err) {
                            statusMsg.innerText = "Microphone access denied or not supported.";
                        }
                    } else {
                        mediaRecorder.stop();
                        recording = false;
                        btn.innerText = "🔴 Record Again";
                        btn.style.backgroundColor = "#e11d48";
                    }
                };
            </script>
            """,
            height=120,
        )

    if st.session_state.amharic_mode:
        st.caption("🇪🇹 **Amharic Mode Active** — Responses will be formatted in Amharic (አማርኛ).")

    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    # 1. Main Chat Input Bar
    chat_input_response = st.chat_input("Ask ESS AI Assistant...", accept_file=True)

    # 2. Side-by-Side Action Buttons directly below the input bar
    # Using small proportional columns [1, 1, 8] keeps them tightly nested on the left side
    btn_col1, btn_col2, _ = st.columns([1, 1, 8])

    with btn_col1:
        mic_label = "🎙️ Record" if st.session_state.show_mic else "🎙️"
        if st.button(mic_label, key="action_mic_btn"):
            st.session_state.show_mic = not st.session_state.show_mic
            st.rerun()

    with btn_col2:
        amh_label = "🇪🇹 አማርኛ" if st.session_state.amharic_mode else "አ"
        if st.button(amh_label, key="action_amh_btn"):
            st.session_state.amharic_mode = not st.session_state.amharic_mode
            st.toast(
                f"Amharic mode {'enabled 🇪🇹' if st.session_state.amharic_mode else 'disabled 🌐'}"
            )
            st.rerun()

    # Chat submission handling
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

            if cached_res and not st.session_state.amharic_mode:
                answer, route, src_doc, src_page = cached_res
            else:
                client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
                docs = []

                if pdf_collection:
                    try:
                        res = pdf_collection.query(query_texts=[user_query], n_results=5)
                        docs = res.get("documents", [[]])[0]
                    except Exception:
                        docs = []

                if client:
                    answer = ask_groq(client, user_query, docs if docs else None, amharic=st.session_state.amharic_mode)
                elif not GROQ_API_KEY:
                    answer = "Error: GROQ_API_KEY is missing."
                else:
                    answer = "Unable to process the request at this time."

                if DB_READY:
                    current_user = st.session_state.get("username", "guest")
                    save_chat(current_user, user_query, answer, "pdf")
                    save_to_cache(user_query, answer, "pdf", "", "")

            st.session_state.chat_history.append((user_query, answer))
            st.rerun()
