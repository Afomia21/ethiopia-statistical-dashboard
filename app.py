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
# 1. PAGE CONFIG & CUSTOM STYLES
# ==============================================================================
st.set_page_config(
    page_title="ESS AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for header and chat layout
st.markdown(
    """
    <style>
    .ess-bot-header {
        display: flex;
        align-items: center;
        gap: 15px;
        background-color: #f8f9fa;
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    .ess-bot-avatar {
        font-size: 36px;
    }
    .ess-bot-title {
        font-size: 24px;
        font-weight: bold;
        margin: 0;
        color: #1a365d;
    }
    .ess-bot-subtitle {
        font-size: 14px;
        color: #6c757d;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Environment setup & constants
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
WIDGET_MODE = st.query_params.get("embed", "false").lower() == "true"
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "ess_pdf_docs"

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "speech_input" not in st.session_state:
    st.session_state.speech_input = ""
if "amharic_mode" not in st.session_state:
    st.session_state.amharic_mode = False

# Handle Amharic toggle triggered from JavaScript floating button
query_params = st.query_params
if "toggle_amharic" in query_params:
    st.session_state.amharic_mode = not st.session_state.amharic_mode
    st.query_params.clear()
    st.rerun()

# ==============================================================================
# 2. CHROMADB & HELPER FUNCTIONS
# ==============================================================================
@st.cache_resource
def get_pdf_collection():
    """Initializes and returns the persistent ChromaDB collection."""
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
            return collection, True
    except Exception as e:
        st.sidebar.warning(f"ChromaDB connection note: {e}")
    return None, False

def ask_groq(client: Groq, query: str, context_chunks: list, amharic_enforced: bool = False) -> str:
    """Generates a grounded response using the Groq LLM API with language rules."""
    context_text = "\n\n".join(context_chunks)
    
    language_instruction = (
        "Respond STRICTLY and FLUENTLY in Amharic (አማርኛ)." 
        if amharic_enforced else 
        "Detect the language of the query. If asked in Amharic, respond strictly in Amharic. If asked in English, respond in English."
    )

    system_prompt = (
        "You are an expert AI assistant for the Ethiopia Statistical Service (ESS).\n"
        "Answer the user query strictly based on the provided context below.\n"
        "If the answer cannot be determined from the context, state that clearly.\n"
        f"LANGUAGE DIRECTIVE: {language_instruction}"
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

# Initialize ChromaDB connection
pdf_collection, DB_READY = get_pdf_collection()

# ==============================================================================
# 3. SIDEBAR: USER AUTHENTICATION & CHAT HISTORY
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
    st.subheader("⚙️ Language Settings")
    amharic_status = "ENABLED 🟢" if st.session_state.amharic_mode else "DISABLED ⚪"
    st.write(f"Amharic Mode: **{amharic_status}**")

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
                <p class="ess-bot-subtitle">Ethiopia Statistical Service · Your Friendly Stats Assistant</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render chat history bubbles
for user_q, bot_a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_q)
    with st.chat_message("assistant"):
        st.write(bot_a)

left_pad, center_col, right_pad = st.columns([1, 10, 1])

with center_col:
    chat_input_response = st.chat_input("Ask ESS AI Assistant...", accept_file=True, key="main_chat_input")

    user_query = ""
    uploaded_files = []

    if chat_input_response:
        if isinstance(chat_input_response, str):
            user_query = chat_input_response
        else:
            user_query = getattr(chat_input_response, "text", "")
            uploaded_files = getattr(chat_input_response, "files", [])

    if uploaded_files and pdf_collection:
        for file_obj in uploaded_files:
            if file_obj.name.lower().endswith(".pdf"):
                process_and_index_pdf(file_obj, pdf_collection)

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document knowledge base..."):
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
                        answer = ask_groq(client, user_query, docs, amharic_enforced=st.session_state.amharic_mode)
                    elif not GROQ_API_KEY:
                        answer = "Error: GROQ_API_KEY is missing."
                    else:
                        answer = "I couldn't locate specific information on that in the documents or tables."

                    if DB_READY:
                        current_user = st.session_state.get("username", "guest")
                        save_chat(current_user, user_query, answer, "pdf")
                        save_to_cache(user_query, answer, "pdf", "", "")

                st.write(answer)
                st.session_state.chat_history.append((user_query, answer))

# ==============================================================================
# 5. FLOATING BUTTONS PINNED AT BOTTOM RIGHT (VOICE & AMHARIC)
# ==============================================================================
amharic_bg = "#2b2b2b" if st.session_state.amharic_mode else "#ffffff"
amharic_color = "#ffffff" if st.session_state.amharic_mode else "#000000"

st.components.v1.html(
    f"""
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 999999; display: flex; gap: 12px;">
        <button id="amharicBtn" style="
            background-color: {amharic_bg};
            color: {amharic_color};
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            transition: all 0.2s ease-in-out;
        " title="Toggle Amharic Mode">አ</button>

        <button id="micBtn" style="
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 18px;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            transition: all 0.2s ease-in-out;
        " title="Record Voice Input">🎙</button>
    </div>

    <script>
    const micBtn = document.getElementById('micBtn');
    const amharicBtn = document.getElementById('amharicBtn');
    let recognition;
    let isListening = false;

    // 1. Voice Recognition Functionality
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'am-ET';

        recognition.onstart = function() {{
            isListening = true;
            micBtn.style.backgroundColor = '#ff4b4b';
            micBtn.style.color = '#ffffff';
        }};

        recognition.onresult = function(event) {{
            const transcript = event.results[0][0].transcript;
            const chatInputs = window.parent.document.querySelectorAll('textarea[data-testid="stChatInputTextArea"]');
            if (chatInputs.length > 0) {{
                chatInputs[0].value = transcript;
                chatInputs[0].dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
        }};

        recognition.onerror = function(event) {{
            console.error("Speech error:", event.error);
            isListening = false;
            micBtn.style.backgroundColor = '#ffffff';
            micBtn.style.color = '#000000';
        }};

        recognition.onend = function() {{
            isListening = false;
            micBtn.style.backgroundColor = '#ffffff';
            micBtn.style.color = '#000000';
        }};

        micBtn.onclick = function() {{
            if (isListening) {{
                recognition.stop();
            }} else {{
                recognition.start();
            }}
        }};
    }} else {{
        micBtn.onclick = function() {{
            alert('Speech recognition is not supported on this browser. Please use Google Chrome or Microsoft Edge.');
        }};
    }}

    // 2. Amharic Toggle Button Functionality
    amharicBtn.onclick = function() {{
        const url = new URL(window.parent.location.href);
        url.searchParams.set('toggle_amharic', 'true');
        window.parent.location.href = url.href;
    }};
    </script>
    """,
    height=90,
)
