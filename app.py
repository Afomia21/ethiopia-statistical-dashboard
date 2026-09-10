import os
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ESS AI - Ethiopia Statistical Service",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS matching the custom UI layout
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background-color: #1a1c1e !important;
    }
    .user-bubble {
        background-color: #2563eb;
        color: white;
        padding: 12px 18px;
        border-radius: 12px 12px 2px 12px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .bot-bubble {
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 14px 18px;
        border-radius: 12px 12px 12px 2px;
        margin: 8px 0;
        max-width: 85%;
        float: left;
        clear: both;
    }
    .file-tag {
        background-color: #e2e8f0;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 13px;
        display: inline-block;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "file_context" not in st.session_state:
    st.session_state.file_context = ""

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text directly from an uploaded PDF stream."""
    try:
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        text = "".join([page.get_text("text") + "\n" for page in doc])
        doc.close()
        return text.strip()
    except Exception as e:
        st.error(f"Error parsing PDF file: {e}")
        return ""

def ask_groq(client: Groq, query: str, context_text: str) -> str:
    """Generates a structured, well-formatted summary or answer using Groq."""
    system_prompt = (
        "You are an expert AI assistant for the Ethiopia Statistical Service (ESS).\n"
        "When requested to summarize a report, give a clean overview with clear bullet points "
        "covering Key Findings, Inflation Rates, and Food/Non-Food breakdowns if present in the text."
    )
    
    user_prompt = f"Context Document:\n{context_text[:14000]}\n\nUser Request: {query}" if context_text else query

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as err:
        return f"Error communicating with Groq API: {err}"

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 ESS AI")
    st.caption("Ethiopia Statistical Service")
    st.markdown("---")
    
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.current_file = None
        st.session_state.file_context = ""
        st.rerun()

    st.markdown("### Topic Shortcuts")
    if st.button("👥 Population Questions"):
        st.session_state.messages.append({"role": "user", "content": "What are the latest population estimates?"})
    if st.button("🌾 Agriculture Data"):
        st.session_state.messages.append({"role": "user", "content": "Provide an overview of agricultural statistics."})
    if st.button("📈 Economy & GDP"):
        st.session_state.messages.append({"role": "user", "content": "What are the latest GDP growth figures?"})
    if st.button("🏠 Housing Census"):
        st.session_state.messages.append({"role": "user", "content": "Summarize key housing survey findings."})

# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------
st.subheader("🤖 ESS AI Assistant")
st.caption("Ethiopia Statistical Service")

# Single file uploader block
uploaded_file = st.file_uploader("Upload report PDF", type=["pdf"], label_visibility="collapsed")

if uploaded_file and uploaded_file.name != st.session_state.current_file:
    st.session_state.current_file = uploaded_file.name
    st.session_state.file_context = extract_text_from_pdf(uploaded_file)
    st.success(f"✅ {uploaded_file.name} uploaded successfully.")

if st.session_state.current_file:
    st.markdown(
        f'<div class="file-tag">📄 <b>{st.session_state.current_file}</b> &nbsp;'
        f'<span style="color:green;">✔ Uploaded</span></div>',
        unsafe_allow_html=True
    )

# Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div><div style="clear:both;"></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div><div style="clear:both;"></div>', unsafe_allow_html=True)

# Single Prompt Input Bar
user_query = st.chat_input("Ask ESS AI Assistant...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        answer = ask_groq(client, user_query, st.session_state.file_context)
    else:
        answer = "GROQ_API_KEY missing. Please set it in your environment or secrets."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
