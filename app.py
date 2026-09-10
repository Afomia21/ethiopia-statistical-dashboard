import os
import streamlit as st
from groq import Groq

# Place your page config, imports, database functions (load_chat_history, save_chat, get_cached_answer, etc.),
# and collection getters (get_pdf_collection) above this section.

current_user = st.session_state.get("username", "guest")
db_history = []
if DB_READY and current_user:
    try:
        db_history = load_chat_history(current_user)
    except Exception:
        pass

all_history = db_history + [
    (q, a, "pdf", "", "")
    for q, a in st.session_state.get("chat_history", [])
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

# --- Question Box & Results Section ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

left_pad, center_col, right_pad = st.columns([1, 2, 1])

with center_col:
    # Enabled attachment paperclip icon directly inside the native Streamlit chat input bar
    chat_input_response = st.chat_input(
        "Ask ESS AI Assistant...", accept_file=True
    )

    if chat_input_response:
        # Support both string inputs and dictionary/object responses containing files
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
                pdf_col = get_pdf_collection()
                docs = []

                if pdf_col:
                    res = pdf_col.query(query_texts=[user_query], n_results=5)
                    docs = res.get("documents", [[]])[0]

                if client and docs:
                    answer = ask_groq(client, user_query, docs)
                else:
                    answer = "I couldn't locate specific information on that in the documents or tables."

                if DB_READY:
                    current_user = st.session_state.get("username", "guest")
                    save_chat(current_user, user_query, answer, "pdf")
                    save_to_cache(user_query, answer, "pdf", "", "")

            st.session_state.chat_history.append((user_query, answer))
            st.markdown(f"**Answer:** {answer}")

# --- Side-by-side Bottom Floating Buttons ---
st.markdown(
    """
    <div class="floating-button-wrapper">
        <button class="custom-icon-btn" onclick="alert('Mic clicked')">🎙</button>
        <button class="custom-icon-btn" onclick="alert('Amharic clicked')">አ</button>
    </div>
    """,
    unsafe_allow_html=True,
)
