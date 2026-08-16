"""
api.py - the small "bridge" between the website widget and the chatbot
brain (core.py). This is the only new server code needed; it does not
touch app.py or how the Streamlit dashboard works.

HOW TO RUN THIS LOCALLY (for testing):
    pip install fastapi uvicorn --break-system-packages
    uvicorn api:app --reload --port 8001

Then the widget calls: http://localhost:8001/ask

HOW TO PUT THIS ON THE REAL WEBSITE LATER:
    Deploy this file to a small always-on host (Render, Railway, Fly.io,
    a small VPS, etc. - anywhere that isn't Streamlit Cloud, since it
    needs to allow normal cross-origin requests instead of Streamlit's
    iframe-based embedding). Then update WIDGET_API_URL in widget.html
    to point at that server's real address instead of localhost.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import process_question, get_embed_fn, get_chroma_client

app = FastAPI(title="ESS AI Buddy - small widget backend")


@app.on_event("startup")
def warm_up():
    """Loads the language-understanding model and connects to the database
    right when the server starts, instead of waiting for the first visitor
    to ask a question - so nobody has to sit through a slow first answer."""
    try:
        get_embed_fn()
        get_chroma_client()
    except Exception:
        pass  # if this fails, the first real question will just retry it

# Allow the widget to call this API from any website. Once you know the
# exact real website domain (e.g. https://ess.gov.et), it's safer to
# replace "*" with that specific domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str
    amharic_mode: bool = False


@app.get("/")
def health_check():
    return {"status": "ok", "message": "ESS AI Buddy widget backend is running."}


@app.post("/ask")
def ask(payload: Question):
    if not payload.question or not payload.question.strip():
        return {"answer": "Please type a question.", "route": "", "source": ""}

    answer, route, source_doc, source_page = process_question(
        payload.question.strip(), amharic_mode=payload.amharic_mode
    )

    source = source_doc
    if source_page:
        source += f", page(s) {source_page}"

    return {"answer": answer, "route": route, "source": source}
