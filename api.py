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

from core import process_question

app = FastAPI(title="ESS AI Buddy - small widget backend")

# Note: we intentionally do NOT pre-load the AI model here anymore.
# On free/low-memory hosting, loading it immediately at startup can use
# more memory than is available, causing the server to fail before it
# even finishes starting. Loading it lazily (on the first real question
# instead) gives the server a much better chance of starting successfully.

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
