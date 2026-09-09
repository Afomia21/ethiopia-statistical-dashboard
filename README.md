Ethiopia Statistical Service (ESS) AI Buddy & Dashboard
An end-to-end Retrieval-Augmented Generation (RAG) platform, interactive analytics dashboard, and API backend for querying Ethiopian Socioeconomic Survey (ESPS) data and statistical PDF documents.

📌 Project Overview
This repository provides a unified data processing pipeline, vector storage system, and multi-interface AI assistant designed for the Ethiopia Statistical Service. It allows users to ask questions in English or Amharic and retrieve grounded statistical answers sourced directly from microdata aggregations and official publication PDFs.

Key Capabilities
Dual-Collection Vector Search: Operates over both preprocessed survey microdata (esps_stats) and structured PDF document chunks/tables (ess_pdf_docs).

Table-Aware PDF Extraction: Preserves row and column alignments from statistical tables in PDFs using PyMuPDF table finder utilities.

Multi-Interface Access:

Streamlit Dashboard (app.py): Web UI featuring interactive data exploration, PDF viewers, and chat.

FastAPI Backend (api.py): Cross-origin REST API (/ask) for web widget integrations.

Terminal CLI (chatbot.py): Command-line interface for local debugging.

Multilingual Support: Powered by paraphrase-multilingual-MiniLM-L12-v2 embeddings and Gemini 2.5 LLM generation.
ethiopia-statistical-dashboard/
### 📁 Repository Structure

```text
ethiopia-statistical-dashboard/
│
├── app.py                      # Main Streamlit dashboard interface
├── core.py                     # Primary RAG search routing and LLM execution backend
├── api.py                      # FastAPI server for website widget integration
├── chatbot.py                  # Interactive CLI tool for testing queries
│
├── local_rebuild.py            # Local builder for root chroma_db using multilingual embeddings
├── build_aggregate_stats.py    # Microdata ETL pipeline (ESPS Wave 5 CSV -> aggregate_stats.csv)
├── ingest_stats.py             # Legacy vector indexer for aggregate stats CSV
├── ingest_single.py            # CLI tool for vectorizing arbitrary merged CSV files
│
├── data set/
│   ├── ETH_2021_ESPS-W5_v02_M_CSV/  # Raw survey microdata
│   └── preprocessed/           # Output folder for aggregate_stats.csv
│
├── data/
│   └── pdf/                    # Source PDF statistical reports
│
├── chroma_db/                  # Production ChromaDB persistent directory
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
[Raw ESPS CSVs] ---> build_aggregate_stats.py ---> [aggregate_stats.csv] ──┐
                                                                          ├──> local_rebuild.py ---> [chroma_db/]
[PDF Reports]   ----------------------------------------------------------┘                              │
                                                                                                         │
[Streamlit UI (app.py)]  <--┐                                                                            │
[FastAPI (api.py)]       <----+--- core.py (Gemini 2.5 + SentenceTransformer) <--------------------------┘
[CLI (chatbot.py)]       <--┘
ETL Processing: build_aggregate_stats.py ingests ESPS Wave 5 CSV files across consumption, demographics, education, and health modules, generating standardized natural language statistical summaries.

Vector Indexing: local_rebuild.py extracts text and formatted tables from ./data/pdf alongside ./data set/preprocessed/aggregate_stats.csv, generating dense embeddings stored in root ./chroma_db.

Retrieval & RAG Pipeline: core.py handles route selection, executes similarity search against ChromaDB, builds strict context prompts, and passes grounded context to Gemini.

🚀 Setup & Installation
1. Prerequisites
Python 3.9+

Google Gemini API Key (GEMINI_API_KEY set in environment or Streamlit secrets)

2. Installation
Bash
# Clone the repository
git clone https://github.com/Afomia21/ethiopia-statistical-dashboard.git
cd ethiopia-statistical-dashboard

# Install required dependencies
pip install -r requirements.txt
💻 Usage Instructions
Rebuilding the Vector Database
To construct the local vector database prior to deployment:

Bash
python local_rebuild.py
Running the Streamlit Dashboard
Launch the web interface locally:

Bash
streamlit run app.py
Running the FastAPI Backend (for Web Widgets)
Start the REST API server:

Bash
uvicorn api:app --reload --port 8001
Health Check: GET http://localhost:8001/

Ask Endpoint: POST http://localhost:8001/ask

JSON
{
  "question": "What is the literacy rate in Amhara?",
  "amharic_mode": false
}
Running the CLI Chatbot
Bash
python chatbot.py
🛠 Tech Stack
Frontend / Dashboard: Streamlit

API Framework: FastAPI, Uvicorn, Pydantic

Vector Database: ChromaDB

Embeddings: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)

LLM Integration: google-generativeai (gemini-2.5-flash)

PDF & Data Processing: PyMuPDF (fitz), Pandas
