# Financial Document Intelligence System

A production-grade RAG (Retrieval-Augmented Generation) application that enables natural language Q&A on financial documents with cited sources, structured analysis reports, and multi-document comparison.

## Live Demo

🚀 Coming Soon (Deployment in Progress).

---

## What Makes This Different From Generic RAG Chatbots

Most RAG projects simply load a PDF and answer questions. This system adds several production-oriented capabilities:

### 1. Structured Output Parsing

Pydantic schemas force the LLM to return consistent JSON reports with predefined sections:

- Executive Summary
- Key Metrics
- Risk Factors
- Recommendations

instead of unpredictable plain text.

### 2. MMR Retrieval

Maximal Marginal Relevance retrieves diverse chunks instead of multiple similar chunks.

- fetch_k = 10
- lambda = 0.7

This balances relevance with diversity and improves answer quality.

### 3. Source Citations

Every answer includes the source page and chunk information, making responses verifiable for financial analysis.

### 4. Document Comparison

Upload two financial documents and compare them side by side.

The system retrieves context from two separate ChromaDB collections simultaneously.

### 5. Production FastAPI Backend

Features include:

- REST APIs
- Automatic Swagger documentation
- Session management
- Pydantic request validation
- Explicit error handling

---

# Architecture

```
User (Browser)
        │
        ▼
Streamlit Frontend (app.py)
        │
        ▼
FastAPI Backend (api.py)
        │
        ▼
Document Upload
        │
        ▼
PyPDF2 Loader
        │
        ▼
RecursiveCharacterTextSplitter
(chunk_size=1000, overlap=200)
        │
        ▼
HuggingFace Embeddings
(all-MiniLM-L6-v2)
        │
        ▼
ChromaDB Vector Store
        │
        ▼
MMR Retrieval
        │
        ▼
Groq LLM (Llama-3.1-8B-Instant)
        │
        ▼
Pydantic Output Parser
        │
        ▼
Answer + Citations + Structured Reports
        │
        ▼
Streamlit UI
(Chat / Report / Compare)
```

---

# Tech Stack

| Component | Technology | Why This Choice |
|------------|-----------------------------|-----------------------------------|
| LLM | Groq + Llama 3.1 8B | Fast and free inference |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Local, privacy-friendly |
| Vector Store | ChromaDB | Persistent document collections |
| Retrieval | LangChain MMR | Diverse retrieval |
| Output Parsing | Pydantic + JsonOutputParser | Structured reports |
| Frontend | Streamlit | Rapid ML deployment |
| Backend | FastAPI | Async APIs + Swagger docs |
| PDF Loading | PyPDF2 | Page metadata preservation |
| Text Splitting | RecursiveCharacterTextSplitter | Better semantic chunks |

---

# Project Structure

```
financial-doc-intelligence/

├── app.py
├── api.py
├── src/
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── output_parser.py
│   └── memory.py
├── data/
│   └── uploaded_docs/
├── vectorstore/
├── requirements.txt
├── .env.example
└── README.md
```

---

# Setup & Installation

## Clone Repository

```bash
git clone https://github.com/AnudeepMedavaram/financial-doc-intelligence.git
cd financial-doc-intelligence
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Create Environment Variables

Create a `.env` file:

```text
GROQ_API_KEY=your-groq-api-key
```

## Run Streamlit Frontend

```bash
streamlit run app.py
```

## Run FastAPI Backend

```bash
python api.py
```

---

# API Endpoints

| Method | Endpoint | Description |
|------------|------------------|------------------------------|
| GET | / | Health Check |
| POST | /upload | Upload PDF |
| POST | /ask | Ask questions |
| POST | /report | Generate structured report |
| POST | /compare | Compare two documents |
| GET | /session/{id} | Session information |
| GET | /sessions | List active sessions |
| DELETE | /session/{id} | Delete session |

Interactive API documentation is available at:

```
http://localhost:8000/docs
```

---

# Key Technical Decisions

### Why chunk_overlap = 200?

Sentences near chunk boundaries appear in both chunks, reducing context loss.

### Why MMR instead of similarity search?

Similarity search often returns nearly identical chunks.

MMR balances:

- relevance
- diversity

leading to better retrieval quality.

### Why local embeddings?

Financial documents may contain confidential information.

Using local HuggingFace embeddings ensures:

- no API cost
- better privacy
- no data leaves the machine

### Why Pydantic output parsing?

LLMs generate inconsistent formats.

Pydantic enforces:

- consistent schema
- type validation
- predictable downstream processing

### Why session-based architecture?

Each uploaded document gets an isolated session with:

- separate vector store
- independent conversation history

allowing multiple users/documents simultaneously.

---

# Known Limitations

- Sessions are stored in memory and are lost after server restart.
- Image-based scanned PDFs are not supported.
- Very large documents may exceed retrieval context limits.
- Groq free tier has request limits.

---

# Future Improvements

- PostgreSQL session persistence
- OCR support for scanned PDFs
- Streaming responses
- LangGraph agent orchestration
- Multi-hop retrieval and reasoning

---

# Author

**M V S Sai Ram Anudeep**

**LinkedIn**

https://linkedin.com/in/anudeep-medavaram-aa0a383a4

**GitHub**

https://github.com/AnudeepMedavaram
