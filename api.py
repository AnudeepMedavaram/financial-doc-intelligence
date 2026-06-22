import os
import sys
import tempfile
import uuid
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import load_and_split_pdf
from src.embeddings import create_vectorstore, load_vectorstore, get_collection_name
from src.retrieval import create_retrieval_chain, ask_question
from src.output_parser import generate_structured_report

load_dotenv()

# ── APP SETUP ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Document Intelligence API",
    description="""
    RAG-powered financial document analysis API.
    
    Upload any financial PDF and get:
    - Natural language Q&A with cited sources
    - Structured analysis reports
    - Conversation memory across questions
    """,
    version="1.0.0"
)

# CORS - allows Streamlit frontend to call FastAPI
# WHY: Streamlit runs on port 8501, FastAPI on 8000
# Without CORS, browser blocks cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── IN-MEMORY STORE ───────────────────────────────────────────────────────────
# WHY in-memory instead of database for now:
# For a portfolio project this is sufficient.
# In production this would be PostgreSQL with session management.
# Day 7 adds document comparison which will show why persistence matters.

active_sessions = {}  # session_id -> {vectorstore, qa_chain, history, metadata}


# ── REQUEST/RESPONSE MODELS ───────────────────────────────────────────────────
# WHY Pydantic models for requests:
# FastAPI automatically validates incoming JSON against these schemas.
# If a required field is missing or wrong type, FastAPI returns a
# clear 422 error before our code even runs.

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    include_sources: bool = True


class QuestionResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list
    num_sources: int
    timestamp: str


class ReportRequest(BaseModel):
    session_id: str


class SessionInfo(BaseModel):
    session_id: str
    document_name: str
    chunks_indexed: int
    messages_count: int
    created_at: str


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Financial Document Intelligence API",
        "version": "1.0.0",
        "endpoints": ["/upload", "/ask", "/report", "/session/{id}", "/sessions"]
    }


@app.post("/upload", response_model=SessionInfo)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document and create a new analysis session.
    
    Returns a session_id to use in subsequent requests.
    
    WHY session_id pattern:
    Allows multiple users to work with different documents
    simultaneously without interfering with each other.
    Each session has its own vectorstore and conversation history.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_FILE_TYPE",
                "message": "Only PDF files are supported",
                "filename": file.filename
            }
        )

    try:
        # Save to temp file
        # WHY: PyPDF2 needs a file path, not bytes
        content = await file.read()

        os.makedirs("data/uploaded_docs", exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
            dir="data/uploaded_docs"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Process document
        chunks = load_and_split_pdf(tmp_path)
        os.unlink(tmp_path)  # Clean up temp file

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "EMPTY_DOCUMENT",
                    "message": "Could not extract text from PDF",
                    "filename": file.filename
                }
            )

        # Create vector store and QA chain
        collection_name = get_collection_name(file.filename)
        vectorstore = create_vectorstore(chunks, collection_name)
        qa_chain = create_retrieval_chain(vectorstore)

        # Create session
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "vectorstore": vectorstore,
            "qa_chain": qa_chain,
            "history": [],
            "document_name": file.filename,
            "chunks_count": len(chunks),
            "created_at": datetime.now().isoformat()
        }

        return SessionInfo(
            session_id=session_id,
            document_name=file.filename,
            chunks_indexed=len(chunks),
            messages_count=0,
            created_at=active_sessions[session_id]["created_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PROCESSING_ERROR",
                "message": str(e)
            }
        )


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    """
    Ask a question about the uploaded document.
    
    Uses conversation history for context-aware follow-up questions.
    Returns answer with cited source chunks.
    """
    if request.session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Session {request.session_id} not found. Upload a document first.",
                "session_id": request.session_id
            }
        )

    session = active_sessions[request.session_id]

    # Format conversation history
    history_lines = []
    for exchange in session["history"][-5:]:
        history_lines.append(f"Human: {exchange['question']}")
        history_lines.append(f"Assistant: {exchange['answer'][:200]}...")
    chat_history = "\n".join(history_lines) if history_lines else "No previous conversation."

    # Get answer
    result = ask_question(
        session["qa_chain"],
        request.question,
        chat_history
    )

    # Save to history
    session["history"].append({
        "question": request.question,
        "answer": result["answer"],
        "timestamp": datetime.now().isoformat()
    })

    return QuestionResponse(
        session_id=request.session_id,
        question=request.question,
        answer=result["answer"],
        sources=result["sources"] if request.include_sources else [],
        num_sources=result["num_sources"],
        timestamp=datetime.now().isoformat()
    )


@app.post("/report")
async def generate_report(request: ReportRequest):
    """
    Generate a structured financial analysis report for the document.
    
    WHY separate endpoint from /ask:
    Report generation uses different retrieval strategy (broad context)
    and different output format (structured JSON vs natural language).
    Keeping them separate follows single responsibility principle.
    """
    if request.session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Session {request.session_id} not found",
                "session_id": request.session_id
            }
        )

    session = active_sessions[request.session_id]
    vectorstore = session["vectorstore"]

    # Broad retrieval for report generation
    results = vectorstore.similarity_search(
        "summary financials risks recommendations overview",
        k=6
    )
    context = "\n\n".join([doc.page_content for doc in results])

    report = generate_structured_report(
        context,
        session["document_name"]
    )

    return {
        "session_id": request.session_id,
        "document_name": session["document_name"],
        "report": report.model_dump(),
        "generated_at": datetime.now().isoformat()
    }


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about an active session."""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Session {session_id} not found",
                "session_id": session_id
            }
        )

    session = active_sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        document_name=session["document_name"],
        chunks_indexed=session["chunks_count"],
        messages_count=len(session["history"]),
        created_at=session["created_at"]
    )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "total": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "document_name": s["document_name"],
                "messages": len(s["history"]),
                "created_at": s["created_at"]
            }
            for sid, s in active_sessions.items()
        ]
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail={"error": "SESSION_NOT_FOUND", "session_id": session_id}
        )

    del active_sessions[session_id]
    return {"message": f"Session {session_id} deleted", "session_id": session_id}
class CompareRequest(BaseModel):
    session_id_1: str
    session_id_2: str
    aspect: str = "overall comparison of key findings and risks"


@app.post("/compare")
async def compare_documents(request: CompareRequest):
    """
    Compare two uploaded documents side by side.
    
    WHY this is architecturally interesting:
    We retrieve context from two separate ChromaDB collections
    simultaneously and inject both into a single prompt.
    The LLM reasons across both documents to identify
    similarities, differences, and relative strengths.
    
    Use cases:
    - Compare two annual reports (year over year)
    - Compare two loan applications
    - Compare competitor financial statements
    """
    # Validate both sessions exist
    missing = []
    if request.session_id_1 not in active_sessions:
        missing.append(request.session_id_1)
    if request.session_id_2 not in active_sessions:
        missing.append(request.session_id_2)

    if missing:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Sessions not found: {missing}",
                "missing_sessions": missing
            }
        )

    session1 = active_sessions[request.session_id_1]
    session2 = active_sessions[request.session_id_2]

    # Retrieve context from both documents independently
    query = request.aspect
    results1 = session1["vectorstore"].similarity_search(query, k=4)
    results2 = session2["vectorstore"].similarity_search(query, k=4)

    context1 = "\n".join([doc.page_content for doc in results1])
    context2 = "\n".join([doc.page_content for doc in results2])

    # Comparison prompt
    from src.retrieval import get_llm
    llm = get_llm()
    llm = get_llm()
    from langchain_core.messages import HumanMessage

    comparison_prompt = f"""You are a senior financial analyst comparing two documents.

Document 1: {session1['document_name']}
{context1}

---

Document 2: {session2['document_name']}
{context2}

---

Compare these two documents focusing on: {request.aspect}

Structure your comparison as:
1. SIMILARITIES: What do both documents have in common?
2. DIFFERENCES: How do they differ on the key aspect?
3. DOCUMENT 1 STRENGTHS: What does Document 1 show better?
4. DOCUMENT 2 STRENGTHS: What does Document 2 show better?
5. OVERALL ASSESSMENT: Which document presents a stronger case and why?

Be specific and cite which document each point comes from."""

    response = llm.invoke([HumanMessage(content=comparison_prompt)])

    return {
        "document_1": session1["document_name"],
        "document_2": session2["document_name"],
        "aspect_compared": request.aspect,
        "comparison": response.content,
        "generated_at": datetime.now().isoformat()
    }


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )