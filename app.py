import streamlit as st
import os
import sys
import tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import load_and_split_pdf
from src.embeddings import create_vectorstore, load_vectorstore, get_collection_name
from src.retrieval import create_retrieval_chain, ask_question
from src.output_parser import generate_structured_report, format_report_for_display
from dotenv import load_dotenv

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Document Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message-user {
       background: #1a3a5c;
       padding: 1rem;
       border-radius: 10px;
       margin: 0.5rem 0;
       border-left: 4px solid #4a9eed;
       color: #ffffff;
    }
    .chat-message-ai {
        background: #1a2e1a;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        color: #ffffff;    
    }
    .source-chip {
        background: #fff3e0;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #ff9800;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-high { color: #f44336; font-weight: bold; }
    .risk-medium { color: #ff9800; font-weight: bold; }
    .risk-low { color: #4caf50; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE INITIALISATION ─────────────────────────────────────────────
# WHY: Streamlit reruns entire script on every interaction.
# session_state persists data across reruns.

def init_session_state():
    defaults = {
        "messages": [],           # Chat history for display
        "conversation_history": [],  # For injecting into prompts
        "vectorstore": None,      # Current document vector store
        "qa_chain": None,         # Current QA chain
        "current_doc": None,      # Current document name
        "report": None,           # Generated structured report
        "doc_processed": False    # Whether document is ready
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()
def safe_process_file(uploaded_file):
    """
    Process uploaded file with comprehensive error handling.
    
    Error types handled:
    - Empty PDF: file exists but no extractable text
    - Corrupted PDF: file cannot be read by PyPDF2
    - Too large: PDF over 50MB would cause memory issues
    - Wrong format: non-PDF uploaded despite filter
    
    WHY explicit error types matter:
    Different errors need different user responses.
    'Empty PDF' needs 'try a different file'.
    'Too large' needs 'compress your PDF first'.
    Generic 'something went wrong' helps nobody.
    """
    # Size check - warn if over 10MB
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > 50:
        return False, f"File too large ({file_size_mb:.1f}MB). Maximum size is 50MB."

    try:
        os.makedirs("data/uploaded_docs", exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
            dir="data/uploaded_docs"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        chunks = load_and_split_pdf(tmp_path)
        os.unlink(tmp_path)

        if not chunks:
            return False, "Could not extract text from this PDF. The file may be scanned or image-based. Try a text-based PDF."

        collection_name = get_collection_name(uploaded_file.name)
        vectorstore = create_vectorstore(chunks, collection_name)
        qa_chain = create_retrieval_chain(vectorstore)

        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = qa_chain
        st.session_state.current_doc = uploaded_file.name
        st.session_state.doc_processed = True
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.report = None
        st.session_state.chunks_count = len(chunks)

        return True, f"Successfully processed {len(chunks)} chunks from {uploaded_file.name}"

    except Exception as e:
        error_msg = str(e)
        if "PdfReadError" in error_msg or "EOF" in error_msg:
            return False, "PDF appears corrupted. Please try a different file."
        elif "PermissionError" in error_msg:
            return False, "Cannot access the file. Please try again."
        else:
            return False, f"Processing error: {error_msg}"


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def process_uploaded_file(uploaded_file) -> bool:
    """
    Process uploaded PDF and create vector store.
    Returns True on success, False on failure.
    
    WHY tempfile: Streamlit uploaded files are in-memory objects.
    We need to save to disk temporarily so PyPDF2 can read them.
    WHY delete=False: We need the file to persist until processing
    is complete. We manually delete after.
    """
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
            dir="data/uploaded_docs"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Load and split
            chunks = load_and_split_pdf(tmp_path)

            if not chunks:
                st.error("Could not extract text from PDF.")
                return False

            # Create vector store
            collection_name = get_collection_name(uploaded_file.name)
            vectorstore = create_vectorstore(chunks, collection_name)

            # Create QA chain
            qa_chain = create_retrieval_chain(vectorstore)

            # Update session state
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = qa_chain
            st.session_state.current_doc = uploaded_file.name
            st.session_state.doc_processed = True
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.report = None

        # Clean up temp file
        os.unlink(tmp_path)
        return True

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False


def get_conversation_context() -> str:
    """Format recent conversation history for prompt injection."""
    if not st.session_state.conversation_history:
        return "No previous conversation."

    recent = st.session_state.conversation_history[-5:]
    lines = []
    for exchange in recent:
        lines.append(f"Human: {exchange['question']}")
        lines.append(f"Assistant: {exchange['answer'][:200]}...")
    return "\n".join(lines)


def display_sources(sources: list):
    """Display source chunks in expandable section."""
    if sources:
        with st.expander(f"📄 Sources used ({len(sources)} chunks)", expanded=False):
            for i, src in enumerate(sources):
                st.markdown(
                    f'<span class="source-chip">Page {src["page"]}</span>',
                    unsafe_allow_html=True
                )
                st.caption(src["preview"][:150] + "...")
                if i < len(sources) - 1:
                    st.divider()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Financial Doc Intelligence")
    st.markdown("---")

    # File uploader
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any financial document — annual report, loan agreement, earnings transcript"
    )

    if uploaded_file:
        if st.session_state.current_doc != uploaded_file.name:
            if st.button("Process Document", type="primary", use_container_width=True, key="process_main"):
                success = process_uploaded_file(uploaded_file)
                if success:
                    st.success(f"✅ Processed: {uploaded_file.name}")
                    st.rerun()

    # Document status
    if st.session_state.doc_processed:
        st.markdown("---")
        st.markdown("### Current Document")
        st.info(f"📄 {st.session_state.current_doc}")

        # Stats
        if st.session_state.vectorstore:
            try:
                count = st.session_state.vectorstore._collection.count()
                st.metric("Chunks indexed", count)
            except Exception:
                pass

        st.metric("Messages", len(st.session_state.messages))

    st.markdown("---")

    # Clear conversation
    if st.button("Process Document", type="primary", use_container_width=True):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            success, message = safe_process_file(uploaded_file)
        if success:
            st.success(f"✅ {message}")
            st.rerun()
        else:
            st.error(f"❌ {message}")

    # Model info
    st.markdown("---")
    st.markdown("### Model Info")
    st.caption("LLM: Llama 3.1 8B (Groq)")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector DB: ChromaDB")
    st.caption("Retrieval: MMR (k=4)")


# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

st.markdown(
    '<p class="main-header">📊 Financial Document Intelligence</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Upload any financial document and ask questions with cited answers</p>',
    unsafe_allow_html=True
)

if not st.session_state.doc_processed:
    # Landing state
    st.info("👈 Upload a PDF document in the sidebar to get started")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📑 What you can upload")
        st.markdown("""
        - Annual Reports
        - Loan Agreements
        - Earnings Transcripts
        - Financial Statements
        - Investment Prospectus
        - Any PDF document
        """)
    with col2:
        st.markdown("### 💬 What you can ask")
        st.markdown("""
        - Key financial metrics
        - Risk factors
        - Revenue trends
        - Management outlook
        - Regulatory concerns
        - Comparison questions
        """)
    with col3:
        st.markdown("### ⚙️ How it works")
        st.markdown("""
        1. Upload PDF
        2. Document is chunked
        3. Chunks are embedded
        4. Ask questions
        5. MMR retrieval finds relevant chunks
        6. Groq LLM generates cited answer
        """)

else:
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Analysis Report", "🔄 Compare Documents"])

    # ── TAB 1: CHAT ──────────────────────────────────────────────────────────
    with tab1:
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message-user">🧑 {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message-ai">🤖 {message["content"]}</div>',
                    unsafe_allow_html=True
                )
                if "sources" in message:
                    display_sources(message["sources"])

        # Chat input
        question = st.chat_input(
            "Ask anything about the document...",
            key="chat_input"
        )

        if question:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            # Get answer
            with st.spinner("Thinking..."):
                history = get_conversation_context()
                result = ask_question(
                    st.session_state.qa_chain,
                    question,
                    history
                )

            answer = result["answer"]
            sources = result["sources"]

            # Add AI message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            # Save to conversation history for context injection
            st.session_state.conversation_history.append({
                "question": question,
                "answer": answer
            })

            st.rerun()

    # ── TAB 2: ANALYSIS REPORT ───────────────────────────────────────────────
    with tab2:
        st.markdown("### 📋 Structured Analysis Report")
        st.caption(
            "Automatically generated analysis using broad document context retrieval"
        )

        if st.session_state.report is None:
            if st.button(
                "Generate Analysis Report",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Generating structured report... this takes 10-15 seconds"):
                    # Get broad context for report
                    results = st.session_state.vectorstore.similarity_search(
                        "summary overview financials risks recommendations",
                        k=6
                    )
                    context = "\n\n".join([doc.page_content for doc in results])
                    st.session_state.report = generate_structured_report(
                        context,
                        st.session_state.current_doc
                    )
                st.rerun()
        else:
            report = st.session_state.report

            # Executive Summary
            st.markdown("#### Executive Summary")
            st.info(report.executive_summary)

            col1, col2 = st.columns(2)

            # Key Metrics
            with col1:
                st.markdown("#### Key Metrics")
                if report.key_metrics:
                    for metric in report.key_metrics:
                        trend_emoji = (
                            "📈" if metric.trend == "Up"
                            else "📉" if metric.trend == "Down"
                            else "➡️"
                        )
                        st.markdown(
                            f"**{metric.name}**: {metric.value} "
                            f"({metric.period}) {trend_emoji}"
                        )
                else:
                    st.caption("No specific metrics identified")

            # Risk Factors
            with col2:
                st.markdown("#### Risk Factors")
                if report.risk_factors:
                    for risk in report.risk_factors:
                        severity_class = (
                            "risk-high" if risk.severity == "High"
                            else "risk-medium" if risk.severity == "Medium"
                            else "risk-low"
                        )
                        st.markdown(
                            f'<span class="{severity_class}">'
                            f'[{risk.severity}]</span> '
                            f'**{risk.category}**: {risk.description}',
                            unsafe_allow_html=True
                        )
                else:
                    st.caption("No specific risks identified")

            # Key Findings
            st.markdown("#### Key Findings")
            if report.key_findings:
                for finding in report.key_findings:
                    st.markdown(f"• {finding}")

            # Recommendations
            st.markdown("#### Recommendations")
            if report.recommendations:
                for rec in report.recommendations:
                    st.markdown(f"→ {rec}")

            # Data Gaps
            if report.data_gaps:
                with st.expander("⚠️ Data Gaps Identified"):
                    for gap in report.data_gaps:
                        st.markdown(f"! {gap}")

            # Confidence
            st.progress(
                report.confidence_score,
                text=f"Analysis Confidence: {report.confidence_score:.0%}"
            )

            if st.button("🔄 Regenerate Report"):
                st.session_state.report = None
                st.rerun()

    # ── TAB 3: DOCUMENT COMPARISON ───────────────────────────────────────────
    with tab3:
        st.markdown("### 🔄 Compare Two Documents")
        st.caption(
            "Upload a second document to compare against the current one"
        )

        st.info(
            f"**Current document:** {st.session_state.current_doc}"
        )

        # Second document upload
        second_file = st.file_uploader(
            "Upload second PDF for comparison",
            type=["pdf"],
            key="compare_upload"
        )

        if second_file:
            if st.button("Process Second Document", type="secondary", key="process_second"):
                with st.spinner(f"Processing {second_file.name}..."):
                    success, message = safe_process_file.__wrapped__(second_file) \
                        if hasattr(safe_process_file, '__wrapped__') \
                        else (False, "Use main upload instead")

                    # Process second doc separately
                    try:
                        os.makedirs("data/uploaded_docs", exist_ok=True)
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf",
                            dir="data/uploaded_docs"
                        ) as tmp:
                            tmp.write(second_file.getvalue())
                            tmp_path = tmp.name

                        chunks2 = load_and_split_pdf(tmp_path)
                        os.unlink(tmp_path)

                        if chunks2:
                            col2_name = get_collection_name(second_file.name)
                            vs2 = create_vectorstore(chunks2, col2_name)
                            st.session_state.second_vectorstore = vs2
                            st.session_state.second_doc_name = second_file.name
                            st.success(f"Processed {len(chunks2)} chunks from {second_file.name}")
                        else:
                            st.error("Could not extract text from second PDF")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        # Comparison interface
        if hasattr(st.session_state, 'second_vectorstore') and \
           st.session_state.second_vectorstore is not None:

            st.markdown("---")
            st.markdown(
                f"**Comparing:** {st.session_state.current_doc} "
                f"vs {st.session_state.second_doc_name}"
            )

            aspect = st.text_input(
                "What aspect to compare?",
                value="key skills, experience, and qualifications",
                help="Be specific about what you want to compare"
            )

            if st.button("Generate Comparison", type="primary", key="generate_comparison"):
                with st.spinner("Comparing documents... this takes 15-20 seconds"):
                    try:
                        # Get context from both docs
                        results1 = st.session_state.vectorstore.similarity_search(
                            aspect, k=4
                        )
                        results2 = st.session_state.second_vectorstore.similarity_search(
                            aspect, k=4
                        )

                        context1 = "\n".join([d.page_content for d in results1])
                        context2 = "\n".join([d.page_content for d in results2])

                        from langchain_groq import ChatGroq
                        from langchain_core.messages import HumanMessage
                        import os

                        llm = ChatGroq(
                            model="llama-3.1-8b-instant",
                            temperature=0,
                            api_key=os.getenv("GROQ_API_KEY")
                        )

                        prompt = f"""Compare these two documents on: {aspect}

Document 1 ({st.session_state.current_doc}):
{context1}

Document 2 ({st.session_state.second_doc_name}):
{context2}

Provide structured comparison:
1. SIMILARITIES
2. KEY DIFFERENCES
3. DOCUMENT 1 STRENGTHS
4. DOCUMENT 2 STRENGTHS
5. OVERALL ASSESSMENT"""

                        response = llm.invoke([HumanMessage(content=prompt)])

                        st.markdown("#### Comparison Results")
                        st.markdown(response.content)

                    except Exception as e:
                        st.error(f"Comparison failed: {str(e)}")
        else:
            if not second_file:
                st.caption(
                    "Upload a second PDF above to enable comparison"
                )