import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

load_dotenv()

LLM_MODEL = "llama-3.1-8b-instant"


def get_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )


def create_qa_prompt_with_memory() -> PromptTemplate:
    """
    Prompt with conversation history injection.
    
    WHY we inject history as text not as message objects:
    RetrievalQA chain uses a single string prompt template.
    Message objects are for ChatPromptTemplate which requires
    different chain setup. String injection is simpler and
    works reliably with RetrievalQA.
    """
    template = """You are a financial document analyst with memory of previous exchanges.

Previous conversation:
{chat_history}

Use ONLY the context below to answer the current question.
If referring to something from conversation history, make that explicit.
If the answer is not in the context, say "I cannot find this in the document."
Always cite which page the information came from.

Document context:
{context}

Current question: {question}

Answer (with page citations):"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question", "chat_history"]
    )


def create_retrieval_chain(vectorstore: Chroma) -> RetrievalQA:
    llm = get_llm()

    # WHY we use a simple prompt without chat_history in the chain:
    # RetrievalQA chain only accepts 'context' and 'question' as variables.
    # We will inject chat_history directly into the question string instead.
    prompt = PromptTemplate(
        template="""You are a financial document analyst.
Use ONLY the context below to answer the question.
Always cite page numbers. If not found, say "Not in document."

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.7
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": False
        }
    )

    return qa_chain

def ask_question(
    qa_chain: RetrievalQA,
    question: str,
    chat_history: str = "No previous conversation."
) -> Dict[str, Any]:
    """Ask a question with conversation history injected into query."""
    try:
        # Inject history directly into the question string
        # This avoids the missing key error with RetrievalQA
        if chat_history and chat_history != "No previous conversation.":
            enriched_question = f"""Previous conversation:
{chat_history}

Current question: {question}"""
        else:
            enriched_question = question

        result = qa_chain.invoke({"query": enriched_question})

        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", "N/A"),
                "preview": doc.page_content[:100]
            })

        return {
            "answer": result["result"],
            "sources": sources,
            "num_sources": len(sources)
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "num_sources": 0
        }