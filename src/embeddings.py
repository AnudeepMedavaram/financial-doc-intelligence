import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Use free local embeddings - no API cost
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTORSTORE_DIR = "vectorstore"


def get_embeddings():
    """
    Load HuggingFace sentence transformer embeddings.
    Runs locally - no API cost, no data leaving machine.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def create_vectorstore(chunks: List[Document], collection_name: str) -> Chroma:
    """
    Create ChromaDB vector store from document chunks.
    Persists to disk so you don't re-embed on every run.
    """
    embeddings = get_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=VECTORSTORE_DIR
    )
    
    print(f"Created vector store with {len(chunks)} chunks")
    print(f"Collection: {collection_name}")
    return vectorstore


def load_vectorstore(collection_name: str) -> Chroma:
    """
    Load existing vector store from disk.
    """
    embeddings = get_embeddings()
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    
    return vectorstore


def get_collection_name(filename: str) -> str:
    """
    Generate a valid collection name from filename.
    ChromaDB collection names must be alphanumeric with hyphens.
    """
    name = os.path.splitext(filename)[0]
    name = "".join(c if c.isalnum() else "-" for c in name)
    name = name[:50]
    return name.lower()