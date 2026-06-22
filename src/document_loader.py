import os
from typing import List
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return list of Document objects
    with page content and metadata.
    """
    documents = []
    
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            if text and text.strip():
                doc = Document(
                    page_content=text.strip(),
                    metadata={
                        "source": os.path.basename(file_path),
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "file_path": file_path
                    }
                )
                documents.append(doc)
        
        print(f"Loaded {len(documents)} pages from {os.path.basename(file_path)}")
        return documents
    
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    chunk_size=1000 and chunk_overlap=200 ensures context
    is not lost at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def load_and_split_pdf(file_path: str) -> List[Document]:
    """
    Complete pipeline: load PDF and split into chunks.
    """
    documents = load_pdf(file_path)
    if not documents:
        return []
    chunks = split_documents(documents)
    return chunks