import os
import uuid
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .pdf_reader import extract_text_from_pdf

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

def get_embeddings():
    # Use HuggingFace sentence transformer for local embeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def index_pdf(pdf_path: str):
    """
    Extracts text from a PDF, chunks it, and indexes it into local ChromaDB.
    """
    text = extract_text_from_pdf(pdf_path)
    
    if text.startswith("[Error]") or text.startswith("[Warning]"):
        print(f"Skipping indexing due to read error: {text}")
        return False
        
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        print("No chunks generated from the document.")
        return False
        
    # Create or update Vector Store
    embeddings = get_embeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    vectorstore.persist()
    print(f"Successfully indexed {len(chunks)} chunks into ChromaDB.")
    return True

def retrieve_chunks(query: str, k: int = 3) -> list[str]:
    """
    Retrieves the most relevant text chunks from the vector store for a given query.
    """
    if not os.path.exists(DB_DIR):
        print("[Warning] DB directory does not exist. Please index a document first.")
        return []
        
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Retrieve top k relevant docs
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


# --- Session-scoped PDF upload support ---

def index_pdf_to_collection(pdf_path: str, collection_name: str) -> bool:
    """
    Index a PDF into a named ChromaDB collection (session-scoped).
    This keeps uploaded PDFs isolated from the persistent /data store.
    """
    text = extract_text_from_pdf(pdf_path)

    if text.startswith("[Error]") or text.startswith("[Warning]"):
        print(f"Skipping indexing due to read error: {text}")
        return False

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        print("No chunks generated from the uploaded document.")
        return False

    embeddings = get_embeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=DB_DIR
    )
    print(f"Successfully indexed {len(chunks)} chunks into collection '{collection_name}'.")
    return True


def retrieve_from_collection(query: str, collection_name: str, k: int = 3) -> list[str]:
    """
    Retrieve chunks from a specific named collection (for uploaded PDFs).
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def delete_collection(collection_name: str):
    """Clean up a session-scoped collection after research completes."""
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        client.delete_collection(collection_name)
        print(f"Cleaned up collection '{collection_name}'.")
    except Exception as e:
        print(f"[Warning] Could not delete collection '{collection_name}': {e}")


if __name__ == "__main__":
    db_path = DB_DIR
    test_pdf = os.path.join(os.path.dirname(__file__), "..", "data", "dummy.pdf")
    # Test only if PDF exists
    if os.path.exists(test_pdf):
        index_pdf(test_pdf)
        results = retrieve_chunks("What is this document about?")
        for idx, res in enumerate(results):
            print(f"Result {idx+1}: {res}\n")
    else:
        print("Provide a dummy.pdf in data/ to test indexing.")
