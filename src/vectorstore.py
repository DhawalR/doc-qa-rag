from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
from src.config import TOP_K_RESULTS
import os

# Path where ChromaDB saves its files locally
CHROMA_DB_PATH = "chroma_db"


def build_vectorstore(chunks: list):
    """
    Takes document chunks, generates embeddings, and stores
    everything in ChromaDB.    
    Args:
        chunks: List of Document objects from chunking.py
    Returns:
        Chroma vectorstore instance
    """
    
    print(f"\nBuilding vector store from {len(chunks)} chunks...")
    print("This will call the OpenAI API to generate embeddings.")
    print("Please wait...\n")
    
    embedding_model = get_embedding_model()
    
    # Chroma.from_documents does three things in one call:
    # 1. Takes each chunk's text and sends it to OpenAI for embedding
    # 2. Stores the embedding vectors in ChromaDB
    # 3. Stores the original text and metadata alongside the vectors
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH
    )
    
    print(f"Vector store built successfully.")
    print(f"Stored at: {CHROMA_DB_PATH}/")
    
    return vectorstore


def load_vectorstore():
    """
    Loads an existing ChromaDB vector store from disk.
    Use this instead of rebuild_vectorstore() if embeddings
    already exist — saves API calls and time.
    
    Returns:
        Chroma vectorstore instance
    """
    
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"No vector store found at {CHROMA_DB_PATH}. "
            "Run build_vectorstore() first."
        )
    
    print(f"Loading existing vector store from {CHROMA_DB_PATH}/...")
    
    embedding_model = get_embedding_model()
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model
    )
    
    print("Vector store loaded successfully.")
    
    return vectorstore


def inspect_vectorstore(vectorstore) -> None:
    """
    Prints a summary of what is stored in ChromaDB.
    Args:
        vectorstore: Chroma vectorstore instance
    """
    
    print("\n" + "="*50)
    print("VECTOR STORE INSPECTION REPORT")
    print("="*50)
    
    # Get the underlying collection to inspect raw data
    collection = vectorstore._collection
    count = collection.count()
    
    print(f"\nTotal vectors stored: {count}")
    
    # Peek at first 3 stored items
    sample = collection.peek(3)
    
    print(f"\nSample stored documents:")
    for i in range(len(sample['ids'])):
        print(f"\n--- Item {i+1} ---")
        print(f"ID: {sample['ids'][i]}")
        print(f"Metadata: {sample['metadatas'][i]}")
        print(f"Text preview: {sample['documents'][i][:200]}")
        print(f"Embedding preview: {sample['embeddings'][i][:5]}...")




def get_or_build_vectorstore(chunks: list):
    """
    Smart loader — checks if vector store already exists.
    If yes, loads from disk (free, instant).
    If no, builds from chunks (costs API credits, takes time).
    
    This prevents duplicate API calls and wasted credits.
    
    Args:
        chunks: List of Document objects (used only if building fresh)
        
    Returns:
        Chroma vectorstore instance
    """
    
    # Check if ChromaDB folder exists and has data inside it
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print("Existing vector store found. Loading from disk...")
        print("No API calls needed.")
        return load_vectorstore()
    else:
        print("No existing vector store found. Building fresh...")
        print("This will use OpenAI API credits.")
        return build_vectorstore(chunks)