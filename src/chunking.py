from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list) -> list:
    """
    Splits a list of Document objects into smaller chunks.
    Args:
        documents: List of Document objects from ingestion.py
    Returns:
        List of smaller Document objects (chunks)
    """
    
    print(f"\nChunking {len(documents)} pages...")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # Maximum characters per chunk
        chunk_overlap=CHUNK_OVERLAP,  # Characters repeated between chunks
        length_function=len,          # How to measure chunk size (character count)
        separators=["\n\n", "\n", " ", ""]  # Priority order for splitting
    )
    
    # split_documents handles the entire list at once
    # It preserves metadata from original documents (page number, source)
    chunks = splitter.split_documents(documents)
    
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks


def inspect_chunks(chunks: list) -> None:
    """
    Prints a summary of chunks for debugging and understanding.
    Args:
        chunks: List of chunk Document objects from chunk_documents()
    """
    
    print("\n" + "="*50)
    print("CHUNK INSPECTION REPORT")
    print("="*50)
    
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Character count statistics
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    print(f"Average chunk size: {sum(chunk_sizes) // len(chunk_sizes)} characters")
    print(f"Smallest chunk: {min(chunk_sizes)} characters")
    print(f"Largest chunk: {max(chunk_sizes)} characters")
    
    # Show first 3 chunks in detail
    print(f"\nFirst 3 chunks in detail:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source page: {chunk.metadata.get('page', 'unknown')}")
        print(f"Characters: {len(chunk.page_content)}")
        print(f"Content preview:")
        print(chunk.page_content[:300])
        print("...")
    
    # Show overlap in action between chunk 1 and chunk 2
    print(f"\n--- OVERLAP DEMONSTRATION ---")
    print(f"End of Chunk 1:")
    print(chunks[0].page_content[-200:])
    print(f"\nStart of Chunk 2:")
    print(chunks[1].page_content[:200])