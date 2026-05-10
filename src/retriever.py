from src.vectorstore import get_or_build_vectorstore
from src.config import TOP_K_RESULTS


def get_retriever(chunks: list):
    """
    Builds or loads the vector store and returns a retriever.
    The retriever is the main interface for finding relevant chunks.
    
    Args:
        chunks: List of Document objects from chunking.py
        
    Returns:
        LangChain retriever instance
    """
    
    vectorstore = get_or_build_vectorstore(chunks)
    
    # as_retriever() wraps the vector store in a standard retriever interface
    # search_type="similarity" uses cosine similarity for ranking
    # search_kwargs={"k": TOP_K_RESULTS} returns top 5 most similar chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    
    return retriever


def retrieve_chunks(retriever, question: str) -> list:
    """
    Takes a question and returns the most relevant chunks.
    Args:
        retriever: LangChain retriever instance
        question: User's question as a string
    Returns:
        List of most relevant Document objects
    """
    
    print(f"\nRetrieving chunks for question:")
    print(f"'{question}'")
    
    # invoke() embeds the question and finds similar chunks automatically
    relevant_chunks = retriever.invoke(question)
    
    print(f"Retrieved {len(relevant_chunks)} relevant chunks")
    
    return relevant_chunks


def inspect_retrieved_chunks(chunks: list, question: str) -> None:
    """
    Displays retrieved chunks in a readable format for debugging.
    Args:
        chunks: List of retrieved Document objects
        question: The original question asked
    """
    
    print("\n" + "="*50)
    print("RETRIEVAL INSPECTION REPORT")
    print("="*50)
    
    print(f"\nQuestion: '{question}'")
    print(f"Top {len(chunks)} relevant chunks found:\n")
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"Source page: {chunk.metadata.get('page', 'unknown') + 1}")
        print(f"Characters: {len(chunk.page_content)}")
        print(f"Content:")
        print(chunk.page_content)
        print()