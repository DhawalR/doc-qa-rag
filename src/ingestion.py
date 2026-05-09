from langchain_community.document_loaders import PyPDFLoader
from src.config import OPENAI_API_KEY
import os

def load_pdf(file_path: str)-> list:
    """ Each Document object represents one page of the PDF.
    Args:file_path: Path to the PDF file
    Returns:List of LangChain Document objects"""

    # Check if file exists before trying to load it
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at path: {file_path}")
    
    print(f"Loading PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} pages")
    return documents


def inspect_documents(documents: list)-> None:
    """
    Prints a summary of loaded documents for debugging and understanding.
    Args:documents: List of Document objects from load_pdf()
    """
    print("\n" + "="*50)
    print("DOCUMENT INSPECTION REPORT")
    print("="*50)
    
    print(f"\nTotal pages loaded: {len(documents)}")
    
    # Show metadata from first page
    print(f"\nMetadata from page 1:")
    print(documents[0].metadata)
    
    # Show first 500 characters of first page
    print(f"\nFirst 500 characters of page 1:")
    print("-"*40)
    print(documents[0].page_content[:500])
    print("-"*40)
    
    # Show character count per page
    print(f"\nCharacter count per page:")
    for i, doc in enumerate(documents):
        print(f"  Page {i+1}: {len(doc.page_content)} characters")