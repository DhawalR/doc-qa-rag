from langchain_openai import OpenAIEmbeddings
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL


def get_embedding_model():
    """
    Initializes and returns the OpenAI embedding model.
    Returns:
        OpenAIEmbeddings instance ready to generate embeddings
    """
    
    # OpenAIEmbeddings handles the API calls to OpenAI automatically
    # It sends text to OpenAI and gets back a list of numbers (the embedding)
    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,      # "text-embedding-3-small"
        openai_api_key=OPENAI_API_KEY
    )
    
    return embedding_model


def inspect_embedding(embedding_model) -> None:
    """
    Generates a sample embedding and inspects its properties.
    Helps us understand what embeddings actually look like.
    Args:
        embedding_model: OpenAIEmbeddings instance
    """
    
    print("\n" + "="*50)
    print("EMBEDDING INSPECTION REPORT")
    print("="*50)
    
    # Generate embedding for a sample sentence
    sample_text = "The Transformer uses self-attention mechanisms"
    embedding = embedding_model.embed_query(sample_text)
    
    print(f"\nSample text: '{sample_text}'")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Last 5 values: {embedding[-5:]}")
    
    # Demonstrate semantic similarity
    print("\n--- SEMANTIC SIMILARITY DEMONSTRATION ---")
    
    texts = [
        "The Transformer uses self-attention mechanisms",
        "The model relies on attention to process sequences",
        "I enjoy eating pizza on weekends"
    ]
    
    embeddings = embedding_model.embed_documents(texts)
    
    # Manual cosine similarity calculation
    def cosine_similarity(a, b):
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x ** 2 for x in a) ** 0.5
        magnitude_b = sum(x ** 2 for x in b) ** 0.5
        return dot_product / (magnitude_a * magnitude_b)
    
    sim_ab = cosine_similarity(embeddings[0], embeddings[1])
    sim_ac = cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"\nText A: '{texts[0]}'")
    print(f"Text B: '{texts[1]}'")
    print(f"Text C: '{texts[2]}'")
    print(f"\nSimilarity A vs B (same topic): {sim_ab:.4f}")
    print(f"Similarity A vs C (different topic): {sim_ac:.4f}")
    print(f"\nHigher score = more similar meaning")