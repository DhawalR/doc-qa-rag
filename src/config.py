import os
from dotenv import load_dotenv

#Read your .env file and loads variables intothe environment
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#fAil immediately if the key is missing - better that failing deep inside the pipeline
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. "
        "Please set it in the .env file."
        )  

#model settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

#chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

#retrieval settings
TOP_K_RESULTS = 5

