from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.retriever import get_retriever, retrieve_chunks
from src.generation import run_all_prompting_strategies, inspect_answers

# ---- SETTINGS ----
PDF_PATH = "data/attention_paper.pdf"
QUESTION = "What attention mechanisms does the Transformer use?"

# ---- PIPELINE ----
print("\n>>> Step 1: Loading PDF")
docs = load_pdf(PDF_PATH)

print("\n>>> Step 2: Chunking")
chunks = chunk_documents(docs)

print("\n>>> Step 3: Retriever")
retriever = get_retriever(chunks)

print("\n>>> Step 4: Retrieving relevant chunks")
relevant_chunks = retrieve_chunks(retriever, QUESTION)

print("\n>>> Step 5: Generating answers")
results = run_all_prompting_strategies(QUESTION, relevant_chunks)
inspect_answers(results, QUESTION)