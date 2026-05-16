import os
import csv
import shutil
import hashlib
import itertools
from datetime import datetime
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion import load_pdf
from src.embeddings import get_embedding_model
from src.generation import run_all_prompting_strategies
from src.evaluation import (
    evaluate_faithfulness,
    evaluate_retrieval_relevance,
    evaluate_answer_relevance
)

# ── Configuration grid ────────────────────────────────────────────────────
CHUNK_SIZES    = [500, 1000, 1500]
CHUNK_OVERLAPS = [200]          # fixed — only varying chunk size and k
TOP_K_VALUES   = [3, 5, 7]

TEST_QUESTIONS = [
    "What is the main contribution of this paper?",
    "What are the limitations of the proposed approach?",
    "What datasets or experiments were used to validate the results?",
]

CACHE_BASE = "chroma_configs"
RESULTS_DIR = "results"


def get_pdf_hash(pdf_path: str) -> str:
    """Generate a short hash fingerprint for the PDF file."""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:10]


def get_config_path(pdf_hash: str, chunk_size: int,
                    overlap: int, k: int) -> str:
    """Returns the disk path for a specific configuration's ChromaDB."""
    return os.path.join(CACHE_BASE, pdf_hash,
                        f"{chunk_size}_{overlap}_{k}")


def build_config_vectorstore(pdf_path: str, chunk_size: int,
                              overlap: int, persist_path: str = None):
    """
    Loads PDF, chunks with given settings, builds ChromaDB.
    If persist_path is None — ephemeral (in memory only).
    If persist_path is given — saves to disk.
    """
    docs = load_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_path  # None = ephemeral
    )

    return vectorstore, len(chunks)


def load_config_vectorstore(persist_path: str):
    """Loads an existing ChromaDB from disk."""
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=persist_path,
        embedding_function=embedding_model
    )


def evaluate_configuration(vectorstore, k: int,
                            questions: list) -> dict:
    """
    Runs all test questions against a configuration
    and returns averaged evaluation scores.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    all_faithfulness = []
    all_retrieval    = []
    all_relevance    = []
    question_results = []

    for question in questions:
        # Retrieve
        relevant_chunks = retriever.invoke(question)

        # Generate using zero-shot only for cost efficiency
        results = run_all_prompting_strategies(question, relevant_chunks)
        answer  = results["zero_shot"]

        # Evaluate
        faith    = evaluate_faithfulness(answer, relevant_chunks)
        retrieval= evaluate_retrieval_relevance(question, relevant_chunks)
        relevance= evaluate_answer_relevance(question, answer)

        all_faithfulness.append(faith["score"])
        all_retrieval.append(retrieval["average_score"])
        all_relevance.append(relevance["score"])

        question_results.append({
            "question":          question,
            "answer":            answer,
            "faithfulness":      faith["score"],
            "faithfulness_reason": faith["reasoning"],
            "retrieval":         retrieval["average_score"],
            "relevance":         relevance["score"],
            "relevance_reason":  relevance["reasoning"],
        })

    return {
        "avg_faithfulness": round(sum(all_faithfulness) /
                                  len(all_faithfulness), 2),
        "avg_retrieval":    round(sum(all_retrieval) /
                                  len(all_retrieval), 2),
        "avg_relevance":    round(sum(all_relevance) /
                                  len(all_relevance), 2),
        "overall":          round((sum(all_faithfulness) +
                                   sum(all_retrieval) +
                                   sum(all_relevance)) /
                                  (3 * len(questions)), 2),
        "question_results": question_results,
    }


def clear_cache(pdf_path: str):
    """Deletes cached configs for this specific PDF only."""
    pdf_hash   = get_pdf_hash(pdf_path)
    cache_path = os.path.join(CACHE_BASE, pdf_hash)

    if not os.path.exists(cache_path):
        print(f"No cache found for {os.path.basename(pdf_path)}.")
        return

    # Count configs and estimate size
    configs = os.listdir(cache_path)
    size_mb = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(cache_path)
        for f in files
    ) / (1024 * 1024)

    print(f"Clearing cache for {os.path.basename(pdf_path)}...")
    print(f"  Configs found: {len(configs)}")
    print(f"  Size on disk:  {size_mb:.1f} MB")

    shutil.rmtree(cache_path)
    print(f"Cache cleared successfully.")


def run_config_search(pdf_path: str,
                      cache_mode: str = "false") -> list:
    """
    Main entry point for configuration search.

    cache_mode:
        "false" — ephemeral, nothing saved to disk (default)
        "true"  — save configs to disk, reuse on next run
        "clear" — delete existing cache, rebuild fresh and save
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Handle clear mode
    if cache_mode == "clear":
        clear_cache(pdf_path)
        cache_mode = "true"  # rebuild and save after clearing

    pdf_hash     = get_pdf_hash(pdf_path)
    pdf_name     = os.path.basename(pdf_path)
    combinations = list(itertools.product(
        CHUNK_SIZES, CHUNK_OVERLAPS, TOP_K_VALUES
    ))

    print(f"\n{'='*55}")
    print(f"CONFIGURATION SEARCH")
    print(f"{'='*55}")
    print(f"PDF:          {pdf_name}")
    print(f"Cache mode:   {cache_mode}")
    print(f"Combinations: {len(combinations)}")
    print(f"Questions:    {len(TEST_QUESTIONS)}")
    print(f"{'='*55}\n")

    all_results = []

    for i, (chunk_size, overlap, k) in enumerate(combinations):
        config_label = f"chunk={chunk_size}, overlap={overlap}, k={k}"
        print(f"[{i+1}/{len(combinations)}] Testing {config_label}...")

        # Determine persist path
        if cache_mode == "true":
            persist_path = get_config_path(
                pdf_hash, chunk_size, overlap, k
            )
            os.makedirs(
                os.path.dirname(persist_path),
                exist_ok=True
            )
        else:
            persist_path = None  # ephemeral

        # Build or load vectorstore
        if persist_path and os.path.exists(persist_path) \
                and os.listdir(persist_path):
            print(f"  Loading from cache...")
            vectorstore = load_config_vectorstore(persist_path)
            # Get chunk count from collection
            chunk_count = vectorstore._collection.count()
        else:
            print(f"  Building fresh embeddings...")
            vectorstore, chunk_count = build_config_vectorstore(
                pdf_path, chunk_size, overlap, persist_path
            )

        print(f"  Chunks: {chunk_count} | Evaluating...")

        # Evaluate
        scores = evaluate_configuration(
            vectorstore, k, TEST_QUESTIONS
        )

        result = {
            "config":           config_label,
            "chunk_size":       chunk_size,
            "overlap":          overlap,
            "top_k":            k,
            "chunk_count":      chunk_count,
            "avg_faithfulness": scores["avg_faithfulness"],
            "avg_retrieval":    scores["avg_retrieval"],
            "avg_relevance":    scores["avg_relevance"],
            "overall":          scores["overall"],
            "question_results": scores["question_results"],
            "timestamp":        datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"),
        }

        all_results.append(result)
        print(f"  Overall score: {scores['overall']}/5\n")

    # Sort by overall score descending
    all_results.sort(key=lambda x: x["overall"], reverse=True)

    # Save CSV
    save_results_csv(all_results, pdf_name)

    print(f"\n{'='*55}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*55}")
    print(f"Best config:  {all_results[0]['config']}")
    print(f"Best score:   {all_results[0]['overall']}/5")
    print(f"Worst config: {all_results[-1]['config']}")
    print(f"Worst score:  {all_results[-1]['overall']}/5")
    print(f"Results saved to results/config_search_results.csv")

    return all_results


def save_results_csv(results: list, pdf_name: str):
    """Saves summary results to CSV."""
    csv_path = os.path.join(RESULTS_DIR, "config_search_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "config", "chunk_size", "overlap",
            "top_k", "chunk_count", "avg_faithfulness",
            "avg_retrieval", "avg_relevance", "overall", "timestamp"
        ])
        writer.writeheader()
        for rank, result in enumerate(results, 1):
            writer.writerow({
                "rank":             rank,
                "config":           result["config"],
                "chunk_size":       result["chunk_size"],
                "overlap":          result["overlap"],
                "top_k":            result["top_k"],
                "chunk_count":      result["chunk_count"],
                "avg_faithfulness": result["avg_faithfulness"],
                "avg_retrieval":    result["avg_retrieval"],
                "avg_relevance":    result["avg_relevance"],
                "overall":          result["overall"],
                "timestamp":        result["timestamp"],
            })