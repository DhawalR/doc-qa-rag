import streamlit as st
import tempfile
import os
import hashlib
from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.retriever import get_retriever, retrieve_chunks
from src.generation import run_all_prompting_strategies
from src.evaluation import (
    evaluate_faithfulness,
    evaluate_retrieval_relevance,
    evaluate_answer_relevance
)

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Document Q&A RAG System",
    page_icon="📄",
    layout="wide"
)

# ---- HEADER ----
st.title("📄 Document Q&A RAG System")
st.markdown(
    "Upload a PDF, ask a question, get a grounded answer "
    "with full retrieval and evaluation transparency."
)
st.divider()


# ---- HELPER FUNCTIONS ----
def get_file_hash(file_bytes: bytes) -> str:
    """
    Creates a unique fingerprint for a file based on its contents.
    Same file always produces the same hash.
    Different file always produces a different hash.
    """
    return hashlib.md5(file_bytes).hexdigest()[:10]


def clean_text_for_display(text: str) -> str:
    """
    Cleans extracted PDF text for better display in Streamlit.
    Handles common math formula rendering issues from academic papers.
    """
    # Replace common LaTeX artifacts that look ugly in plain text
    replacements = {
        "∗": "*",
        "×": "x",
        "→": "->",
        "≤": "<=",
        "≥": ">=",
        "∑": "sum",
        "√": "sqrt",
        "∈": "in",
        "·": ".",
    }
    for symbol, replacement in replacements.items():
        text = text.replace(symbol, replacement)
    return text


# ---- CACHED FUNCTIONS ----
@st.cache_resource
def process_pdf(file_path: str, file_hash: str):
    """
    Loads, chunks and builds retriever from a PDF.
    Cached by file_hash so same file never gets reprocessed.
    file_hash is included as parameter so Streamlit's cache
    key includes it — guaranteeing different files get
    different cache entries.
    """
    docs = load_pdf(file_path)
    chunks = chunk_documents(docs)
    retriever = get_retriever(chunks)
    return retriever, chunks, len(docs)


# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Settings")

    strategy = st.selectbox(
        "Prompting Strategy",
        options=["zero_shot", "few_shot", "chain_of_thought"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    run_evaluation = st.checkbox(
        "Run Evaluation",
        value=True,
        help=(
            "Evaluate faithfulness, retrieval relevance and answer "
            "relevance. Uses additional API calls."
        )
    )

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Built with LangChain, ChromaDB and OpenAI. "
        "Retrieves relevant chunks using cosine similarity "
        "and generates grounded answers."
    )


# ---- FILE UPLOAD ----
st.header("1️⃣ Upload Your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Upload any text-based PDF document"
)

if uploaded_file is not None:

    # Get file bytes and generate fingerprint
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)

    # Show file info
    st.caption(
        f"File: {uploaded_file.name} — "
        f"Hash: {file_hash} — "
        f"Size: {len(file_bytes) / 1024:.1f} KB"
    )

    # Save to temp file so LangChain can read it by path
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    # Process PDF — cache key includes file_hash
    # Same file = same hash = instant load from cache
    # New file = new hash = fresh processing
    with st.spinner("Processing PDF..."):
        retriever, chunks, total_pages = process_pdf(
            tmp_path,
            file_hash
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", total_pages)
    col2.metric("Chunks", len(chunks))
    col3.metric("File Hash", file_hash)

    st.success("✅ PDF ready for questions")
    st.divider()

    # ---- QUESTION INPUT ----
    st.header("2️⃣ Ask a Question")

    question = st.text_input(
        "Your question",
        placeholder=(
            "What attention mechanisms does the Transformer use?"
        ),
        help="Ask anything about the uploaded document"
    )

    ask_button = st.button("🔍 Ask", type="primary")

    if ask_button and question.strip():

        # ---- RETRIEVAL ----
        with st.spinner("Retrieving relevant chunks..."):
            relevant_chunks = retrieve_chunks(retriever, question)

        # ---- GENERATION ----
        with st.spinner(
            f"Generating answer using "
            f"{strategy.replace('_', ' ')}..."
        ):
            all_results = run_all_prompting_strategies(
                question,
                relevant_chunks
            )
            answer = all_results[strategy]

        st.divider()

        # ---- DISPLAY ANSWER ----
        st.header("3️⃣ Answer")
        st.markdown(answer)

        st.divider()

        # ---- DISPLAY RETRIEVED CHUNKS ----
        st.header("4️⃣ Retrieved Chunks")
        st.markdown(
            f"Top **{len(relevant_chunks)} chunks** retrieved "
            f"by ChromaDB using cosine similarity."
        )

        for i, chunk in enumerate(relevant_chunks):
            page_num = chunk.metadata.get('page', 0) + 1
            clean_content = clean_text_for_display(
                chunk.page_content
            )
            with st.expander(
                f"Chunk {i+1} — Page {page_num} "
                f"({len(chunk.page_content)} characters)"
            ):
                st.text(clean_content)

        # ---- EVALUATION ----
        if run_evaluation:
            st.divider()
            st.header("5️⃣ Evaluation")

            with st.spinner("Running evaluation..."):
                faith_eval = evaluate_faithfulness(
                    answer,
                    relevant_chunks
                )
                retrieval_eval = evaluate_retrieval_relevance(
                    question,
                    relevant_chunks
                )
                answer_eval = evaluate_answer_relevance(
                    question,
                    answer
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Faithfulness",
                    value=f"{faith_eval['score']} / 5"
                )
                st.caption(faith_eval['reasoning'])

            with col2:
                st.metric(
                    label="Retrieval Relevance",
                    value=f"{retrieval_eval['average_score']} / 5"
                )
                for chunk_score in retrieval_eval['chunk_scores']:
                    st.caption(
                        f"Chunk {chunk_score['chunk_number']} "
                        f"(Page {chunk_score['page']}): "
                        f"{chunk_score['score']}/5"
                    )

            with col3:
                st.metric(
                    label="Answer Relevance",
                    value=f"{answer_eval['score']} / 5"
                )
                st.caption(answer_eval['reasoning'])

    elif ask_button and not question.strip():
        st.warning("⚠️ Please enter a question before clicking Ask.")

else:
    st.info(
        "Upload a PDF above to get started. "
        "The system will process it and prepare it for questions."
    )