import streamlit as st
import pandas as pd
import os
import tempfile
from src.config_search import run_config_search
from src.visualize_results import generate_all_visualizations

st.set_page_config(
    page_title="RAG Configuration Search",
    page_icon="⚙️",
    layout="wide"
)

# ── Minimal clean styling ─────────────────────────────────────────────────
st.markdown("""
<style>
    .config-card {
        border-left: 3px solid #2563eb;
        padding: 10px 14px;
        background: #f8fafc;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .config-card.best  { border-left-color: #15803d; background: #f0fdf4; }
    .config-card.worst { border-left-color: #b91c1c; background: #fef2f2; }
    .config-label {
        font-weight: 600;
        font-size: 13px;
        color: #0f172a;
    }
    .config-scores {
        font-size: 12px;
        color: #475569;
        margin-top: 3px;
    }
    .section-rule {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.title("RAG Configuration Search")
st.markdown(
    "Systematically test chunk size and retrieval depth combinations "
    "to identify the optimal configuration for your document."
)
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "PDF document",
        type="pdf",
        help="Upload the document to test configurations against"
    )

    cache_mode = st.radio(
        "Cache mode",
        options=["false", "true", "clear"],
        format_func=lambda x: {
            "false": "No cache — fresh every run",
            "true":  "Persist to disk — free reruns",
            "clear": "Clear existing cache and rebuild"
        }[x],
        index=0,
        help=(
            "false: runs in memory, nothing saved\n"
            "true: saves configs to disk, reuses on next run\n"
            "clear: deletes this PDF's cache and rebuilds"
        )
    )

    run_button = st.button(
        "Run Search",
        type="primary",
        disabled=uploaded_file is None
    )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("**Search space**")
    st.markdown(
        "Chunk sizes: `500`, `1000`, `1500`  \n"
        "Overlap: `200` (fixed)  \n"
        "Top-K: `3`, `5`, `7`  \n"
        "Total combinations: **9**"
    )

# ── Run search ────────────────────────────────────────────────────────────
if run_button and uploaded_file:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner("Running configuration search. This may take several minutes..."):
        results = run_config_search(
            pdf_path=tmp_path,
            cache_mode=cache_mode
        )
        generate_all_visualizations()

    st.success(
        f"Search complete. "
        f"{len(results)} configurations tested."
    )
    st.session_state["results"] = results

# ── Results ───────────────────────────────────────────────────────────────
csv_path = "results/config_search_results.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    # ── Summary metrics ───────────────────────────────────────────────
    st.header("Summary")
    best  = df.iloc[0]
    worst = df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Best configuration",
        f"chunk={int(best['chunk_size'])}, k={int(best['top_k'])}"
    )
    c2.metric("Best overall score",  f"{best['overall']} / 5")
    c3.metric("Worst overall score", f"{worst['overall']} / 5")
    c4.metric("Configurations tested", len(df))

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Configuration cards ───────────────────────────────────────────
    st.header("📁 Configurations")
    st.caption(
        "Each card shows the summary scores. "
        "Click 'Details' to expand the full evaluation breakdown."
    )
    st.markdown("")

    for row_start in range(0, len(df), 3):
        cols = st.columns(3)
        for col_idx, df_idx in enumerate(
            range(row_start, min(row_start + 3, len(df)))
        ):
            row = df.iloc[df_idx]
            rank = int(row["rank"])

            if rank == 1:
                card_class = "best"
            elif rank == len(df):
                card_class = "worst"
            else:
                card_class = ""

            with cols[col_idx]:
                st.markdown(
                    f"""
                    <div class="config-card {card_class}">
                        <div class="config-label">
                            #{rank} &nbsp;
                            chunk={int(row['chunk_size'])},
                            overlap={int(row['overlap'])},
                            k={int(row['top_k'])}
                        </div>
                        <div class="config-scores">
                            Overall: <b>{row['overall']}/5</b>
                            &nbsp;&nbsp;
                            F: {row['avg_faithfulness']}
                            &nbsp;
                            R: {row['avg_retrieval']}
                            &nbsp;
                            A: {row['avg_relevance']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander("Details"):
                    st.markdown(
                        f"**Configuration:** `{row['config']}`"
                    )
                    st.markdown(
                        f"**Chunks created:** {int(row['chunk_count'])}"
                    )
                    st.markdown(
                        f"**Run timestamp:** {row['timestamp']}"
                    )
                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    m1.metric(
                        "Faithfulness",
                        f"{row['avg_faithfulness']} / 5"
                    )
                    m2.metric(
                        "Retrieval",
                        f"{row['avg_retrieval']} / 5"
                    )
                    m3.metric(
                        "Answer Relevance",
                        f"{row['avg_relevance']} / 5"
                    )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────
    st.header("Visualizations")

    chart_files = {
        "Heatmap": "results/heatmap_overall.png",
        "Metrics by Configuration": "results/bar_all_metrics.png",
        "Score vs Chunk Size": "results/line_chunk_vs_score.png",
        "Ranking Table": "results/ranking_table.png",
    }

    tabs = st.tabs(list(chart_files.keys()))
    for tab, (title, img_path) in zip(tabs, chart_files.items()):
        with tab:
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.caption(
                    "Chart not found. "
                    "Run the search to generate visualizations."
                )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Full data table ───────────────────────────────────────────────
    st.header("Full Results")
    st.dataframe(
        df[[
            "rank", "config", "chunk_count",
            "avg_faithfulness", "avg_retrieval",
            "avg_relevance", "overall"
        ]].rename(columns={
            "rank":             "Rank",
            "config":           "Configuration",
            "chunk_count":      "Chunks",
            "avg_faithfulness": "Faithfulness",
            "avg_retrieval":    "Retrieval",
            "avg_relevance":    "Relevance",
            "overall":          "Overall",
        }).style.background_gradient(
            subset=["Overall"],
            cmap="YlGn"
        ),
        use_container_width=True
    )

    with open(csv_path, "rb") as f:
        st.download_button(
            label="Download CSV",
            data=f,
            file_name="config_search_results.csv",
            mime="text/csv"
        )

else:
    st.info(
        "No results available. "
        "Upload a PDF in the sidebar and run the search."
    )