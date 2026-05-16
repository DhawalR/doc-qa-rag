import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

RESULTS_DIR = "results"


def load_results() -> pd.DataFrame:
    """Loads config search results from CSV."""
    csv_path = os.path.join(RESULTS_DIR, "config_search_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "No results found. Run the config search first."
        )
    return pd.read_csv(csv_path)


def plot_heatmap(df: pd.DataFrame):
    """
    Heatmap of overall score by chunk size vs top-k.
    Best visual for showing which combination wins.
    """
    pivot = df.pivot_table(
        values="overall",
        index="chunk_size",
        columns="top_k",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        linewidths=0.5,
        linecolor="white",
        vmin=1, vmax=5,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_title(
        "Overall Score — Chunk Size vs Top-K",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Top-K Retrieved Chunks", fontsize=11)
    ax.set_ylabel("Chunk Size (characters)", fontsize=11)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "heatmap_overall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_metric_bars(df: pd.DataFrame):
    """
    Grouped bar chart showing all three metrics
    side by side for each configuration.
    """
    configs = df["config"].tolist()
    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, df["avg_faithfulness"],
                   width, label="Faithfulness",
                   color="#2563eb", alpha=0.85)
    bars2 = ax.bar(x, df["avg_retrieval"],
                   width, label="Retrieval Relevance",
                   color="#16a34a", alpha=0.85)
    bars3 = ax.bar(x + width, df["avg_relevance"],
                   width, label="Answer Relevance",
                   color="#dc2626", alpha=0.85)

    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_ylabel("Score (out of 5)", fontsize=11)
    ax.set_title(
        "Evaluation Metrics Across Configurations",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45,
                       ha="right", fontsize=8)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=10)
    ax.axhline(y=4.0, color="gray",
               linestyle="--", alpha=0.5, label="Threshold")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "bar_all_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_line_chunk_vs_score(df: pd.DataFrame):
    """
    Line chart showing how overall score changes
    with chunk size for each k value.
    Reveals the sweet spot.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {3: "#2563eb", 5: "#16a34a", 7: "#dc2626"}

    for k in sorted(df["top_k"].unique()):
        subset = df[df["top_k"] == k].sort_values("chunk_size")
        ax.plot(
            subset["chunk_size"],
            subset["overall"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=f"k={k}",
            color=colors.get(k, "gray")
        )
        # Annotate each point with its score
        for _, row in subset.iterrows():
            ax.annotate(
                f"{row['overall']}",
                (row["chunk_size"], row["overall"]),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=8,
                ha="center"
            )

    ax.set_xlabel("Chunk Size (characters)", fontsize=11)
    ax.set_ylabel("Overall Score (out of 5)", fontsize=11)
    ax.set_title(
        "Overall Score vs Chunk Size for Each Top-K",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xticks(sorted(df["chunk_size"].unique()))
    ax.set_ylim(0, 5.5)
    ax.legend(title="Top-K", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "line_chunk_vs_score.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_ranking_table(df: pd.DataFrame):
    """
    Clean ranking table saved as an image.
    Splits configuration into separate columns for readability.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")

    top9 = df.head(9)[[
        "rank", "chunk_size", "overlap", "top_k",
        "chunk_count", "avg_faithfulness",
        "avg_retrieval", "avg_relevance", "overall"
    ]].copy()

    # Convert to int where applicable for clean display
    top9["rank"]       = top9["rank"].astype(int)
    top9["chunk_size"] = top9["chunk_size"].astype(int)
    top9["overlap"]    = top9["overlap"].astype(int)
    top9["top_k"]      = top9["top_k"].astype(int)
    top9["chunk_count"]= top9["chunk_count"].astype(int)

    top9.columns = [
        "Rank", "Chunk", "Overlap", "K",
        "Chunks", "Faithfulness",
        "Retrieval", "Relevance", "Overall"
    ]

    table = ax.table(
        cellText=top9.values,
        colLabels=top9.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)

    # Column widths — narrow for numbers, wider for scores
    col_widths = {
        0: 0.06,   # Rank
        1: 0.08,   # Chunk
        2: 0.08,   # Overlap
        3: 0.06,   # K
        4: 0.08,   # Chunks
        5: 0.12,   # Faithfulness
        6: 0.10,   # Retrieval
        7: 0.10,   # Relevance
        8: 0.10,   # Overall
    }
    for col, width in col_widths.items():
        table.column_width_set = True
        for row in range(len(top9) + 1):
            table[row, col].set_width(width)

    # Style header row
    for j in range(len(top9.columns)):
        table[0, j].set_facecolor("#0f172a")
        table[0, j].set_text_props(
            color="white",
            fontweight="bold"
        )

    # Highlight best row green
    for j in range(len(top9.columns)):
        table[1, j].set_facecolor("#dcfce7")

    # Highlight worst row red
    for j in range(len(top9.columns)):
        table[len(top9), j].set_facecolor("#fee2e2")

    ax.set_title(
        "Configuration Search — Full Rankings",
        fontsize=13,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ranking_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def generate_all_visualizations():
    """Generates all four charts from saved CSV results."""
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50 + "\n")

    df = load_results()

    print(f"Loaded {len(df)} configurations from CSV\n")

    plot_heatmap(df)
    plot_metric_bars(df)
    plot_line_chunk_vs_score(df)
    plot_ranking_table(df)

    print(f"\nAll charts saved to results/")
    print(f"Files:")
    print(f"  results/heatmap_overall.png")
    print(f"  results/bar_all_metrics.png")
    print(f"  results/line_chunk_vs_score.png")
    print(f"  results/ranking_table.png")