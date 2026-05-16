import argparse
from src.config_search import run_config_search
from src.visualize_results import generate_all_visualizations


def main():
    parser = argparse.ArgumentParser(
        description=(
            "RAG Configuration Search\n"
            "Tests different chunk size and top-k combinations\n"
            "to find the optimal settings for your document."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--pdf",
        type=str,
        default="data/text_classification.pdf",
        help="Path to the PDF file to test against"
    )

    parser.add_argument(
        "--cache",
        type=str,
        default="false",
        choices=["true", "false", "clear"],
        help=(
            "Cache behaviour:\n"
            "  false  run in memory, nothing saved to disk (default)\n"
            "  true   save configs to disk, reuse on next run\n"
            "  clear  delete this PDF's cached configs and rebuild"
        )
    )

    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip chart generation after search completes"
    )

    args = parser.parse_args()

    results = run_config_search(
        pdf_path=args.pdf,
        cache_mode=args.cache
    )

    if not args.no_visualize:
        generate_all_visualizations()

    return results


if __name__ == "__main__":
    main()