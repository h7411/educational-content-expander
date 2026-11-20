"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core import build_educational_model, expand_text, suggest_next_words


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="educational-content-expander",
        description=(
            "Educators use this CLI tool to expand bullet points or short phrases "
            "into coherent sentences for lesson plans by generating next-word suggestions "
            "based on educational texts."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Initial smoke test command
    sub.add_parser("hello", help="check that everything is wired up")

    # Expand command for milestones 3 & 4 and stretch ideas
    expand_parser = sub.add_parser(
        "expand",
        help="expand a bullet point or short phrase into a full educational sentence",
    )
    expand_parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="short phrase or bullet point to expand (e.g. 'Photosynthesis is')",
    )
    expand_parser.add_argument(
        "-p",
        "--prompt-flag",
        dest="prompt_flag",
        default=None,
        help="alternative flag for prompt text",
    )
    expand_parser.add_argument(
        "-m",
        "--max-words",
        type=int,
        default=15,
        help="maximum number of new words to append (default: 15)",
    )
    expand_parser.add_argument(
        "-n",
        "--order",
        type=int,
        default=3,
        help="n-gram order (default: 3 for trigram)",
    )
    expand_parser.add_argument(
        "-k",
        "--smoothing",
        type=float,
        default=1.0,
        help="add-k smoothing parameter (default: 1.0 for Laplace)",
    )
    expand_parser.add_argument(
        "-s",
        "--suggest",
        action="store_true",
        help="display top candidate next-word suggestions with probabilities",
    )
    expand_parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        default=None,
        help="path to text file with multiple bullet points (one per line) to expand",
    )
    expand_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="path to text file to save generated expansions",
    )
    expand_parser.add_argument(
        "--interactive",
        action="store_true",
        help="interactive CLI session for educators to expand phrases",
    )

    return parser


def handle_expand(args: argparse.Namespace) -> None:
    model = build_educational_model(n=args.order, k=args.smoothing)
    output_lines: list[str] = []

    # Batch processing from input file
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
            return

        lines = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]
        for prompt in lines:
            expanded = expand_text(model, prompt, max_new_words=args.max_words)
            output_lines.append(f"Prompt: {prompt}\nExpanded: {expanded}\n")
            print(f"Expanded: {expanded}")

    # Single prompt
    else:
        prompt = args.prompt or args.prompt_flag
        if not prompt and args.interactive:
            prompt = input("Enter educational phrase or bullet point: ").strip()

        if not prompt:
            prompt = "Photosynthesis is"

        expanded = expand_text(model, prompt, max_new_words=args.max_words)
        output_lines.append(f"Prompt: {prompt}")
        output_lines.append(f"Expanded: {expanded}")
        print(f"Expanded: {expanded}")

        if args.suggest:
            suggestions = suggest_next_words(model, prompt, top_k=5)
            print("\nNext-word suggestions:")
            for word, prob in suggestions:
                print(f"  - '{word}': {prob * 100:.2f}%")
            output_lines.append("\nSuggestions: " + ", ".join(f"{w} ({p*100:.1f}%)" for w, p in suggestions))

    # Save to output file if requested
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(output_lines) + "\n")
        print(f"\nSaved expanded lesson plan content to: {args.output_file}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Educational Content Expander is ready. Open README.md for your milestones.")
    elif args.command == "expand":
        handle_expand(args)


if __name__ == "__main__":
    main()
