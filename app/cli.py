"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="educational-content-expander", description="Educators use this CLI tool to expand bullet points or short phrases into coherent sentences for lesson plans by generating next-word suggestions based on educational texts.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Educational Content Expander is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
