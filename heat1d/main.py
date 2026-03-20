from __future__ import annotations

import argparse
from pathlib import Path

from .solver import run_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 1D transient heat conduction solver")
    parser.add_argument("input", type=Path, help="Path to the YAML input file")
    args = parser.parse_args()
    run_from_file(args.input)


if __name__ == "__main__":
    main()
