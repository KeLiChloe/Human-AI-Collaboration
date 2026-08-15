"""
Code Q Race.13 post-ML diagrams into structure metrics.

Writes columns in-place into All_Participants_All_Questions.csv:
  - Q Race.13 Number of paths
  - Q Race.13 Maximum path length
  - Q Race.13 Number of latent variables
  - Q Race.13 Coding reasoning

Empty / no-diagram / ill-defined responses are coded as -1.
Coding prompt is phase-blind (same as Q5); only CSV columns differ.

Do NOT run this concurrently with other diagram coders — they rewrite the same CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagram_coding_common import run_diagram_coding  # noqa: E402

TARGET_COLUMN = "Q Race.13 post-ML diagram (main effects)"
METRIC_COLUMNS = [
    "Q Race.13 Number of paths",
    "Q Race.13 Maximum path length",
    "Q Race.13 Number of latent variables",
    "Q Race.13 Coding reasoning",
]


def main() -> None:
    run_diagram_coding(
        target_column=TARGET_COLUMN,
        metric_columns=METRIC_COLUMNS,
        outcome="race",
        progress_desc="Q Race.13 diagram coding",
    )


if __name__ == "__main__":
    main()
