"""Locked read/write helpers for shared survey CSV score updates."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

DEFAULT_ENCODING = "utf-8-sig"


@contextmanager
def locked_csv_read(csv_path: str | Path):
    """Read CSV under an exclusive lock without writing on exit."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding=DEFAULT_ENCODING, newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield pd.read_csv(handle, dtype=str, keep_default_na=False)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def locked_csv(csv_path: str | Path):
    """Exclusive lock around one read-modify-write cycle."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r+", encoding=DEFAULT_ENCODING, newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            df = pd.read_csv(handle, dtype=str, keep_default_na=False)
            yield df
            handle.seek(0)
            handle.truncate()
            df.to_csv(handle, index=False, encoding=DEFAULT_ENCODING)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_columns_after(
    df: pd.DataFrame,
    anchor_column: str,
    columns: list[str],
) -> None:
    """Insert missing columns immediately after anchor_column, in order."""
    if anchor_column not in df.columns:
        raise ValueError(f"Column not found: {anchor_column}")

    insert_at = df.columns.get_loc(anchor_column) + 1
    for col in columns:
        if col in df.columns:
            continue
        df.insert(insert_at, col, "")
        insert_at += 1
