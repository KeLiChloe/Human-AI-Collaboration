#!/usr/bin/env python3
"""Evaluate Gender × SOI theories (pre + post, shuffled batches)."""

from __future__ import annotations

from eval_common.conditions import gender_soi
from eval_common.runner import main_for

if __name__ == "__main__":
    main_for(gender_soi())
