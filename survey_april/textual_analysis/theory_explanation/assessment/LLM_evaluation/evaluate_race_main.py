#!/usr/bin/env python3
"""Evaluate Race × Main-effects theories (pre + post, shuffled batches)."""

from __future__ import annotations

from eval_common.conditions import race_main
from eval_common.runner import main_for

if __name__ == "__main__":
    main_for(race_main())
