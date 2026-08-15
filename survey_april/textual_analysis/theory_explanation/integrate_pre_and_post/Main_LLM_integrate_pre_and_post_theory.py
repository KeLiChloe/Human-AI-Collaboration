"""
Run all four post-ML theory reconstruction scripts in sequence.

For targeted runs, use the individual scripts instead:
    - LLM_integrate_race_main_effects.py   (Q Race.12)
    - LLM_integrate_race_soi.py            (Q Race.15)
    - LLM_integrate_gender_main_effects.py   (Q Gender.12)
    - LLM_integrate_gender_soi.py            (Q Gender.15)

Example:
    python LLM_integrate_pre_and_post_theory.py --model "gpt-5.5"
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "LLM_integrate_race_main_effects.py",
    "LLM_integrate_race_soi.py",
    "LLM_integrate_gender_main_effects.py",
    "LLM_integrate_gender_soi.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    extra_args = sys.argv[1:]

    for name in SCRIPTS:
        script = here / name
        print(f"\n{'=' * 72}\nRunning {name}\n{'=' * 72}")
        result = subprocess.run(
            [sys.executable, str(script), *extra_args],
            check=False,
        )
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
