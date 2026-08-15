#!/usr/bin/env bash
# Fill API-derived columns in All_Participants_All_Questions.csv.
# Safe to re-run: each step scans all rows and only fills missing metrics.
#
# Prerequisites:
#   - OPENAI_API_KEY set in the environment
#   - New sample rows already have manual survey content (Q1–Q17 text, rank/sign, meta, etc.)
#
# Usage:
#   ./fill_missing_columns.sh
#   MODEL=gpt-5.5 ./fill_missing_columns.sh
#
# Fills:
#   1. Q5 diagram metrics (Race + Gender)
#   2. Q13 diagram metrics (Race + Gender)
#   3. post-ML LLM_refined / LLM_status / LLM_uncertainty_note (4 tasks)
#   4. Q4 and Q12 theory quality dimension scores (Race + Gender)
#   5. Q4 and Q12 Overall Quality Score (mean of 5 dimensions)
#
# Does NOT fill (intentionally left blank for GenAI samples):
#   - demographics / background questions
#   - rank/sign columns for non-Top5 features

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="${ROOT}/All_Participants_All_Questions.csv"
MODEL="${MODEL:-gpt-5.5}"
PYTHON="${PYTHON:-python3}"

DIAGRAM_PRE="${ROOT}/textual_analysis/diagram/pre-data"
DIAGRAM_POST="${ROOT}/textual_analysis/diagram/post-data"
ASSESS="${ROOT}/textual_analysis/theory_explanation/assessment/LLM_evaluation"
INTEGRATE="${ROOT}/textual_analysis/theory_explanation/integrate_pre_and_post"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  exit 1
fi

if [[ ! -f "$CSV" ]]; then
  echo "ERROR: CSV not found: $CSV" >&2
  exit 1
fi

run_step() {
  local label="$1"
  shift
  echo ""
  echo "======================================================================"
  echo "$label"
  echo "======================================================================"
  "$@"
}

echo "CSV:   $CSV"
echo "Model: $MODEL"

run_step "Step 1/7: Q Race.5 diagram metrics" \
  "$PYTHON" "${DIAGRAM_PRE}/API_diagram_race(Survey_Q5).py"

run_step "Step 2/7: Q Gender.5 diagram metrics" \
  "$PYTHON" "${DIAGRAM_PRE}/API_diagram_gender(Survey_Q5).py"

run_step "Step 3/7: Q Race.13 diagram metrics" \
  "$PYTHON" "${DIAGRAM_POST}/API_diagram_race(Survey_Q13).py"

run_step "Step 4/7: Q Gender.13 diagram metrics" \
  "$PYTHON" "${DIAGRAM_POST}/API_diagram_gender(Survey_Q13).py"

run_step "Step 5/7: post-ML LLM_refined (race/gender main effects + SOI)" \
  "$PYTHON" "${INTEGRATE}/Main_LLM_integrate_pre_and_post_theory.py" \
  --model "$MODEL"

run_step "Step 6/7: theory quality dimension scores (Q4 + Q12 LLM_refined)" \
  bash -c "
    set -euo pipefail
    '$PYTHON' '${ASSESS}/API_theory_main_race(Survey_Q4Q12).py'
    '$PYTHON' '${ASSESS}/API_theory_main_gender(Survey_Q4Q12).py'
  "

run_step "Step 7/7: Q4 and Q12 Overall Quality Score (mean of 5 dimensions)" \
  "$PYTHON" "${ASSESS}/compute_assessment_overall_quality(Survey_Q4Q12).py"

echo ""
echo "Done. Updated: $CSV"
