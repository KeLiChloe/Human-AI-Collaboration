import pandas as pd
from pathlib import Path

INPUT_CSV = Path(__file__).resolve().parents[3] / "All_Participants_All_Questions.csv"


def normalize_header(text: str) -> str:
    return " ".join(str(text).split()).strip().strip('"')


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Resolve a real column name from candidate names with robust matching.
    """
    for col in candidates:
        if col in df.columns:
            return col

    normalized_to_real = {normalize_header(c): c for c in df.columns}
    for col in candidates:
        norm = normalize_header(col)
        if norm in normalized_to_real:
            return normalized_to_real[norm]

    raise ValueError(f"Could not resolve any of columns: {candidates}")


def upsert_overall_column(
    df: pd.DataFrame,
    dimension_cols: list[str],
    output_col: str,
    insert_after_col: str,
) -> pd.DataFrame:
    """
    Compute row-wise mean of 5 dimensions and write into output column.
    """
    score_frame = df[dimension_cols].apply(pd.to_numeric, errors="coerce")
    overall = score_frame.mean(axis=1)

    if output_col not in df.columns:
        insert_at = df.columns.get_loc(insert_after_col) + 1
        df.insert(insert_at, output_col, overall)
    else:
        df[output_col] = overall

    return df


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    blocks = [
        {
            "name": "Q Race.4",
            "dims": [
                ["Q Race.4 Clarity and Coherence"],
                ["Q Race.4 Causal Reasoning"],
                ["Q Race.4 Theoretical Depth"],
                ["Q Race.4 Creativity"],
                ["Q Race.4 Persuasiveness"],
            ],
            "insert_after": ["Q Race.4 Persuasiveness"],
            "output": "Q Race.4 Overall Quality Score",
        },
        {
            "name": "Q Race.12",
            "dims": [
                ["Q Race.12 Updated Theory Clarity and Coherence", "Q Race.12 Clarity and Coherence"],
                ["Q Race.12 Updated Theory Causal Reasoning", "Q Race.12 Causal Reasoning"],
                ["Q Race.12 Updated Theory Theoretical Depth", "Q Race.12 Theoretical Depth"],
                ["Q Race.12 Updated Theory Creativity", "Q Race.12 Creativity"],
                ["Q Race.12 Updated Theory Persuasiveness", "Q Race.12 Persuasiveness"],
            ],
            "insert_after": ["Q Race.12 Updated Theory Persuasiveness", "Q Race.12 Persuasiveness"],
            "output": "Q Race.12 Overall Quality Score",
        },
        {
            "name": "Q Gender.4",
            "dims": [
                ["Q Gender.4 Clarity and Coherence"],
                ["Q Gender.4 Causal Reasoning"],
                ["Q Gender.4 Theoretical Depth"],
                ["Q Gender.4 Creativity"],
                ["Q Gender.4 Persuasiveness"],
            ],
            "insert_after": ["Q Gender.4 Persuasiveness"],
            "output": "Q Gender.4 Overall Quality Score",
        },
        {
            "name": "Q Gender.12",
            "dims": [
                ["Q Gender.12 Clarity and Coherence", "Q Gender.12 Updated Theory Clarity and Coherence"],
                ["Q Gender.12 Causal Reasoning", "Q Gender.12 Updated Theory Causal Reasoning"],
                ["Q Gender.12 Theoretical Depth", "Q Gender.12 Updated Theory Theoretical Depth"],
                ["Q Gender.12 Creativity", "Q Gender.12 Updated Theory Creativity"],
                ["Q Gender.12 Persuasiveness", "Q Gender.12 Updated Theory Persuasiveness"],
            ],
            "insert_after": ["Q Gender.12 Persuasiveness", "Q Gender.12 Updated Theory Persuasiveness"],
            "output": "Q Gender.12 Overall Quality Score",
        },
    ]

    print("Checking Q4/Q12 assessment dimensions and computing overall quality...")
    for block in blocks:
        resolved_dims = [resolve_column(df, candidates) for candidates in block["dims"]]
        insert_after = resolve_column(df, block["insert_after"])

        if len(resolved_dims) != 5:
            raise ValueError(f"{block['name']} does not have 5 dimensions: {resolved_dims}")

        df = upsert_overall_column(
            df=df,
            dimension_cols=resolved_dims,
            output_col=block["output"],
            insert_after_col=insert_after,
        )
        print(f"  - {block['name']}: 5 dimensions OK -> wrote '{block['output']}'")

    df.to_csv(INPUT_CSV, index=False)
    print(f"Done. Updated file: {INPUT_CSV}")


if __name__ == "__main__":
    main()
