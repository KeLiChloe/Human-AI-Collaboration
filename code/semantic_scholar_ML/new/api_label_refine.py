import pandas as pd
import time
import requests
import os
import argparse
from tqdm import tqdm

# === API Setup ===
DEEPSEEK_API_KEY = "sk-f1423b3921ff40e18be54d158f37be41"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}

# === Instruction for DeepSeek ===
system_instruction = {
    "role": "system",
    "content": (
        "You will be given the title and abstract of a research paper. Your task is to analyze the text and determine "
        "whether this paper discusses issues related to social inequality. Social inequality refers "
        "to disparities or injustices between different groups of people in terms of access to resources, opportunities, rights, "
        "or treatment in society. This can include, but is not limited to, topics such as gender inequality, racial or ethnic disparities, "
        "income or wealth inequality, class-based discrimination, unequal access to education or healthcare, systemic oppression, or "
        "the marginalization of specific communities. It also includes structural or institutional factors that produce or reinforce these inequalities.\n\n"
        "You must output only a single number:\n"
        "1 — if the paper is about social inequality\n"
        "0 — if the paper is not related to social inequality\n\n"
        "Do not provide any explanation, reasoning, or additional text. Your output must be exactly one character: either 0 or 1."
    )
}


def load_dataset(input_file: str, checkpoint_file: str) -> pd.DataFrame:
    """Load dataset from checkpoint if exists, otherwise from input CSV."""
    if os.path.exists(checkpoint_file):
        print(f"Checkpoint found. Loading {checkpoint_file}")
        df = pd.read_csv(checkpoint_file)
    else:
        df = pd.read_csv(input_file)
        print(f"No checkpoint found. Loaded input file {input_file}")
    return df


def process_and_label(df: pd.DataFrame, checkpoint_file: str, output_file: str, save_every_n: int = 100) -> pd.DataFrame:
    """Run labeling via DeepSeek API with checkpoint saving."""

    # Add columns if they don't exist
    if 'AI_label' not in df.columns:
        df['AI_label'] = 0
    if 'label_status' not in df.columns:
        df['label_status'] = 0  # 0 = not processed, 1 = done, -1 = error

    rows_to_check = df[(df['count_inequality_words'] > 0) & (df['count_inequality_words'] < 8) & (df['label_status'] == 0)]
    processed = 0

    for idx, row in tqdm(rows_to_check.iterrows(), total=len(rows_to_check), desc="Labeling papers"):
        title = str(row['title']).strip().replace('\n', ' ')
        abstract = str(row['paper_abstract']).strip().replace('\n', ' ')

        payload = {
            "model": "deepseek-chat",
            "messages": [
                system_instruction,
                {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"}
            ],
            "temperature": 1
        }

        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            label = int(result['choices'][0]['message']['content'].strip())
            df.at[idx, 'AI_label'] = label
            df.at[idx, 'label_status'] = 1
        except Exception as e:
            print(f"❌ Error on row {idx}: {e}")
            df.at[idx, 'AI_label'] = -1
            df.at[idx, 'label_status'] = -1

        processed += 1
        time.sleep(1)

        # Save checkpoint every N rows
        if processed % save_every_n == 0:
            df.to_csv(checkpoint_file, index=False)
            print(f"💾 Saved checkpoint after {processed} papers")

    # Final save
    df.to_csv(checkpoint_file, index=False)
    df.to_csv(output_file, index=False)
    print(f"✅ Final save complete at {output_file}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Label research papers with DeepSeek API.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to final output CSV file")
    parser.add_argument("--checkpoint", default="checkpoint.csv", help="Path to checkpoint file (default: checkpoint.csv)")
    parser.add_argument("--save-every", type=int, default=200, help="Save checkpoint every N rows (default: 100)")
    args = parser.parse_args()

    df = load_dataset(args.input, args.checkpoint)
    process_and_label(df, args.checkpoint, args.output, save_every_n=args.save_every)


if __name__ == "__main__":
    main()
