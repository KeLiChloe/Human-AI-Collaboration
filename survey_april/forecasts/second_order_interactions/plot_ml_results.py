import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

mpl.rcParams.update({
    "font.family": "Times New Roman",
})

X_VOTE_MAX = 1200

FEATURE_LABELS = {
    "social_science": "Social Science",
    "natural_science": "Natural Science",
    "engineering_and_technology": "Engineering & Tech",
    "num_authors": "Num. Authors",
    "female": "Female",
    "asian": "Asian",
    "black": "Black",
    "hispanic_and_other": "Hispanic & Other",
    "white": "White",
    "authors_race_diversity_score": "Author Race Diversity",
    "country_race_diversity_score": "Country Race Diversity",
    "news_inequality_mentions_3_years": "\"Inequality\" Mentions in News (3yr)",
    "paper_inequality_mentions_3_years": "\"Inequality\" Mentions in Papers (3yr)",
}


def format_feature_label(feature_1, feature_2):
    left = FEATURE_LABELS.get(feature_1, feature_1)
    right = FEATURE_LABELS.get(feature_2, feature_2)
    return f"{left}\n*\n{right}"


def plot_task(task_name, entries, output_dir):
    df = pd.DataFrame(entries)
    if df.empty:
        print(f"Skip empty task: {task_name}")
        return

    if "votes" in df.columns and "feature_importance" in df.columns:
        df = df.sort_values(by=["votes", "feature_importance"], ascending=[True, True])
    elif "rank" in df.columns:
        df = df.sort_values(by="rank", ascending=False)

    df["feature_label"] = df.apply(
        lambda row: format_feature_label(row["feature_1"], row["feature_2"]),
        axis=1,
    )

    norm = mcolors.Normalize(
        vmin=df["feature_importance"].min(),
        vmax=df["feature_importance"].max(),
    )
    cmap = sns.color_palette("crest", as_cmap=True)
    colors = cmap(norm(df["feature_importance"].values))

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.barh(
        df["feature_label"],
        df["votes"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    label_offset = X_VOTE_MAX * 0.015
    for bar, importance in zip(bars, df["feature_importance"]):
        ax.text(
            bar.get_width() + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{importance:.3f}",
            va="center",
            ha="left",
            fontsize=20,
        )

    task_title = f"{task_name.title()} Task"
    ax.set_xlabel(
        "Votes (Top 5) Across 1000 Subsample-Runs\n(100 reshuffles × 10 subsamples)",
        fontsize=30,
        labelpad=18,
    )
    ax.set_title(
        f"Top Voted Features - {task_title}\nMDI (Gini) Importance",
        fontsize=30,
        pad=20,
        weight="bold",
    )
    ax.tick_params(axis="y", labelsize=24)
    ax.tick_params(axis="x", labelsize=25)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.set_xlim(0, X_VOTE_MAX)
    ax.xaxis.set_major_locator(MultipleLocator(150))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Averaged MDI (Gini) Importance", fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    png_path = output_dir / f"{task_name}_interaction_importance_votes.png"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)

    print(f"Saved: {png_path}")


def main():
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "ML_results.json"
    output_dir = base_dir / "figures"
    output_dir.mkdir(exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    for task_name, entries in results.items():
        plot_task(task_name, entries, output_dir)


if __name__ == "__main__":
    main()
