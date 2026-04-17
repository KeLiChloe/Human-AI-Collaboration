import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from os.path import dirname


# ---------------------
# Config
# ---------------------
FILE_PATH = "code/semantic_scholar_ML/pilot/race/Structured_Race_Results.csv"
OUTPUT_DIR = dirname(FILE_PATH)

if 'race' in FILE_PATH:
    ML_TOP_FEATURES = ["social_science", "country_race_diversity_score", "female_score_avg", "engineering_and_technology", "asian"]
    TRUE_SIGNS = {
        "social_science": "+",
        "country_race_diversity_score": "+",
        "female_score_avg": "+",
        "engineering_and_technology": "-",
        "asian": "-",
    }
elif 'gender' in FILE_PATH:
    pass
else:
    pass


TOTAL_PARTICIPANTS = 69
TOP_N = 8

# ---------------------
# Load Data
# ---------------------
df = pd.read_csv(FILE_PATH)

# ---------------------
# Utility Functions
# ---------------------

def get_feature_ranking(df, rank_type="top5", top_n=8):
    """
    Extract feature rankings as frequency and percentage.
    rank_type: "top5_mostly_selected" or "top1_mostly_selected"
    """
    if rank_type == "top5_mostly_selected":
        feature_columns = [f"feature_{i}" for i in range(1, 6)]
        all_features = [feat for col in feature_columns for feat in df[col].dropna()]
    elif rank_type == "top1_mostly_selected":
        all_features = df["feature_1"].dropna().tolist()
    else:
        raise ValueError("rank_type must be 'top5_mostly_selected' or 'top1_mostly_selected'")

    counts = Counter(all_features)
    df_counts = pd.DataFrame(counts.items(), columns=["Feature", "Frequency"])
    percent_col = "Top5_Percentage" if rank_type == "top5_mostly_selected" else "Top1_Percentage"
    df_counts[percent_col] = (df_counts["Frequency"] / TOTAL_PARTICIPANTS) * 100
    return df_counts.sort_values(by=percent_col, ascending=False).head(top_n)

def plot_feature_ranking_overestimation(ranking_df, percent_col, palette_color, title):
    """
    Plot ranked features with specified color and title.
    """
    n = len(ranking_df)
    palette = sns.light_palette(palette_color, n_colors=n, reverse=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=percent_col,
        y="Feature",
        data=ranking_df,
        palette=palette
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    
    plt.xlabel("Participants Selecting Features (%)", labelpad=10)
    plt.ylabel("")
    plt.yticks(rotation=30)
    
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    y_labels = ["ratio_of_asian_authors" if label == "asian" else label for label in y_labels]
    y_labels = ["ratio_of_white_authors" if label == "white" else label for label in y_labels]
    y_labels = ["ratio_of_black_authors" if label == "black" else label for label in y_labels]
    y_labels = ["ratio_of_hispanic_authors" if label == "hispanic" else label for label in y_labels]
    y_labels = ["ratio_of_native_hawaiian_authors" if label == "native_hawaiian" else label for label in y_labels]
    ax.set_yticklabels(y_labels)
    
    plt.title(title, fontsize=18, weight='bold')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/most_frequently_selected_top{5 if percent_col == 'Top5_Percentage' else 1}.png", dpi=300)
    print(f"✅ Figure saved as {OUTPUT_DIR}/most_frequently_selected_top{5 if percent_col == 'Top5_Percentage' else 1}.png")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_ml_feature_underestimation(df, ml_top_features, total_participants, output_dir):
    """
    Publication-quality barplot: ML feature underestimation.
    Includes gradient color, sorted descending (top-down), and visible bar edges.
    """

    # ---------------------
    # Data Prep
    # ---------------------
    selected = df[[f"feature_{i}" for i in range(1, 6)]].values.flatten()
    selected = pd.Series(selected).dropna()
    counts = selected.value_counts()

    ml_feature_selection = {
        feat: (counts.get(feat, 0) / total_participants) * 100
        for feat in ml_top_features
    }

    ml_df = (
        pd.DataFrame(list(ml_feature_selection.items()), columns=["Feature", "Selected_Percentage"])
        .sort_values(by="Selected_Percentage", ascending=True)
    )

    # ---------------------
    # Aesthetic Setup
    # ---------------------
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))

    # Gradient palette (from light to dark purple)
    colors = sns.color_palette("Purples", n_colors=len(ml_df))

    # Sort high → low, display high on top
    ax = sns.barplot(
        data=ml_df,
        y="Feature",
        x="Selected_Percentage",
        palette=list(colors),  # lighter top, darker bottom
    )

    # ---------------------
    # Style & Typography
    # ---------------------
    sns.despine(left=True, bottom=True)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.linewidth": 1,
        "axes.edgecolor": "black",
    })

    ax.set_xlabel("Participants Selecting Feature (%)", fontsize=13, labelpad=10)
    ax.set_ylabel("")
    ax.set_xlim(0, 100)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=16)
    # rename one of y ticks labels
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    y_labels = ["ratio_of_asian_authors" if label == "asian" else label for label in y_labels]
    ax.set_yticklabels(y_labels)
    
    plt.yticks(rotation=30)

    # Flip y-axis so the highest is on top
    ax.invert_yaxis()


    # ---------------------
    # Title & Save
    # ---------------------
    plt.title(
        "Humans Perception of ML-Identified Important Features \n(Underestimation)",
        fontsize=18,
        weight="bold",
        pad=15
    )
    plt.tight_layout()
    save_path = f"{output_dir}/ml_top_features_participant_selection_PRO_v2.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✅ Figure saved as {save_path}")

def plot_sign_accuracy_table(df, true_signs, output_dir):
    """
    Plot a table showing how many participants selected each feature and assigned the correct sign.
    """
    results = []

    for feature, correct_sign in true_signs.items():
        selected_by = 0
        correct_sign_assigned = 0

        for i in range(1, 6):
            f_col = f"feature_{i}"
            s_col = f"sign_{i}"
            matched_rows = df[df[f_col] == feature]

            selected_by += matched_rows.shape[0]
            correct_sign_assigned += (matched_rows[s_col] == correct_sign).sum()

        accuracy = (correct_sign_assigned / selected_by * 100) if selected_by > 0 else 0

        # Wrap long feature names for better display
        wrapped_feature = "\n".join(feature[i:i+20] for i in range(0, len(feature), 20))

        results.append([
            wrapped_feature,
            selected_by,
            correct_sign_assigned,
            f"{round(accuracy, 2)}%"
        ])

    # Plot table using matplotlib
    col_labels = ["Feature", "Selected by", "Correctly Assigned", "Accuracy (%)"]
    fig, ax = plt.subplots(figsize=(10, 2 + len(results) * 0.5))
    ax.axis('off')

    table = ax.table(
        cellText=results,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]  # Keeps the table within figure bounds
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)  # Adjust as needed
    table.scale(1.2, 2.2)

    plt.title("Sign Assignment Accuracy for ML-Identified Features", fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sign_assignment_accuracy_table.png", dpi=300)
    plt.close()
    print("✅ Table figure saved as sign_assignment_accuracy_table.png")



# ---------------------
# Style Settings
# ---------------------
sns.set_theme(style="whitegrid", font_scale=1.5)

# ---------------------
# Generate & Plot Top 5 Features
# ---------------------
top5_ranking = get_feature_ranking(df, rank_type="top5_mostly_selected", top_n=TOP_N)
plot_feature_ranking_overestimation(
    top5_ranking,
    percent_col="Top5_Percentage",
    palette_color="orange",
    title=f"Features Most Frequently Selected as Top 5 by Humans \nfor {'Racial' if 'race' in FILE_PATH else 'Gender'} Inequality"
)

# ---------------------
# Generate & Plot Top 1 Features
# ---------------------
top1_ranking = get_feature_ranking(df, rank_type="top1_mostly_selected", top_n=TOP_N)
plot_feature_ranking_overestimation(
    top1_ranking,
    percent_col="Top1_Percentage",
    palette_color="orange",
    title=f"Features Most Frequently Selected as Top 1 by Humans \nfor{'Racial' if 'race' in FILE_PATH else 'Gender'} Inequality"
)


# ---------------------
# Plot ML-Identified Underestimated Features
# ---------------------
plot_ml_feature_underestimation(df, ML_TOP_FEATURES, TOTAL_PARTICIPANTS, OUTPUT_DIR)

# ---------------------
# Plot Sign Assignment Accuracy
# ---------------------
plot_sign_accuracy_table(df, TRUE_SIGNS, OUTPUT_DIR)

