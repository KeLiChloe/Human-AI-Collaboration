import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from os.path import dirname
# ------------------------
# Step 1: Load Data
# ------------------------
FILE_PATH = "code/semantic_scholar_ML/pilot/race/Results_with_accuracy_columns.csv"

df = pd.read_csv(FILE_PATH)

# ------------------------
# Step 2: Create true_accuracy_dict
# ------------------------

# List of the 15 features to track
target_features = [
    "social_science", "female_score_avg", "authors_race_diversity_score",
    "country_race_diversity_score", "black", "natural_science",
    "first_author_female_score", "white", "female_score_max",
    "female_score_min", "asian", "num_authors",
    "native_hawaiian", "engineering_and_technology", "hispanic"
]

# Top 5 feature and sign columns
feature_cols = [f"feature_{i}" for i in range(1, 6)]
sign_cols = [f"sign_{i}" for i in range(1, 6)]

# Function to build the participant's true_accuracy_dict
def build_true_accuracy_dict(row):
    accuracy_dict = {feature: 0 for feature in target_features}
    for feat_col, sign_col in zip(feature_cols, sign_cols):
        feature = row[feat_col]
        sign = row[sign_col]
        if pd.notnull(feature) and feature in accuracy_dict:
            if sign == "+":
                accuracy_dict[feature] = 1
            elif sign == "-":
                accuracy_dict[feature] = -1
    return accuracy_dict

# Apply function row-wise
df["true_accuracy_dict"] = df.apply(build_true_accuracy_dict, axis=1)

# ------------------------
# Step 3: Define Ground Truth
# ------------------------

ground_truth = {
    "social_science": 1,
    "female_score_avg": 1,
    "country_race_diversity_score": 1,
    "female_score_max": 1,
    "female_score_min": -1,
    "authors_race_diversity_score": 0,
    "black": 0,
    "natural_science": 0,
    "first_author_female_score": 0,
    "white": 0,
    "asian": 0,
    "num_authors": 0,
    "native_hawaiian": 0,
    "engineering_and_technology": 0,
    "hispanic": 0
}

# Feature order for consistent vector generation
feature_list = list(ground_truth.keys())

# ------------------------
# Step 4: Calculate correlation and distance
# ------------------------

def calculate_vector_cosine_similarity(row_dict):
    participant_vec = np.array([row_dict[feat] for feat in feature_list])
    truth_vec = np.array([ground_truth[feat] for feat in feature_list])
    
    if norm(participant_vec) == 0 or norm(truth_vec) == 0:
        return np.nan  # Avoid division by zero errors
    
    return dot(participant_vec, truth_vec) / (norm(participant_vec) * norm(truth_vec))

def calculate_vector_distance(row_dict):
    participant_vec = np.array([row_dict[feat] for feat in feature_list])
    truth_vec = np.array([ground_truth[feat] for feat in feature_list])
    # return euclidean(participant_vec, truth_vec)
    return np.sum((participant_vec - truth_vec) ** 2)

# Apply correlation and distance calculations
df["vector_correlation"] = df["true_accuracy_dict"].apply(calculate_vector_cosine_similarity)
df["vector_distance"] = df["true_accuracy_dict"].apply(calculate_vector_distance)

max_sos = df["vector_distance"].max()



# Normalize into (0, 1) score: higher = better
df["normalized_true_accuracy_score"] = 1 - (df["vector_distance"] / max_sos)

mean_corr = df["vector_correlation"].mean()
mean_score = df["normalized_true_accuracy_score"].mean()

# ------------------------
# Step 6: Random Benchmark Simulation + Visualization
# ------------------------

import matplotlib.pyplot as plt
import seaborn as sns

# Simulate random guessers
random_correlations = []
random_distances = []

for _ in range(1000):
    random_dict = {feat: 0 for feat in feature_list}
    chosen_feats = np.random.choice(feature_list, 5, replace=False)
    for feat in chosen_feats:
        random_dict[feat] = np.random.choice([-1, 1])
    
    corr = calculate_vector_cosine_similarity(random_dict)
    dist = calculate_vector_distance(random_dict)
    
    random_correlations.append(corr)
    random_distances.append(dist)

# Calculate average metrics for random guessers
avg_random_corr = np.nanmean(random_correlations)
avg_random_dist = np.mean(random_distances)
avg_random_norm_score = 1 - (avg_random_dist / max_sos)

# ------------------------
# Step 7: Visualization (Sorted, No Lines)
# ------------------------

sns.set(style="whitegrid", font_scale=1.2)

# Sort participants by each metric
sorted_corr = df["vector_correlation"].sort_values(ascending=True).reset_index(drop=True)
sorted_norm_score = df["normalized_true_accuracy_score"].sort_values(ascending=True).reset_index(drop=True)

# ------------------------
# Step 8: Aggregate Human Performance
# ------------------------

# Build the average human prediction vector
all_participant_vectors = np.array([
    [row[feat] for feat in feature_list] for row in df["true_accuracy_dict"]
])

average_human_vector = np.mean(all_participant_vectors, axis=0)

# Build ground truth vector
ground_truth_vector = np.array([ground_truth[feat] for feat in feature_list])

# Calculate aggregate metrics
if np.std(average_human_vector) == 0 or np.std(ground_truth_vector) == 0:
    agg_corr = np.nan
else:
    # cosine similarity
    agg_corr = dot(average_human_vector, ground_truth_vector) / (norm(average_human_vector) * norm(ground_truth_vector))

# Sum of Squares Distance
agg_dist = np.sum((average_human_vector - ground_truth_vector) ** 2)

# Normalized Accuracy
agg_norm_score = 1 - (agg_dist / max_sos)

# ------------------------
# Step 9: Updated Visualization (with Human Aggregate)
# ------------------------

sns.set_style("white")
plt.figure(figsize=(12, 7))

# Custom color palette (visually distinct and colorblind-friendly)
colors = {
    "ML Benchmark": "#00C9FB",
    "Human Aggregate": "#d95f02",
    "Human Average": "#3627ff",
    "GenAI": "#e7298a",
    "Random Benchmark": "#00ab22",
    "Participants": "#4D4D4D"
}

# Scatter plot for participant scores
sns.scatterplot(
    x=sorted_corr.index,
    y=sorted_corr,
    s=120,
    color=colors["Participants"],
    label="Participants"
)

plt.axhline(1, color=colors["ML Benchmark"], linestyle='--', linewidth=2, label="ML Benchmark")
plt.axhline(agg_corr, color=colors["Human Aggregate"], linestyle='-', linewidth=2.5, label="Human Aggregation")

plt.axhline(0.40, color=colors["GenAI"], linestyle='-.', linewidth=2, 
            label="GenAI (GPT-o3, Gemini 2.5 Pro, Claude 3, DeepSeek R1)")
plt.axhline(mean_corr, color=colors["Human Average"], linestyle=':', linewidth=2.5, label="Human Average")
plt.axhline(avg_random_corr, color=colors["Random Benchmark"], linestyle=(0, (3, 3, 1, 3)), linewidth=2, label="Random Benchmark (1000 trials)")

# Titles and labels
plt.title("Correlation Score (Cosine Similarity, Higher = Better)", fontsize=18)
plt.xlabel("Sorted Participants", fontsize=16)
plt.ylabel("Correlation Score", fontsize=16)

# Legend formatting
plt.legend(fontsize=12, loc='best', frameon=True)

# Remove grid for a clean look
sns.despine()
plt.tight_layout()
# plt.show()


# # --- Plot normalized_true_accuracy_score ---
sns.set_style("white")
plt.figure(figsize=(12, 7))

# Custom color palette (visually distinct and colorblind-friendly)
colors = {
    "ML Benchmark": "#ff33b4c4",
    "Human Aggregate": "#47D600",
    "Human Average": "#004bf8",
    "GenAI": "#ff3636",
    "Random Benchmark": "#979797",
    "Participants": "#666666"
}

# Scatter plot for participant scores
sns.scatterplot(
    x=sorted_norm_score.index,
    y=sorted_norm_score,
    s=100,
    color=colors["Participants"],
    label="Participants"
)

# Reference lines with unique styles
plt.axhline(1, color=colors["ML Benchmark"], linestyle='-', linewidth=2, label="ML Benchmark")
plt.axhline(agg_norm_score, color=colors["Human Aggregate"], linestyle='-', linewidth=2, label="Human Aggregation")
plt.axhline(0.675, color=colors["GenAI"], linestyle='--', linewidth=2, label="GenAI Average (GPT-o3, GPT-5-thinking, Gemini 2.5 Pro, Claude 3, DeepSeek R1)")
# plt.axhline(0.428, color=colors["GenAI (DeepSeek)"], linestyle=':', linewidth=2, label="GenAI (DeepSeek)")

plt.axhline(mean_score, color=colors["Human Average"], linestyle=':', linewidth=2.5, label="Human Average")
plt.axhline(avg_random_norm_score, color=colors["Random Benchmark"], linestyle=(0, (3, 3, 1, 3)), linewidth=2, label="Random Benchmark (1000 trials)")

# Titles and labels
plt.title("Accuracy Score (Combining Magnitude and Sign, Higher = Better)", fontsize=20, weight='bold', pad=10)
plt.xlabel("Sorted Participants", fontsize=16)
plt.ylabel("Accuracy Score", fontsize=16)

# Legend formatting
plt.legend(fontsize=12, loc='best', frameon=True)

# Remove grid for a clean look
sns.despine()
plt.tight_layout()
plt.show()
