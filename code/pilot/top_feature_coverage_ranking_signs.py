import pandas as pd
import numpy as np
from os.path import dirname

# Load your dataset
FILE_PATH = "data/pilot/race/Structured_Race_Results.csv"
output_dir = dirname(FILE_PATH)
df = pd.read_csv(FILE_PATH)

if 'race' in FILE_PATH:
    top_5_model_features = ["social_science", "country_race_diversity_score", "female_score_avg", "female_score_max", "female_score_min"]
    top_5_model_ranks = {feature: i for i, feature in enumerate(top_5_model_features)}
    top_1_model_feature = "social_science"

    # Define true sign mappings
    true_positive = {
        "social_science", "female_score_avg", "authors_race_diversity_score",
        "country_race_diversity_score", "black", "natural_science",
        "first_author_female_score", "white", "female_score_max"
    }
    true_negative = {
        "female_score_min", "asian", "num_authors",
        "native_hawaiian", "engineering_and_technology", "hispanic"
    }

elif 'gender' in FILE_PATH:
    top_5_model_features = ["social_science", "female_score_avg", "first_author_female_score", "female_score_max", "female_score_min"]
    top_5_model_ranks = {feature: i for i, feature in enumerate(top_5_model_features)}
    top_1_model_feature = "social_science"

    # Define true sign mappings
    true_positive = {
        "social_science", "female_score_avg", "authors_race_diversity_score",
        "country_race_diversity_score", "black", 
        "first_author_female_score", "white", 
    }
    true_negative = {
        "female_score_min","female_score_max", "asian", "num_authors", "natural_science",
        "native_hawaiian", "engineering_and_technology", "hispanic"
    }

# Initialize new column values
coverage = []
includes_top1 = []
ranks_top1_first = []
rank_agreement_scores = []
sign_accuracy = []


# Process each row (participant)
for _, row in df.iterrows():
    selected_feats = [row[f"feature_{i}"] for i in range(1, 6) if pd.notna(row[f"feature_{i}"])]
    selected_feats_set = set(selected_feats)
    
    # Top 3 feature coverage
    match_count = len(selected_feats_set & set(top_5_model_features))
    coverage.append(round(match_count / 5, 2))
    
    # Includes top 1
    includes_top1.append(int(top_1_model_feature in selected_feats_set))
    
    # Ranks top 1 first
    ranks_top1_first.append(int(row["feature_1"] == top_1_model_feature))
    
    # Rank agreement calculation
    matched_features = [f for f in top_5_model_features if f in selected_feats_set]
    
    if len(matched_features) < 2:
        rank_agreement_scores.append(np.nan)  # Not enough features to evaluate ranking
    else:
        # Get ranks in human list
        human_ranks = {f: selected_feats.index(f) for f in matched_features}
        model_ranks = {f: top_5_model_ranks[f] for f in matched_features}
        
        # Compare relative pairwise orders
        correct_pairs = 0
        total_pairs = 0
        for i in range(len(matched_features)):
            for j in range(i + 1, len(matched_features)):
                f1, f2 = matched_features[i], matched_features[j]
                if (model_ranks[f1] < model_ranks[f2] and human_ranks[f1] < human_ranks[f2]) or \
                   (model_ranks[f1] > model_ranks[f2] and human_ranks[f1] > human_ranks[f2]):
                    correct_pairs += 1
                total_pairs += 1
        
        agreement_score = correct_pairs / total_pairs if total_pairs > 0 else np.nan
        rank_agreement_scores.append(round(agreement_score, 2))

    # Sign accuracy calculation
    correct = 0
    for i in range(1, 6):
        f = row.get(f"feature_{i}")
        s = row.get(f"sign_{i}")
        if (f in true_positive and s == "+") or (f in true_negative and s == "-"):
            correct += 1
    score = correct / 5
    sign_accuracy.append(round(score, 2))

# Add new columns
df["Model_Feature_TOP5_Coverage"] = coverage
df["Includes_Top1_Model_Feature"] = includes_top1
df["Ranks_Top1_Model_Feature_First"] = ranks_top1_first
df["ML_Feature_Rank_Agreement"] = rank_agreement_scores
df["Sign_Accuracy"] = sign_accuracy


composite_scores = []
for cov, rank in zip(df["Model_Feature_TOP5_Coverage"], df["ML_Feature_Rank_Agreement"]):
    if pd.isna(rank):
        composite_scores.append(round(cov, 2)) 
    else:
        score = 0.8 * cov + 0.2 * rank
        composite_scores.append(round(score, 2))
# Add to DataFrame
df["Combined_Coverage_and_Ranking_Score"] = composite_scores




# Save updated file
df.to_csv(f"{output_dir}/Results_with_accuracy_columns.csv", index=False)
print("✅ Updated CSV saved with new columns.")
