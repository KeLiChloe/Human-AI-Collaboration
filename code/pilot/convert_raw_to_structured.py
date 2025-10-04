# Re-import necessary packages after code execution state reset
import pandas as pd

# Reload the original file
file_path = "data/pilot/gender/Gender_Results.csv"
df = pd.read_csv(file_path)

# Start building the structured DataFrame
structured_data = pd.DataFrame()

# Copy over metadata
structured_data["Name"] = df["Name"]
structured_data["Gender"] = df["Gender"]
structured_data["Position"] = df["Academic Positions"]
structured_data["Institution"] = df["Institutions"]
structured_data["AI_Knowledge"] = df["Self-rated master of AI knowledge"]

# Determine top 5 features based on importance ranking
feature_rankings = []

for idx, row in df.iterrows():
    # Split top_5 features
    features = row["Top_5"].split(",")
    ranked_features = []

    for feat in features:
        rank_col = f"Importance ranking - {feat}"
        sign_col = f"sign - {feat}"

        # Get rank and sign
        rank = row[rank_col] 
        sign = row[sign_col] 

        ranked_features.append((feat, rank, sign))

    # Sort features by rank
    ranked_features.sort(key=lambda x: x[1])

    # Store sorted features and signs
    for i in range(5):
        structured_data.at[idx, f"feature_{i+1}"] = ranked_features[i][0]
        structured_data.at[idx, f"sign_{i+1}"] = ranked_features[i][2]

# Parse and assign top 3 second-order interactions and their signs
for i in range(1, 4):
    soi_col = f"SOI_top{i}"
    sign_col = f"sign - SOI_top{i}"
    feature_pairs = df[soi_col].str.split(",", expand=False)
    structured_data[f"SOI{i}_feature"] = feature_pairs
    structured_data[f"SOI{i}_sign"] = df[sign_col]
    
# Save to CSV
output_path = "data/pilot/gender/Structured_Gender_Results.csv"
structured_data.to_csv(output_path, index=False)