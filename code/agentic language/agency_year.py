import pandas as pd
import matplotlib.pyplot as plt

# List of file paths
file_paths = [
    "data/samples/race-pos/agency_scored_ml_sample.csv",
    "data/samples/gender-pos/agency_scored_ml_sample.csv",
    "data/samples/econ-pos/agency_scored_ml_sample.csv"
]

# Read and concatenate all files
df_list = [pd.read_csv(path) for path in file_paths]
df_all = pd.concat(df_list, ignore_index=True)

# Filter for papers mentioning inequality
df_ai = df_all[df_all['AI_label'] == 1]

# Group by year and calculate mean values
agency_trends = df_ai.groupby('year')[['syntactic_agency', 'semantic_agency']].mean().reset_index()

# --- Plot 1: Syntactic Agency ---
plt.figure(figsize=(10, 5))
plt.plot(agency_trends['year'], agency_trends['syntactic_agency'], marker='o', color='blue', label='Syntactic Agency')
plt.title("Syntactic Agency Over Time (Papers Mentioning Inequality)")
plt.xlabel("Year")
plt.ylabel("Mean Syntactic Agency")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Semantic Agency ---
plt.figure(figsize=(10, 5))
plt.plot(agency_trends['year'], agency_trends['semantic_agency'], marker='o', color='green', label='Semantic Agency')
plt.title("Semantic Agency Over Time (Papers Mentioning Inequality)")
plt.xlabel("Year")
plt.ylabel("Mean Semantic Agency")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
