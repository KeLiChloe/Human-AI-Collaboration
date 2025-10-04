import json
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Load data
# ------------------------------
DIR = "code/semantic_scholar/"
start_year = 2000
end_year = 2025
Title_name = DIR.split("/")[-2].replace("_", " ").title()

print(f"Processing data for: {Title_name}")

with open(DIR+"category_yearly_counts.json", "r") as f:
    category_data = json.load(f)

with open(DIR+"all-docs-yearly-count.json", "r") as f:
    all_docs_data = json.load(f)

# Convert all-docs yearly counts into DataFrame
df_all = pd.DataFrame(all_docs_data)

# Convert category data into DataFrames per category, then merge
dfs = []
for category, records in category_data.items():
    df = pd.DataFrame(records)
    df.rename(columns={"count": f"{category}_count"}, inplace=True)
    dfs.append(df)

# Merge all categories into one DataFrame on 'year'
df_cat = dfs[0]
for d in dfs[1:]:
    df_cat = pd.merge(df_cat, d, on="year", how="outer")

# Merge with total documents
df = pd.merge(df_all, df_cat, on="year", how="outer")

# ------------------------------
# Compute relative frequencies
# ------------------------------
for category in category_data.keys():
    df[f"{category}_relfreq"] = (df[f"{category}_count"] / df["count"])*100

# Add a "total inequality" relative frequency = sum of all categories / total
df["inequality_total_count"] = df[[f"{c}_count" for c in category_data.keys()]].sum(axis=1)
df["inequality_total_relfreq"] = (df["inequality_total_count"] / df["count"])*100

# store category_relfreq and total_relfreq into a csv file from year_start to year_end
df_relfreq = df[["year"] + [f"{c}_relfreq" for c in category_data.keys()] + ["inequality_total_relfreq"]]
df_relfreq = df_relfreq[(df_relfreq["year"] >= start_year) & (df_relfreq["year"] <= end_year)]
df_relfreq.to_csv(f"figures/{Title_name.lower().replace(' ', '_')}_relative_frequency.csv", index=False)

# ------------------------------
# Plotting function
# ------------------------------
def plot_relative_frequencies(df, categories, start_year=None, end_year=None):
    """
    Plot relative frequencies of inequality mentions by category over time,
    including the total inequality mentions.
    
    Parameters:
        df (pd.DataFrame): Dataset with relative frequencies.
        categories (list): Categories to plot.
        start_year (int, optional): Start year for the time frame.
        end_year (int, optional): End year for the time frame.
    """
    # Filter by time frame
    data = df.copy()
    if start_year is not None:
        data = data[data["year"] >= start_year]
    if end_year is not None:
        data = data[data["year"] <= end_year]
    
    # Plot relative frequency trends
    plt.figure(figsize=(12, 6))
    # for category in categories:
    #     if category == "gender":
    #         plt.plot(data["year"], data[f"{category}_relfreq"], label=category)
    
    # Plot total inequality mentions
    plt.plot(data["year"], data["inequality_total_relfreq"], 
             label="total inequality", linestyle="--", linewidth=2, color="black")
    
    plt.title(f"{Title_name} - Relative Frequency of Inequality Mentions (%)", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Percentage of docs mentioning inequality", fontsize =16)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/{Title_name.lower().replace(' ', '_')}_relative_frequencies.png", dpi=300)

# ------------------------------
# Example usage
# ------------------------------
plot_relative_frequencies(df, list(category_data.keys()), start_year=start_year, end_year=end_year)


