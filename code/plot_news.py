import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
BASE_DIR = "code/news"
SOURCES = ["news-alternative", "news-mainstream", "youtube-news"]
save_csv_file = "news_relative_frequency.csv"
start_year = 2000
end_year = 2025

# ------------------------------
# Helper to load one dataset
# ------------------------------
def load_dataset(dir_path):
    with open(os.path.join(dir_path, "category_yearly_counts.json"), "r") as f:
        category_data = json.load(f)

    with open(os.path.join(dir_path, "all-docs-yearly-count.json"), "r") as f:
        all_docs_data = json.load(f)

    df_all = pd.DataFrame(all_docs_data)

    dfs = []
    for category, records in category_data.items():
        df = pd.DataFrame(records)
        df.rename(columns={"count": f"{category}_count"}, inplace=True)
        dfs.append(df)

    df_cat = dfs[0]
    for d in dfs[1:]:
        df_cat = pd.merge(df_cat, d, on="year", how="outer")

    df = pd.merge(df_all, df_cat, on="year", how="outer")
    return df

# ------------------------------
# Aggregate all sources
# ------------------------------
dfs = []
for source in SOURCES:
    dir_path = os.path.join(BASE_DIR, source)
    dfs.append(load_dataset(dir_path))

# Merge by summing counts per year
df_news = dfs[0].set_index("year")
for d in dfs[1:]:
    df_news = df_news.add(d.set_index("year"), fill_value=0)

df_news = df_news.reset_index()

# ------------------------------
# Compute relative frequencies
# ------------------------------
categories = [c.replace("_count", "") for c in df_news.columns if c.endswith("_count")]

for category in categories:
    df_news[f"{category}_relfreq"] = (df_news[f"{category}_count"] / df_news["count"]) * 100

df_news["inequality_total_count"] = df_news[[f"{c}_count" for c in categories]].sum(axis=1)
df_news["inequality_total_relfreq"] = (df_news["inequality_total_count"] / df_news["count"]) * 100


# store category_relfreq and total_relfreq into a csv file from year_start to year_end
df_relfreq = df_news[["year"] + [f"{c}_relfreq" for c in categories] + ["inequality_total_relfreq"]]
df_relfreq = df_relfreq[(df_relfreq["year"] >= start_year) & (df_relfreq["year"] <= end_year)]
df_relfreq.to_csv(f"figures/{save_csv_file}", index=False)


# ------------------------------
# Plot final integrated figure
# ------------------------------
def plot_news(df, categories, start_year=None, end_year=None):
    data = df.copy()
    if start_year:
        data = data[data["year"] >= start_year]
    if end_year:
        data = data[data["year"] <= end_year]

    plt.figure(figsize=(12, 6))
    # for category in categories:
    #     if category == "gender":
    #         plt.plot(data["year"], data[f"{category}_relfreq"], label=category)

    plt.plot(data["year"], data["inequality_total_relfreq"],
             label="total inequality", linestyle="--", linewidth=2, color="black")

    plt.title("News \n Relative Frequency of Inequality Mentions (%)", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Percentage of docs mentioning inequality", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/news_integrated_relative_frequencies.png", dpi=300)

# ------------------------------
# Example usage
# ------------------------------
plot_news(df_news, categories, start_year=start_year, end_year=end_year)
