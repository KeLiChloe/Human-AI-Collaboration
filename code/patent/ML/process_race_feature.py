import pandas as pd
import numpy as np
import ast

# Load the CSV files
df1 = pd.read_csv("code/patent/merged_year.csv")
# df2 = pd.read_csv("code/patent/patent_women_race_year.csv")

# Function to convert list-like string to actual list and compute mean
def mean_of_list_column(column):
    return column.apply(lambda x: np.mean(ast.literal_eval(x)) if pd.notnull(x) else np.nan)

def max_of_list_column(column):
    return column.apply(lambda x: np.max(ast.literal_eval(x)) if pd.notnull(x) else np.nan)

def min_of_list_column(column):
    return column.apply(lambda x: np.min(ast.literal_eval(x)) if pd.notnull(x) else np.nan)
# Replace list-columns with their mean values
# for df in [df1, df2]:
#     df["country_race_shannon_entropy"] = mean_of_list_column(df["country_race_shannon_entropy"])
#     df["country_race_simpson_index"] = mean_of_list_column(df["country_race_simpson_index"])
#     df["country_race_inverse_dominance"] = mean_of_list_column(df["country_race_inverse_dominance"])


df1["country_race_shannon_entropy"] = mean_of_list_column(df1["country_race_shannon_entropy"])
df1["country_race_simpson_index"] = mean_of_list_column(df1["country_race_simpson_index"])
df1["country_race_inverse_dominance"] = mean_of_list_column(df1["country_race_inverse_dominance"])
df1["female_score_max"] = max_of_list_column(df1["female_percents"])
df1["female_score_min"] = min_of_list_column(df1["female_percents"])



# Function to compute race percentage features
def compute_race_percentages(column):
    races = ['asian', 'white', 'black', 'hispanic']
    results = {f"percentage_of_{race}": [] for race in races}
    
    for val in column:
        if pd.isnull(val):
            counts = {race: 0 for race in races}
            total = 0
        else:
            parsed = ast.literal_eval(val)
            total = len(parsed)
            counts = {race: parsed.count(race) for race in races}
        
        for race in races:
            percentage = counts[race] / total if total > 0 else 0
            results[f"percentage_of_{race}"].append(percentage)
    
    return pd.DataFrame(results)

# Add race percentage features
df1 = df1.join(compute_race_percentages(df1["country_highest_ratio_race"]))
# df2 = df2.join(compute_race_percentages(df2["country_highest_ratio_race"]))


# Concatenate the two dataframes
# df_combined = pd.concat([df1, df2], ignore_index=True)


# delete columns that are not needed
columns_to_drop = [
    "inventors_firstname", "inventors_lastname", "inventors_country",
    "country_highest_ratio_race"]
df1.drop(columns=columns_to_drop, inplace=True)

# Save to CSV
df1.to_csv("code/patent/short_sample_patent_combined.csv", index=False)
