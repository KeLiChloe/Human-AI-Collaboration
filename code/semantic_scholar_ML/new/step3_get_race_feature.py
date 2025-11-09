import sqlite3
import pandas as pd
import json
import sys
import numpy as np
import time
from tqdm import tqdm


# This script gets race features at both country and paper level

# Country name mapping for standardization
COUNTRY_MAPPING = {
    "England": "UNITED KINGDOM",
    "South Korea": "KOREA, REPUBLIC OF",
    "Vietnam": "VIET NAM",
    "Iran": "IRAN, ISLAMIC REPUBLIC OF",
    "Russia": "RUSSIAN FEDERATION",
}


def get_country_by_name_sql(db_path: str, firstname: str, lastname: str) -> str:
    """Query country by firstname and lastname from SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
    SELECT place
    FROM surnames
    WHERE name_lookup = ?
    ORDER BY incidence DESC
    LIMIT 1;
    """
    
    # Try last name first
    cursor.execute(query, (lastname,))
    result = cursor.fetchone()
    
    if not result:
        # Try first name if last name didn't match
        cursor.execute(query, (firstname,))
        result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return result[0] if result else None


def create_country_race_diversity_columns(json_file_path, df):
    """Add country-level and paper-level race diversity features to DataFrame."""
    with open(json_file_path, 'r') as f:
        country_data = json.load(f)

    # Dictionary to collect all results
    results = {
        "country_race_shannon_entropy_mean": [],
        "country_race_simpson_index_mean": [],
        "country_race_inverse_dominance_mean": [],
        "paper_race_shannon_entropy": [],
        "paper_race_simpson_index": [],
        "paper_race_inverse_dominance": [],
        "asian_composition": [],
        "white_composition": [],
        "black_composition": [],
        "hispanic_composition": [],
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding race diversity features"):
        countries = row['authors_country']
        
        # Initialize lists for country-level metrics
        country_shannon_entropy_list = []
        country_simpson_index_list = []
        country_inverse_dominance_list = []
        
        # Initialize lists for race composition
        asian_composition_list = []
        white_composition_list = []
        black_composition_list = []
        hispanic_composition_list = []

        for country in countries:
            # Standardize country name
            country = COUNTRY_MAPPING.get(country, country)
            country_info = country_data.get(country.upper())
            
            if country_info:
                # Collect country-level diversity metrics
                country_shannon_entropy_list.append(country_info['shannon_entropy'])
                country_simpson_index_list.append(country_info['simpson_index'])
                country_inverse_dominance_list.append(country_info['inverse_dominance'])
                
                # Collect race composition data
                asian_composition_list.append(country_info['asian'])
                white_composition_list.append(country_info['white'])
                black_composition_list.append(country_info['black'])
                hispanic_composition_list.append(country_info['hispanic'])

        # Calculate country-level means (use np.nan if list is empty)
        def safe_mean(lst):
            return np.mean(lst) if lst else np.nan
        
        results["country_race_shannon_entropy_mean"].append(safe_mean(country_shannon_entropy_list))
        results["country_race_simpson_index_mean"].append(safe_mean(country_simpson_index_list))
        results["country_race_inverse_dominance_mean"].append(safe_mean(country_inverse_dominance_list))
        
        # Calculate race composition means
        results["asian_composition"].append(safe_mean(asian_composition_list))
        results["black_composition"].append(safe_mean(black_composition_list))
        results["white_composition"].append(safe_mean(white_composition_list))
        results["hispanic_composition"].append(safe_mean(hispanic_composition_list))
        
        # Calculate paper-level race diversity metrics
        if asian_composition_list:
            proportions = np.array([
                np.mean(asian_composition_list),
                np.mean(black_composition_list),
                np.mean(white_composition_list),
                np.mean(hispanic_composition_list)
            ])
            prop_sum = proportions.sum()
            
            if prop_sum > 0:
                # Normalize proportions to sum to 1
                proportions = proportions / prop_sum
                
                # Shannon entropy: H = -Σ(p_i * log(p_i))
                shannon_entropy = -np.sum([p * np.log(p) if p > 0 else 0 for p in proportions])
                
                # Simpson index: D = Σ(p_i^2)
                simpson_index = np.sum(proportions ** 2)
                
                # Inverse dominance: 1 / max(p_i)
                inverse_dominance = 1.0 / np.max(proportions) if np.max(proportions) > 0 else np.nan
                
                results["paper_race_shannon_entropy"].append(shannon_entropy)
                results["paper_race_simpson_index"].append(simpson_index)
                results["paper_race_inverse_dominance"].append(inverse_dominance)
            else:
                # Handle edge case where all proportions are zero
                results["paper_race_shannon_entropy"].append(np.nan)
                results["paper_race_simpson_index"].append(np.nan)
                results["paper_race_inverse_dominance"].append(np.nan)
        else:
            # No valid data available
            results["paper_race_shannon_entropy"].append(np.nan)
            results["paper_race_simpson_index"].append(np.nan)
            results["paper_race_inverse_dominance"].append(np.nan)

    # Write all results back to DataFrame
    for col, values in results.items():
        df[col] = values

    return df


def query_country_by_name(input_csv, db_path):
    """Query countries for each author based on their names."""
    df = pd.read_csv(input_csv)
    df["authors_country"] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Querying countries by name"):
        # Parse name lists
        firstnames = eval(row["firstname"].lower())
        lastnames = eval(row["lastname"].lower())

        countries = []
        for fn, ln in zip(firstnames, lastnames):
            country = get_country_by_name_sql(db_path, fn.lower(), ln.lower())
            countries.append(country)

        df.at[idx, "authors_country"] = countries

    # Remove rows with None in authors_country list
    df = df[df["authors_country"].apply(lambda x: x is not None and all(c is not None for c in x))]
    return df


if __name__ == "__main__":
    start_time = time.time()
    
    db_path = "database/forebears-surnames.sqlite"
    json_file_path = "meta_data/country_race_diversity_data.json"
    
    if len(sys.argv) < 3:
        print("Usage: python step3_get_race_feature.py <input_file_path> <output_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    # Query countries by author names
    df_tmp = query_country_by_name(input_csv, db_path)
    
    # Add race diversity features
    df = create_country_race_diversity_columns(json_file_path, df_tmp)
    
    # Remove rows with missing values and save
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    
    print(f"Dropped {rows_dropped} rows with NA values ({rows_before} -> {rows_after})")
    
    df.to_csv(output_csv, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")