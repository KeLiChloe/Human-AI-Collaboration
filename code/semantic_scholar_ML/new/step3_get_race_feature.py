import sqlite3
import pandas as pd
import json
import pandas as pd
import sys
import numpy as np
import time
from tqdm import tqdm
from collections import Counter



# This scirpt get race features: both country and paper level

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def get_country_by_name_sql(db_path: str, firstname: str, lastname: str) -> str:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
        
    # Query using the last name first
    query_lastname = """
    SELECT place
    FROM surnames
    WHERE name_lookup = ?
    ORDER BY incidence DESC
    LIMIT 1;
    """
    cursor.execute(query_lastname, (lastname,))
    result = cursor.fetchone()

    # If a result is found with the last name, return it
    if result:
        cursor.close()
        conn.close()
        return result[0]

    # If no result, query using the first name
    query_firstname = """
    SELECT place
    FROM surnames
    WHERE name_lookup = ?
    ORDER BY incidence DESC
    LIMIT 1;
    """
    cursor.execute(query_firstname, (firstname,))
    result = cursor.fetchone()

    # Close the database connection
    cursor.close()
    conn.close()

    # If a result is found with the first name, return it
    if result:
        return result[0]
    
    # If no match for both names, return None
    return None


def create_country_race_diversity_columns(json_file_path, df):
    with open(json_file_path, 'r') as f:
        country_data = json.load(f)

    races = ['asian', 'white', 'black', 'hispanic']

    # 用字典收集所有结果
    results = {
        "country_highest_ratio_race": [],
        "country_race_shannon_entropy_mean": [],
        "country_race_simpson_index_mean": [],
        "country_race_inverse_dominance_mean": [],
        "paper_race_shannon_entropy": [],
        "paper_race_simpson_index": [],
        "paper_race_inverse_dominance": []
    }
    for race in races:
        results[f'percentage_of_{race}'] = []

    # 遍历行
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding race diversity features"):
        countries = row['authors_country']
        highest_ratio_race_list = []
        shannon_entropy_list = []
        simpson_index_list = []
        inverse_dominance_list = []

        for country in countries:
            # 统一国家名
            mapping = {
                "England": "UNITED KINGDOM",
                "South Korea": "KOREA, REPUBLIC OF",
                "Vietnam": "VIET NAM",
                "Iran": "IRAN, ISLAMIC REPUBLIC OF",
                "Russia": "RUSSIAN FEDERATION",
            }
            country = mapping.get(country, country)
            country_info = country_data.get(country.upper())
            if country_info:
                highest_ratio_race_list.append(country_info['highest_ratio_race'])
                shannon_entropy_list.append(country_info['shannon_entropy'])
                simpson_index_list.append(country_info['simpson_index'])
                inverse_dominance_list.append(country_info['inverse_dominance'])
            else:
                break

        if highest_ratio_race_list:
            # 国家层面的均值
            results["country_highest_ratio_race"].append(highest_ratio_race_list)
            results["country_race_shannon_entropy_mean"].append(np.mean(shannon_entropy_list))
            results["country_race_simpson_index_mean"].append(np.mean(simpson_index_list))
            results["country_race_inverse_dominance_mean"].append(np.mean(inverse_dominance_list))

            # race 百分比
            total = len(highest_ratio_race_list)
            counts = Counter(highest_ratio_race_list)
            probs = []
            for race in races:
                p = counts[race] / total if total > 0 else 0
                results[f'percentage_of_{race}'].append(p)
                probs.append(p)

            # Shannon entropy
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            results["paper_race_shannon_entropy"].append(entropy)

            # Simpson index
            simpson = 1 - sum(p**2 for p in probs)
            results["paper_race_simpson_index"].append(simpson)

            # Inverse dominance
            inv_dom = 1 / max(probs) if max(probs) > 0 else 0
            results["paper_race_inverse_dominance"].append(inv_dom)

        else:
            results["country_highest_ratio_race"].append(None)
            results["country_race_shannon_entropy_mean"].append(None)
            results["country_race_simpson_index_mean"].append(None)
            results["country_race_inverse_dominance_mean"].append(None)
            results["paper_race_shannon_entropy"].append(None)
            results["paper_race_simpson_index"].append(None)
            results["paper_race_inverse_dominance"].append(None)
            for race in races:
                results[f'percentage_of_{race}'].append(0)

    # 一次性写回 DataFrame
    for col, values in results.items():
        df[col] = values

    return df



def query_country_by_name(input_csv, db_path):

    df = pd.read_csv(input_csv)
    df["authors_country"] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Querying countries by name"):

        # 解析名字列表
        firstnames = eval(row["firstname"].lower()) 
        lastnames  = eval(row["lastname"].lower())  

        countries = []
        for fn, ln in zip(firstnames, lastnames):
            country = get_country_by_name_sql(db_path, fn.lower(), ln.lower())
            countries.append(country)

        # 存结果
        df.at[idx, "authors_country"] = countries

    # remove rows with None in authors_country list
    df = df[df["authors_country"].apply(lambda x: x is not None and all(c is not None for c in x))]
    return df


def compute_shannon_entropy(row, races):
    probs = [row[f'percentage_of_{race}'] for race in races if row[f'percentage_of_{race}'] > 0]
    return -sum(p * np.log(p) for p in probs)

# 使用示例
if __name__ == "__main__":
    
    start_time = time.time()
    
    db_path = "database/forebears-surnames.sqlite"  # SQLite 数据库路径
    json_file_path = "meta_data/country_race_diversity_data.json"
    
    if len(sys.argv) < 3:
        print("Usage: python step3_get_race_feature.py <input_file_path> <output_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    df_tmp = query_country_by_name(input_csv, db_path)
    df = create_country_race_diversity_columns(json_file_path, df_tmp)
    
    
    df = df.dropna()
    df.to_csv(f"{output_csv}", index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")