import requests
from requests.auth import HTTPBasicAuth
import json
import time
from tqdm import tqdm
import sys
import os
import random
import re
import pandas as pd

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def contains_weird_characters(text):
    if isinstance(text, str):
        return bool(re.search(r'[^\x00-\xFF]', text))
    return False

FIELDS = ["title", "paper_abstract", "year", "authors", "fields_of_study"]

def get_query_result(session, params):
    params['wt'] = 'json'
    try:
        response = session.get(solr_url, params=params,
                               auth=HTTPBasicAuth(username, password),
                               timeout=30)
        response.raise_for_status()
        return response.json().get("response", {}).get("docs", [])
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []

def flush_to_csv(response, output_csv, header_written):
    """将一批数据写入 CSV"""
    if not response:  # 空结果就跳过
        return header_written
    df_batch = pd.DataFrame(response, columns=FIELDS)
    df_batch.to_csv(output_csv, mode='a', index=False, encoding='utf-8', header=not header_written)
    return True  # 表示 header 已写过

if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 3:
        print("Usage: python create_data_sample.py <output_csv> <query_mode>")
        sys.exit(1)
    
    output_csv = sys.argv[1]
    query_mode = sys.argv[2]  # "ineq" or "non-ineq"
    
    wordlist = read_json_file("meta_data/wordlist.json")

    solr_url = 'https://solr-semantic-scholar-raw.totosearch.org/solr/semantic_scholar_raw/select'
    username = 'research'
    password = 'insead123456'

    categories = ["economic"]
    sentiments = [[1,-1]]

    # 如果已有旧文件，先清空
    open(output_csv, 'w', encoding='utf-8').close()

    header_written = False

    total_iterations = sum(len(wordlist.get(category, {}).get(str(sentiment), [])) 
                     for category, sentiments_for_category in zip(categories, sentiments) 
                     for sentiment in sentiments_for_category)

    with requests.Session() as session:  # 复用连接
        if query_mode == "ineq":
            with tqdm(total=total_iterations, desc="Processing Keywords") as pbar:
                for i, category in enumerate(categories):
                    category_sentiments = sentiments[i]
                    for sentiment in category_sentiments:
                        keywords = wordlist.get(category, {}).get(str(sentiment), [])
                        for keyword in keywords:
                            query = f'(title_lookup:"{keyword}" OR paper_abstract_lookup:"{keyword}") AND year:[2005 TO *]'
                            random_seed = random.randint(0, 10000)
                            params = {
                                'q': query,
                                'rows': 10000,
                                'fl': ','.join(FIELDS),
                                'sort': f'random_{random_seed} asc'
                            }
                            response = get_query_result(session, params)
                            header_written = flush_to_csv(response, output_csv, header_written)
                            pbar.update(1)

        elif query_mode == "non-ineq":
            # 按年份循环 query，避免一次性卡死
            years = range(2005, 2021)  # 2005 到 2020
            with tqdm(total=len(years), desc="Processing Years") as pbar:
                for year in years:
                    query = f"year:{year}"
                    random_seed = random.randint(0, 10000)
                    params = {
                        'q': query,
                        'rows': 20000,  
                        'fl': ','.join(FIELDS),
                        'sort': f'random_{random_seed} asc'
                    }
                    response = get_query_result(session, params)
                    header_written = flush_to_csv(response, output_csv, header_written)
                    pbar.update(1)

        else: 
            raise ValueError("Invalid query_mode. Use 'ineq' or 'non-ineq'.")

    # ---- 数据清洗 ----
    df = pd.read_csv(output_csv)
    print(f"Initial rows: {len(df)}")
    df_cleaned = df.copy()

    df_cleaned = df_cleaned.dropna(subset=["year", "authors", "fields_of_study"])
    print(f"Rows after dropping NA: {len(df_cleaned)}")

    df_cleaned = df_cleaned[~df_cleaned.map(contains_weird_characters).any(axis=1)] 
    print(f"Rows after dropping weird characters: {len(df_cleaned)}")

    df_cleaned = df_cleaned.drop_duplicates(subset='title', keep='first')
    print(f"Rows after dropping duplicates: {len(df_cleaned)}")
    
    print(f"Dropped {(len(df) - len(df_cleaned))/len(df):.2%} in total")
    print(f"Remaining rows: {len(df_cleaned)}")

    # 保存清洗后的结果
    df_cleaned.to_csv(output_csv, index=False, encoding='utf-8')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
