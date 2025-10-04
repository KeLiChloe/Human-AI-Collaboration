
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
import csv


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def contains_weird_characters(text):
    # Check if the text contains non-ASCII characters
    if isinstance(text, str):
        return bool(re.search(r'[^\x00-\xFF]', text))  # 覆盖了 ASCII（0-127）和 Latin-1（128-255）字符集
    return False

FIELDS = ["title", "paper_abstract", "year", "authors", "journal_name", "fields_of_study"]

def write_to_csv(docs, output_csv):
    write_header = not os.path.exists(output_csv) or os.stat(output_csv).st_size == 0
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for doc in docs:
            row = {field: doc.get(field, "") for field in FIELDS}
            writer.writerow(row)

        
def get_query_result(params):
    params['wt'] = 'json'  # force JSON
    try:
        response = requests.get(
            solr_url, params=params,
            auth=HTTPBasicAuth(username, password),
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("response", {}).get("docs", [])
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []




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

    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        file.write('')  # This ensures the file is empty before appending data

   # Calculate total keywords for progress bar initialization
    total_iterations = sum(len(wordlist.get(category, {}).get(str(sentiment), [])) 
                     for category, sentiments_for_category in zip(categories, sentiments) 
                     for sentiment in sentiments_for_category)

    if query_mode == "ineq":
        with tqdm(total=total_iterations, desc="Processing Keywords") as pbar:
            for i, category in enumerate(categories):
                category_sentiments = sentiments[i]
                for sentiment in category_sentiments:
                    keywords = wordlist.get(category, {}).get(str(sentiment), [])
                    for keyword in keywords:
                        query = f'(title_lookup:"{keyword}" OR paper_abstract_lookup:"{keyword}") AND year:[2008 TO *]'
                        random_seed = random.randint(0, 10000)
                        params = {
                            'q': query,
                            'wt': 'json',
                            'rows': 2,
                            'fl': 'title,paper_abstract,year,authors,journal_name,fields_of_study',  # Fields to include in the CSV
                            'sort': f'random_{random_seed} asc'
                        }
                        response = get_query_result(params)
                        write_to_csv(response, output_csv)
                        pbar.update(1)  # Update the progress bar after each keyword

    elif query_mode == "non-ineq":
        query = "year:[2008 TO *]"
        random_seed = random.randint(0, 10000)
        params = {
            'q': query,
            'wt': 'json',
            'rows': 200,
            'fl': 'title,paper_abstract,year,authors,journal_name,fields_of_study',  # Fields to include in the CSV
            'sort': f'random_{random_seed} asc'
        }
        response = get_query_result(params)
        write_to_csv(response, output_csv)

    
    # read the output_csv and do data cleaning  
    df = pd.read_csv(output_csv)
    print(f"Initial rows: {len(df)}")
    
    df_cleaned = df.dropna(how='any')
    print(f"Rows after dropping NA: {len(df_cleaned)}")
    
    df_cleaned = df_cleaned[df_cleaned.apply(lambda row: len(row) == 6, axis=1)]
    print(f"Rows after ensuring 6 fields: {len(df_cleaned)}")
    
    df_cleaned = df_cleaned[~df_cleaned.map(contains_weird_characters).any(axis=1)]
    print(f"Rows after dropping weird characters: {len(df_cleaned)}")
    
    df_cleaned = df_cleaned.drop_duplicates(subset='title', keep='first')
    print(f"Rows after dropping duplicates: {len(df_cleaned)}")
    
    
    print(f"Dropped {(len(df) - len(df_cleaned))/len(df)} in total")
    print(f"Remaining rows: {len(df_cleaned)}")
    
    # save cleaned data
    df_cleaned.to_csv(output_csv, index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")