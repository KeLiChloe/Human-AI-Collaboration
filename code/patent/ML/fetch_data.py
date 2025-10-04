import csv
import json
import requests

URL = "https://dataset-search.botosearch.org/SolrQuery"
KEY = "d1961b5b83a27666"
CORE = "patent"

def fetch_docs_for_term(session, term):
    """
    Query Solr for docs containing a single term and return specific fields.
    """
    q_clause = f'(patent_title_lookup:"{term}") OR (patent_abstract:"{term}")'
    
    params = {
        "q": q_clause,
        "fl": "patent_year,patent_title,inventors_firstname,inventors_lastname,patent_abstract",
        "rows": 10000,
        "wt": "json",
        "key": KEY,
        "c": CORE
    }
    
    r = session.get(URL, params=params)
    r.raise_for_status()
    docs = r.json()["response"]["docs"]
    
    return docs


import random

def fetch_docs_general(session, start_year, end_year):
    """
    Fetch up to 1000 random documents per year between start_year and end_year.
    Returns a combined list of docs.
    """
    all_docs = []

    for year in range(start_year, end_year + 1):
        # Generate a random seed for reproducibility per year
        seed = random.randint(1, 1000000)

        params = {
            "q": "*:*",  # match everything
            "fq": f"patent_year:{year}",
            "fl": "patent_year,patent_title,inventors_firstname,inventors_lastname,patent_abstract",
            "rows": 1000,   # fetch 1000 docs max
            "sort": f"random_{seed} asc",  # random order
            "wt": "json",
            "key": KEY,
            "c": CORE
        }

        r = session.get(URL, params=params)
        r.raise_for_status()
        docs = r.json()["response"]["docs"]

        # Keep track of which year they came from
        for d in docs:
            d["sample_year"] = year

        all_docs.extend(docs)
        print(f"Collected {len(docs)} docs for year {year}")

    return all_docs




def save_results_to_csv(results, filename="patent_results.csv"):
    """
    Save list of Solr documents to a CSV file.
    Ensures missing fields are filled with blanks.
    """
    if not results:
        print("No results to save.")
        return

    fieldnames = [
        "patent_year",
        "patent_title",
        "inventors_firstname",
        "inventors_lastname",
        "patent_abstract"
    ]
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            clean_row = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(clean_row)


def main():
    session = requests.Session()

   
    save_file_path = "code/patent/patent_general_results.csv"
    all_results = []
    
    # get women-related patents
    # with open("meta_data/women-related.json", "r", encoding="utf-8") as f:
    #     wordlist = json.load(f)
    # all_results = []
    # for category, terms in wordlist.items():
    #     print(f"Processing category: {category}")
    #     for term in terms:
    #         print(f"  Searching for: {term}")
    #         docs = fetch_docs_for_term(session, term)
    #         all_results.extend(docs)
            
    # get general patents from 2000 to 2025
    general_docs = fetch_docs_general(session, 2000, 2025)
    all_results.extend(general_docs)
    
        
    print(f"Total results collected: {len(all_results)}")
    save_results_to_csv(all_results, f"{save_file_path}")

if __name__ == "__main__":
    main()
