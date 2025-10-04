import requests, json
from tqdm import tqdm   # pip install tqdm

URL = "https://dataset-search.botosearch.org/SolrQuery"
KEY = "9dd6576867cd3a16"
CORE = "patent"

START_YEAR = 2000
END_YEAR = 2025
DIR = f"/Users/like/Desktop/Research/Human-AI/code/{CORE}/"

with open("/Users/like/Desktop/Research/Human-AI/meta_data/women-related.json") as f:
    WORDLIST = json.load(f)

def count_for_year(session, year, term):
    """
    Query Solr for docs containing a single term in title_t or description_t.
    """
    # q_clause = f'(title:"{term}") OR (description:"{term}")'
    q_clause = f'(patent_title_lookup:"{term}") OR (patent_abstract:"{term}")'
    
    params = {
        "q": q_clause,
        "fq": f"patent_year:{year}",
        "rows": 0,
        "key": KEY,
        "c": CORE
    }
    r = session.get(URL, params=params)
    r.raise_for_status()
    return r.json()["response"]["numFound"]


session = requests.Session()
results_alt = {}


# Outer loop: over keyword categories
for category, terms in tqdm(WORDLIST.items(), desc="Categories"):
    year_counts = []

    # Loop over years
    for y in tqdm(range(START_YEAR, END_YEAR + 1), desc=f"{category} years", leave=False):
        total = 0

        # Loop over terms
        for t in tqdm(terms, desc=f"{category} {y}", leave=False):
            total += count_for_year(session, y, t)

        year_counts.append({"year": y, "count": total})

    results_alt[category] = year_counts


# 保存结果
with open(DIR+"category-yearly-count.json", "w") as f:
    json.dump(results_alt, f, indent=2)

print(json.dumps(results_alt, indent=2))
