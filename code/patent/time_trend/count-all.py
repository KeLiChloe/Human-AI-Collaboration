import requests, json
from tqdm import tqdm   # pip install tqdm

URL = "https://dataset-search.botosearch.org/SolrQuery"
KEY = "9dd6576867cd3a16"
CORE = "patent"

START_YEAR = 2000
END_YEAR = 2025

DIR = f"/Users/like/Desktop/Research/Human-AI/code/{CORE}/"
def count_for_year(session, year):
    """
    Query Solr for docs containing a single term in title_t or description_t.
    """
    
    params = {
        "q": "*:*",
        "fq": f"patent_year:{year}",
        "rows": 0,
        "key": KEY,
        "c": CORE
    }
    r = session.get(URL, params=params)
    r.raise_for_status()
    return r.json()["response"]["numFound"]

session = requests.Session()

year_counts = []
for y in tqdm(range(START_YEAR, END_YEAR + 1), desc="years", leave=False):
    total = count_for_year(session, y)
    year_counts.append({"year": y, "count": total})

# save the list as json
with open(DIR+"all-docs-yearly-count.json", "w") as f:
    json.dump(year_counts, f, indent=2)
