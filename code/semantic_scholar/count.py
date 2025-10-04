import requests, json
from datetime import datetime
from tqdm import tqdm   # pip install tqdm

URL = "https://dataset-search.botosearch.org/SolrQuery"
KEY = "9dd6576867cd3a16"
CORE = "semantic_scholar"

START_YEAR = 1980
END_YEAR = 2025
DIR = f"/Users/like/Desktop/Research/Human-AI/code/{CORE}/"

with open("/Users/like/Desktop/Research/Human-AI/meta_data/wordlist.json") as f:
    WORDLIST = json.load(f)

def count_for_year(session, year, term):
    q_clause = f'(title:"{term}") OR (paperAbstract:"{term}")'
    params = {
        "q": q_clause,
        "fq": f"year:{year}",
        "rows": 0,
        "key": KEY,
        "c": CORE
    }
    r = session.get(URL, params=params)
    r.raise_for_status()
    return r.json()["response"]["numFound"]

session = requests.Session()
results_alt = {}

# 外层循环：类别
for category, pol_map in tqdm(WORDLIST.items(), desc="Categories"):
    
    terms = []
    
    # skip if category == "susceptibility"
    if category == "susceptibility":
        continue
    
    # skip if polarity == 0
    for k, tlist in pol_map.items():
        if k == "0":
            continue
        terms.extend(tlist)
    
    year_counts = []
    # 年份循环
    for y in tqdm(range(START_YEAR, END_YEAR + 1), desc=f"{category} years", leave=False):
        total = 0
        # 词循环
        for t in tqdm(terms, desc=f"{category} {y}", leave=False):
            total += count_for_year(session, y, t)
        year_counts.append({"year": y, "count": total})

    results_alt[category] = year_counts

# 保存结果
with open(DIR+"category_yearly_counts.json", "w") as f:
    json.dump(results_alt, f, indent=2)

print(json.dumps(results_alt, indent=2))
