import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# === Step 1: Load your data ===
gender_freq = pd.read_csv("figures/patent_gender_relative_frequency.csv")  # Patent data
news_freq = pd.read_csv("figures/news_inequality_frequency.csv")           # News data

# === Step 2: Merge datasets on year ===
df = pd.merge(
    gender_freq[['year', 'gender_relfreq']], 
    news_freq[['year', 'gender_relfreq']], 
    on='year', 
    suffixes=('_patent', '_news')
)

# === Step 3: Define Granger causality test function ===
def granger_test(df, max_lag=5):
    summary = {}
    
    print("=== Patent → News ===")
    results1 = grangercausalitytests(df[['gender_relfreq_patent', 'gender_relfreq_news']], 
                                     maxlag=max_lag, verbose=False)
    pvals1 = {}
    for lag, res in results1.items():
        p_val = res[0]['ssr_chi2test'][1]
        pvals1[lag] = p_val
        print(f"Lag {lag}: p = {p_val:.4f}")
    best_lag1 = min(pvals1, key=pvals1.get)
    print(f"--> Best lag = {best_lag1}, p = {pvals1[best_lag1]:.4f}\n")
    summary["Patent→News"] = (best_lag1, pvals1[best_lag1])

    print("=== News → Patent ===")
    results2 = grangercausalitytests(df[['gender_relfreq_news', 'gender_relfreq_patent']], 
                                     maxlag=max_lag, verbose=False)
    pvals2 = {}
    for lag, res in results2.items():
        p_val = res[0]['ssr_chi2test'][1]
        pvals2[lag] = p_val
        print(f"Lag {lag}: p = {p_val:.4f}")
    best_lag2 = min(pvals2, key=pvals2.get)
    print(f"--> Best lag = {best_lag2}, p = {pvals2[best_lag2]:.4f}\n")
    summary["News→Patent"] = (best_lag2, pvals2[best_lag2])
    
    return summary

# === Step 4: Run the test ===
results_summary = granger_test(df, max_lag=5)

# === Step 5: Print final summary ===
print("=== Summary of Granger Causality Tests ===")
for direction, (lag, pval) in results_summary.items():
    print(f"{direction}: Best lag = {lag}, p = {pval:.4f}")
