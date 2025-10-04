import json
import matplotlib.pyplot as plt

# Load category data
with open("code/news-mainstream/category_counts_wordbyword.json") as f:
    cat_data = json.load(f)

# Load yearly totals
with open("code/news-mainstream/year_counts.json") as f:
    total_data = json.load(f)

# Build a lookup for totals
totals = {item["year"]: item["count"] for item in total_data["year_counts"]}

# Plot each category
plt.figure(figsize=(12, 6))

for cat, counts in cat_data.items():
    years = []
    shares = []
    for entry in counts:
        year = entry["year"]
        total = totals.get(year, 0)
        if total > 0:
            years.append(year)
            shares.append(entry["count"] / total)
    plt.plot(years, shares, marker="o", label=cat)

plt.xlabel("Year")
plt.ylabel("Share of total documents")
plt.title("Category trends over time (normalized)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
