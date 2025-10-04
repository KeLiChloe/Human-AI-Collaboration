import pandas as pd
import statsmodels.formula.api as smf

# Load dataset
file_path = "data/pilot/race/Results_with_accuracy_columns.csv"  # Update if needed
df = pd.read_csv(file_path)

# ---- Gender ----
df["Gender"] = df["Gender"].astype("category")


# ---- Academic Position ----
df["Position"] = df["Position"].map({
    "Master Student": 1,
    "PhD Student": 2,
    "Postdoctoral Researcher": 3,
    "Professor (e.g., Assistant Professor, Associate Professor, etc.)": 4,
    "Engineer": 0,
    "AI entrepreneur":0,
}).fillna("Other")

# ---- AI/ML Knowledge Level (ordinal scale) ----
df["AI_Knowledge"] = df["AI_Knowledge"].map({
    "No knowledge": 0,
    "Basic understanding (e.g., general concepts)": 1,
    "Intermediate (e.g., understand algorithms or papers)": 2,
    "Advanced (e.g., conducted ML related research or implemented ML/AI models)": 2
})


model = smf.ols("Model_Feature_TOP5_Coverage ~ C(Gender) + AI_Knowledge", data=df).fit()
model_summary = model.summary()

print(model_summary)