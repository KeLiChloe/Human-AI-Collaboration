import pandas as pd
import statsmodels.api as sm

# Load the CSV file
df = pd.read_csv("data/pilot/race/Results_with_true_accuracy.csv")

X1 = df['Model_Feature_TOP5_Coverage']
X2 = df['vector_correlation']

# calculate correlation
correlation = X1.corr(X2)
print(f"Correlation between {X1.name} and {X2.name}: {correlation:.3f}")
