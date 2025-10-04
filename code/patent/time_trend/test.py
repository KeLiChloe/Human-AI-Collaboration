import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Load your CSV files ===
patent_df = pd.read_csv("figures/patent_gender_relative_frequency.csv")
news_df = pd.read_csv("figures/news_inequality_frequency.csv")

# === Merge patent with relevant news inequality variables ===
merged_df = pd.merge(
    patent_df,
    news_df[['year', 'gender_relfreq', 'economic_relfreq', 'race_relfreq']],
    on='year',
    suffixes=('_patent', '_news')
)

# === Rename for clarity ===
merged_df.rename(columns={'gender_relfreq': 'gender_relfreq_news'}, inplace=True)
merged_df.rename(columns={'economic_relfreq': 'economic_relfreq_news'}, inplace=True)
merged_df.rename(columns={'race_relfreq': 'race_relfreq_news'}, inplace=True)

# === Create lagged features (3 years) ===
for lag in range(1, 5):
    merged_df[f'patent_lag{lag}'] = merged_df['gender_relfreq_patent'].shift(lag)
    merged_df[f'news_gender_lag{lag}'] = merged_df['gender_relfreq_news'].shift(lag)
    merged_df[f'news_economic_lag{lag}'] = merged_df['economic_relfreq_news'].shift(lag)
    merged_df[f'news_race_lag{lag}'] = merged_df['race_relfreq_news'].shift(lag)

# === Drop rows with NaNs due to lagging ===
ml_df = merged_df.copy()
# create a column which is all 1s in ml_df
ml_df['all_ones'] = 0

# === Define features and target ===
feature_cols = (
    [f'patent_lag{l}' for l in range(1, 4)]
    # [f'news_gender_lag{l}' for l in range(1, 4)]
    # [f'news_economic_lag{l}' for l in range(1, 4)]
    # [f'news_race_lag{l}' for l in range(1, 4)]
    # ['all_ones']
)


target_col = 'gender_relfreq_patent'

X = ml_df[feature_cols]
y = ml_df[target_col]
years = ml_df['year'].values

# === Rolling window forecasting ===
initial_train_size = 13  # start with 10 years of training data

preds = []
actuals = []
years_pred = []

for i in range(initial_train_size, len(X)):
    X_train = X.iloc[:i]
    y_train = y.iloc[:i]
    X_test = X.iloc[i:i+1]
    y_test = y.iloc[i:i+1]

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)[0]
    
    preds.append(y_pred)
    actuals.append(y_test.values[0])
    years_pred.append(years[i])

# === Evaluate ===
mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(actuals, preds)


print(f"✅ MAE: {mae:.3f}")
print(f"✅ RMSE: {rmse:.3f}")
print(f"✅ MAPE: {mape:.3f}")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === 计算误差指标 ===
mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))

# === 设置图形样式 ===
sns.set(style="whitegrid", font_scale=1.2)

# === 画图 ===
plt.figure(figsize=(12, 6))
plt.plot(years_pred, actuals, label="Actual", color="#f28b4f", linewidth=2, marker='o', markersize=6)
plt.plot(years_pred, preds, label=f"Predicted (MAE={mae:.3f}, RMSE={rmse:.3f})", 
         color="#5c91f5", linewidth=2, linestyle='--', marker='x', markersize=6)

# # 可选：误差阴影区域
# plt.fill_between(years_pred,
#                  np.array(actuals),
#                  np.array(preds),
#                  color="gray",
#                  alpha=0.2,
#                  label="Prediction Error")

# === 标签与图例 ===
plt.title("Forecast of Patent Gender Frequency\n", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Relative Frequency (%)")
plt.legend(loc="best", frameon=True)
plt.tight_layout()

# === 显示图形 ===
plt.savefig("figures/patent_gender_forecast.png", dpi=300)
