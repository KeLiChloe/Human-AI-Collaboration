import pandas as pd
import sys
import time

def load_ml_dataset(ml_path: str) -> pd.DataFrame:
    """Load main ML dataset with a 'year' column."""
    return pd.read_csv(ml_path)

def prepare_lagged_features(
    yearly_df: pd.DataFrame,
    year_col: str,
    value_col: str,
    prefix: str,
    lags: int = 4,
    rolling_years: int = 3,

) -> pd.DataFrame:
    """
    Generate lag features, rolling average, and buckets for t-1, t-2, t-3, and rolling average.
    """
    df = yearly_df[[year_col, value_col]].copy()

    # === Lag features ===
    for lag in range(lags):
        df[f"{prefix}_t-{lag}"] = df[value_col].shift(lag)

    # === Rolling average (for t, t-1, ..., t-(rolling_years-1)) ===
    df[f"{prefix}_{rolling_years}yr_avg"] = df[value_col].rolling(window=rolling_years, min_periods=1).mean()

    # Drop raw value column
    df.drop(columns=[value_col], inplace=True)

    return df

def merge_yearly_features(ml_df: pd.DataFrame, *yearly_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple yearly feature DataFrames into the ML dataset.
    :param ml_df: Main dataset with 'year' column
    :param yearly_features: Any number of DataFrames to merge on 'year'
    :return: Merged dataset
    """
    merged_df = ml_df.copy()
    for feat_df in yearly_features:
        merged_df = merged_df.merge(feat_df, on="year", how="left")
    return merged_df

if __name__ == "__main__":
    start_time = time.time()
    acad_ineq_path = "figures/semantic_scholar_relative_frequency.csv"
    news_ineq_path = "figures/news_inequality_frequency.csv"
    
    if len(sys.argv) != 3:
        print("Usage: python add_year_feature.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # === Load datasets ===
    input_df = load_ml_dataset(input_path)
    acad_df = pd.read_csv(acad_ineq_path)
    news_df = pd.read_csv(news_ineq_path)
    
    # === Prepare features ===
    acad_feats = prepare_lagged_features(acad_df, "year", "inequality_total_relfreq", "acad_ineq", lags=4, rolling_years=3)
    news_feats = prepare_lagged_features(news_df, "year", "inequality_total_relfreq", "news_ineq", lags=4, rolling_years=3)
    news_gender_feats = prepare_lagged_features(news_df, "year", "gender_relfreq", "news_gender_ineq", lags=4, rolling_years=3)
    news_econ_feats = prepare_lagged_features(news_df, "year", "economic_relfreq", "news_econ_ineq", lags=4, rolling_years=3) 
    news_race_feats = prepare_lagged_features(news_df, "year", "race_relfreq", "news_race_ineq", lags=4, rolling_years=3) 
    
    # === Merge and save ===
    final_df = merge_yearly_features(input_df, acad_feats, news_feats, news_gender_feats, news_econ_feats, news_race_feats)
    final_df.to_csv(f"{output_path}", index=False)
    
    print(f"✅ Merged dataset saved as '{output_path}.csv'")


    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")