import pandas as pd

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
    bucket: bool = True,
    bucket_method: str = "quantile",  # or "uniform"
    n_bins: int = 4
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

    # === Bucket features ===
    if bucket:
        for lag in [1, 2, 3]:
            col = f"{prefix}_t-{lag}"
            if col in df.columns:
                if bucket_method == "quantile":
                    df[f"{col}_bucket"] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
                else:
                    df[f"{col}_bucket"] = pd.cut(df[col], bins=n_bins, labels=False)

        # Bucket for rolling average
        avg_col = f"{prefix}_{rolling_years}yr_avg"
        if bucket_method == "quantile":
            df[f"{prefix}_avg{rolling_years}_bucket"] = pd.qcut(df[avg_col], q=n_bins, labels=False, duplicates='drop')
        else:
            df[f"{prefix}_avg{rolling_years}_bucket"] = pd.cut(df[avg_col], bins=n_bins, labels=False)

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
    # rename "patent_year" to "year"
    merged_df.rename(columns={"patent_year": "year"}, inplace=True)
    for feat_df in yearly_features:
        merged_df = merged_df.merge(feat_df, on="year", how="left")
    return merged_df

def main():
    # === File paths ===
    ml_path = "code/patent/merged.csv"
    output_path = "code/patent/merged_year.csv"
    
    # acad_ineq_path = "figures/semantic_scholar_relative_frequency.csv"
    news_ineq_path = "figures/news_inequality_frequency.csv"
    patent_ineq_path = "figures/patent_gender_relative_frequency.csv"

    
    # === Load datasets ===
    ml_df = load_ml_dataset(ml_path)
    
    news_df = pd.read_csv(news_ineq_path)
    patent_df = pd.read_csv(patent_ineq_path)
    
    # === Prepare features ===
    patent_feats = prepare_lagged_features(patent_df, "year", "gender_relfreq", "patent_ineq", lags=4, rolling_years=3, bucket=False)
    news_feats = prepare_lagged_features(news_df, "year", "inequality_total_relfreq", "news_ineq", lags=4, rolling_years=3, bucket=False)
    
    # === Merge and save ===
    final_df = merge_yearly_features(ml_df, patent_feats, news_feats)
    final_df.to_csv(f"{output_path}", index=False)
    
    print(f"✅ Merged dataset saved as '{output_path}.csv'")

if __name__ == "__main__":
    main()
