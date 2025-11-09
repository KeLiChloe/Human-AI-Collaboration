import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import statsmodels.api as sm


from scipy import stats
from matplotlib.patches import Patch

# Function to scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将 ndarray 转回 DataFrame，并保留列名
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled

def initial_screen_features_RF(X, y, threshold=0.01):
    """
    Select important features based on Random Forest importance scores.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        threshold (float): Minimum importance score to retain a feature.

    Returns:
        DataFrame: Filtered feature matrix.
        list: Selected feature names.
    """
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X, y)
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Filter features based on importance threshold
    selected_features = importance_df[importance_df['Importance'] > threshold]['Feature']
    X_filtered = X[selected_features]

    print("\nFeature Importance:")
    print(importance_df)
    print(f"\nSelected {len(selected_features)} features based on importance > {threshold}")

    return X_filtered

def initial_screen_features_lasso(X, y, alpha=0.001):
    """
    Select important features based on LASSO coefficients.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        alpha (float): Regularization strength (LASSO parameter).

    Returns:
        DataFrame: Filtered feature matrix.
        list: Selected feature names.
    """

    # Train LASSO model
    lasso_model = Lasso(alpha=alpha, random_state=42)
    lasso_model.fit(X, y)

    # Get coefficients and feature importance
    coefficients = lasso_model.coef_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    # Filter features based on non-zero coefficients
    selected_features = importance_df[importance_df['Coefficient'] != 0]['Feature']
    X_filtered = X[selected_features]

    print("\nLASSO Feature Coefficients:")
    print(importance_df[importance_df['Coefficient'] != 0])
    print(f"\nSelected {len(selected_features)} features with non-zero coefficients")

    return X_filtered



def tune_hyperparameters(X_train, y_train, subset_id, model_save_dir):
    """
    Tune Logistic Regression hyperparameters using GridSearchCV.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series or array): Training labels.
        subset_id (int): Subset index for tracking and saving results.
        model_save_dir (str): Directory to save best parameter files.

    Returns:
        dict: Best hyperparameters for Logistic Regression.
    """
    print(f"🔍 Tuning hyperparameters for subset {subset_id}...")

    # Define parameter grid
    param_grid = {
        "solver": ["liblinear", "lbfgs"],   # support both L1 & L2
        "max_iter": [500, 1000],
        "C": [0.01, 0.1, 0.3, 1, 3, 10],
        "class_weight": ["balanced"],
    }

    # Logistic Regression model
    base_model = LogisticRegression(random_state=42)

    # Stratified 5-fold CV (for balanced evaluation)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use ROC-AUC as the metric — more robust than accuracy for classification
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Extract best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"✅ Best params for subset {subset_id}: {best_params}")
    print(f"   Mean CV ROC-AUC: {best_score:.4f}")

    # Save best parameters to file
    best_params_path = os.path.join(model_save_dir, f"LR_best_params_subset_{subset_id}.pkl")
    with open(best_params_path, "wb") as f:
        pickle.dump(best_params, f)

    return best_params



def add_second_order_interactions(X, sep=" * "):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X)
    
    # Modify interaction feature names to use custom separator
    original_names = X.columns
    raw_feature_names = poly.get_feature_names_out(original_names)
    
    # Replace space with desired separator
    custom_feature_names = [name.replace(" ", sep) for name in raw_feature_names]
    
    return pd.DataFrame(X_interactions, columns=custom_feature_names)


# load and prepare data with second-order interactions, and do initail feature screening
def load_and_prepare_data(file_path, drop_columns=None, add_SOI=False):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"country_race_shannon_entropy_mean": "country_race_diversity_score", 
                            "paper_race_shannon_entropy":"authors_race_diversity_score",
                            "female_score_mean": "female_score_avg",
                            "white":"ratio_of_white_authors",
                            "black":"ratio_of_black_authors",
                            "asian":"ratio_of_asian_authors",})
    
    # Create the target variable
    # df['target'] = df['count_frequency_inequality_words'].apply(lambda x: 1 if x > 0 else 0)
    df['target'] = df["AI_label"].apply(lambda x: 1 if x == 1 else 0)
    
    y = df['target']
    
    # Select features 
    X = df.drop(columns=drop_columns)
    print(f"Initial number of features: {X.shape[1]}")
    
    if add_SOI:
        # Add second-order interaction features
        X = add_second_order_interactions(X)
        print(f"Number of features after adding second-order interactions: {X.shape[1]}")
        
        # Initial feature screening using Random Forest importance
        X = initial_screen_features_lasso(X, y, 0.05)
        print(f"Number of features after initial screening: {X.shape[1]}")
        
    # shuffle the dataset
    random_seed = np.random.randint(100000)
    dataset = pd.concat([X, y], axis=1).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    X = dataset.iloc[:, :-1]  # Features
    y = dataset.iloc[:, -1]   # Target

    return X, y, df



def plot_coef_summary(all_coefs, all_pvals, X, model_save_dir, split_N):
    # --- 1️⃣ 计算均值、95% CI ---
    coef_matrix = np.vstack(all_coefs)
    pval_matrix = np.vstack(all_pvals)
    n_runs = coef_matrix.shape[0]

    mean_coef = coef_matrix.mean(axis=0)
    std_coef = coef_matrix.std(axis=0, ddof=1)
    se = std_coef / np.sqrt(n_runs)

    t_val = stats.t.ppf(1 - 0.025, df=n_runs - 1)
    ci_lower = mean_coef - t_val * se
    ci_upper = mean_coef + t_val * se

    # ✅ 计算每个特征的平均 p 值（基于每次 subset 的模型）
    mean_pvals = pval_matrix.mean(axis=0)
    

    stars = []
    for p in mean_pvals:
        if p < 0.001:
            stars.append("***")
        elif p < 0.01:
            stars.append("**")
        elif p < 0.05:
            stars.append("*")
        else:
            stars.append("")

    # --- 3️⃣ DataFrame ---
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Mean": mean_coef,
        "Lower95CI": ci_lower,
        "Upper95CI": ci_upper,
        "Stars": stars
    }).sort_values(by="Mean", ascending=False)
    
    
    # Select top N features based on absolute mean coefficient
    top_n = 10
    top_coef = coef_df.reindex(coef_df["Mean"].abs().sort_values(ascending=False).head(top_n).index)
    top_coef = top_coef.iloc[::-1]

    # --- 4️⃣ 绘图 ---
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = top_coef["Mean"].apply(lambda x: "#c64646fa" if x > 0 else "#0d53b5eb")
    if split_N > 1:
        xerr = [
            top_coef["Mean"] - top_coef["Lower95CI"],
            top_coef["Upper95CI"] - top_coef["Mean"]
        ]
        
        ci_width = top_coef["Upper95CI"] - top_coef["Lower95CI"]
        offsets = 0.08 + 0.3 * ci_width # value text 偏移，基于 ci 宽度动态调整
        
        max_limit = np.ceil(np.max(np.abs(ci_upper)) * 10) / 10 + 0.2
    
    else:
        xerr = None  # 不绘制置信区间
        offsets = np.full_like(top_coef["Mean"], 0.2) # value text 固定偏移
        
        max_limit = np.ceil(np.max(np.abs(top_coef["Mean"])) * 10) / 10 
        
    
    bars = ax.barh(
        top_coef["Feature"],
        top_coef["Mean"],
        xerr=xerr,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=4 if split_N > 1 else 0,  # 去除 capsize
        alpha=0.9
        )

            

    # --- 5️⃣ 添加数值+星号 ---
    

    for bar, coef, offset, star in zip(bars, top_coef["Mean"], offsets, top_coef["Stars"]):
        label = f"{coef:.2f} ({star})" if star else f"{coef:.2f}"
        xpos = coef + np.sign(coef) * offset
        ax.text(
            xpos,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left" if coef > 0 else "right",
            fontsize=20,
            color="black"
    )

    # --- 6️⃣ 轴线 & 样式 ---
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean Coefficient (95% CI)", fontsize=18, labelpad=30)
    if split_N > 1:
        title = f"Logistic Regression Coefficients (across {split_N} subsamples)"
    else:
        title = f"Logistic Regression Coefficients (across all samples)"
    ax.set_title(title, fontsize=25, fontweight="bold", pad=30)

    # ✅ 去除网格
    ax.grid(False)
    
    # # ✅ 去除坐标轴边框（脊柱）
    # for spine in ["top", "right"]:
    #     ax.spines[spine].set_visible(False)


    # ✅ legend
    significance_legend = [
        Patch(facecolor='none', edgecolor='none', label='*    p < 0.05'),
        Patch(facecolor='none', edgecolor='none', label='**   p < 0.01'),
        Patch(facecolor='none', edgecolor='none', label='***  p < 0.001')
    ]

    ax.legend(
        handles=significance_legend,
        loc='lower right',          
        frameon=True,               # ✅ 有 legend 框
        fontsize=14,
        title='Significance',
        title_fontsize=16,
        borderpad=0.8,
        handlelength=0,             # ✅ 去掉前面的空 handle 符号（可选）
        handletextpad=0.4,
    )

    ax.set_xlim(-max_limit-0.4, max_limit+0.6)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=20)


    plt.savefig(f"{model_save_dir}/logit_feature_importance.png", dpi=600, bbox_inches="tight")
    # plt.show()

    print(f"✅ Saved improved coefficient summary to {model_save_dir}/logit_feature_importance.png")
    return coef_df


def main(file_path, load_existing_best_params, model_save_dir, drop_columns=None, add_SOI=False, scale=False):
    """
    Find robust feature importance rankings using shuffled and split subsets.
    """
    # Step 1: Load and prepare data with optional second-order interactions
    X, y, _ = load_and_prepare_data(file_path, drop_columns, add_SOI)

    Split_N = 10
    subsets_X = np.array_split(X.sample(frac=1, random_state=42), Split_N)
    subsets_y = np.array_split(y.sample(frac=1, random_state=42), Split_N)

    all_coefs = []
    all_pvals = [] 
    random_seed = np.random.randint(100000)

    for i in range(Split_N):
        print(f"\nProcessing subset {i + 1}...")
        X_split, y_split = subsets_X[i], subsets_y[i]

        # Step 2: Get or tune hyperparameters
        if load_existing_best_params:
            with open(f"{model_save_dir}/LR_best_params_subset_{i}.pkl", "rb") as file:
                best_params = pickle.load(file)
        else:
            best_params = tune_hyperparameters(X_split, y_split, i, model_save_dir)

        # Split for training/testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_split, y_split, test_size=0.2, random_state=random_seed + i
        )
        
        if scale:
            X_train, X_test = scale_data(X_train, X_test)
        

        model = LogisticRegression(**best_params, random_state=random_seed + i)
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"Split {i+1}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, AUC: {auc_score:.2f}")

        
        coefs = model.coef_[0]
        pvals = np.ones_like(coefs)  # 默认值，如果没法算就保1
        
        try:
            # statsmodels 用于得到每个 coef 的 p-value
            X_train_sm = sm.add_constant(X_train)
            sm_model = sm.Logit(y_train, X_train_sm).fit(disp=0)
            pvals = sm_model.pvalues[1:].values  # exclude intercept
        except Exception as e:
            print(f"⚠️ Failed to compute p-values for subset {i}: {e}")
        
        all_coefs.append(coefs)
        all_pvals.append(pvals)

    plot_coef_summary(all_coefs, all_pvals, X, model_save_dir, Split_N)






    
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python feature_importance_random_split_validation.py <file_path> <model_save_dir>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_save_dir = sys.argv[2]
    
    
    
        # Select features, dropping non-relevant columns
    drop_columns = ['count_frequency_inequality_words', 'AI_label', 'label_status', 'target', 'title', 'paper_abstract', # lables
                    'mixed', 'other', 'native_americans', 'native_hawaiian_or_other_pacific_islander',
                    'acad_ineq_t-0', 'acad_ineq_t-1', 'acad_ineq_t-2', 
                    'news_ineq_t-0', 'news_ineq_t-1', 'news_ineq_t-2', 
                    # 'acad_ineq_t-3', 'news_ineq_t-3',
                    'acad_ineq_3yr_avg', 'news_ineq_3yr_avg',
                    'female_score_min', 'female_score_max', 'first_author_female_score', 
                    # 'female_score_mean'
                    'year',
                    ]
    add_SOI = False # whether to add second-order interaction features
    
    scale = True
    
    load_existing_best_params = True

    main(file_path, load_existing_best_params, model_save_dir, drop_columns, add_SOI, scale)
    
    
    
    