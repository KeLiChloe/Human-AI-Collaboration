import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',                # 使用衬线字体
    'font.serif': ['Times New Roman'],     # Times 字体
})

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
    Tune hyperparameters for models using GridSearchCV.

    Parameters:
        X_train (array-like): Features for training.
        y_train (array-like): Labels for training.

    Returns:
        dict: Best hyperparameters for each model.
    """
    # Define hyperparameter grids
    param_grid_rf = {
        "n_estimators": [100, 150],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    }
    param_grid_gb = {
        "n_estimators": [100, 150],
        "learning_rate": [0.01, 0.05, 0.01],
        "max_depth": [5, 10],
    }

    # Initialize models
    models = {
        "random_forest": (RandomForestClassifier(random_state=42), param_grid_rf),
        "gradient_boosting": (GradientBoostingClassifier(random_state=42), param_grid_gb),
    }

    # Tune models
    best_params = {}
    for model_name, (model, param_grid) in models.items():
        print(f"Tuning {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params[model_name] = grid_search.best_params_
    
    # save best_params
    with open(f"{model_save_dir}/best_params_subset_{subset_id}.pkl", "wb") as file:
        pickle.dump(best_params, file)
    

    return best_params

# load and prepare data with second-order interactions, and do initail feature screening
def load_and_prepare_data(file_path, drop_columns=None, add_SOI=False):
    df = pd.read_csv(file_path)
    # rename variables 
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
        X = initial_screen_features_RF(X, y, threshold=0.02)
        print(f"Number of features after initial screening: {X.shape[1]}")
        
    # shuffle the dataset
    random_seed = np.random.randint(100000)
    dataset = pd.concat([X, y], axis=1).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    X = dataset.iloc[:, :-1]  # Features
    y = dataset.iloc[:, -1]   # Target

    return X, y, df

def add_second_order_interactions(X, sep=" * "):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X)
    
    # Modify interaction feature names to use custom separator
    original_names = X.columns
    raw_feature_names = poly.get_feature_names_out(original_names)
    
    # Replace space with desired separator
    custom_feature_names = [name.replace(" ", sep) for name in raw_feature_names]
    
    return pd.DataFrame(X_interactions, columns=custom_feature_names)




def plot_feature_importance(combined_df, model_name, model_save_dir, top_N, split_N):
    # Normalize color by importance
    norm = mcolors.Normalize(vmin=combined_df["Feature Importance"].min(), vmax=combined_df["Feature Importance"].max())
    cmap = sns.color_palette("crest", as_cmap=True)
    colors = cmap(norm(combined_df["Feature Importance"].values))

    # Sort bottom-up
    combined_df = combined_df.sort_values(by=["Votes", "Feature Importance"], ascending=[True, True])

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.barh(combined_df["Feature"], combined_df["Votes"], color=colors, edgecolor='black', linewidth=0.6)

    # Add value labels and size
    for bar, importance in zip(bars, combined_df["Feature Importance"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{importance:.3f}", va="center", ha="left", fontsize=20)


    # Styling
    ax.set_xlabel(f"Votes (Top {top_N})", fontsize=30, labelpad=15)
    if split_N > 1:
        title = f"Top Voted Features Across {split_N} Subsamples\n{model_name.replace('_', ' ').title()}"
    else:
        title = f"Top Voted Features Across All Samples\n{model_name.replace('_', ' ').title()}"
    
    ax.set_title(title,
                 fontsize=30, pad=20, weight='bold')
    # x and y tick label size
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    # set x limit
    ax.set_xlim(0, combined_df["Votes"].max() + 2)
    
    plt.yticks(rotation=30)
    
    # Ensure x-axis has integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Averaged Feature Importance", fontsize=20, labelpad=10)
    cbar.ax.tick_params(labelsize=15)

    # Improve layout and export
    plt.tight_layout()
    plt.savefig(f"{model_save_dir}/RF_feature_importance_votes.png", dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.savefig(f"{model_save_dir}/RF_feature_importance_votes.pdf", bbox_inches='tight', pad_inches=0.2)
    # plt.show()

    print(f"✅ Saved improved feature importance plots to:")
    print(f"   → {model_save_dir}/RF_feature_importance_votes.png")
    print(f"   → {model_save_dir}/RF_feature_importance_votes.pdf")

def main(file_path, load_existing_best_params, model_name, model_save_dir, drop_columns=None, add_SOI=False, scale=False):
    """
    Find robust feature importance rankings using shuffled and split subsets.

    Parameters:
        file_path (str): Path to the dataset.
        model_save_dir (str): Directory to save model parameters.

    Returns:
        DataFrame: Robust feature importance ranking.
    """
    # Step 1: Load and prepare data with second-order interactions
    X, y, _ = load_and_prepare_data(file_path, drop_columns, add_SOI)
    
    # train with whole dataset to get AUC
    random_seed = np.random.randint(100000)
    X_train_whole, X_test_whole, y_train_whole, y_test_whole = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    if model_name == 'random_forest':
        model_whole = RandomForestClassifier(random_state=random_seed)
    elif model_name == 'gradient_boosting':
        model_whole = GradientBoostingClassifier(random_state=random_seed)
    model_whole.fit(X_train_whole, y_train_whole)
    y_pred_whole = model_whole.predict(X_test_whole)        
    y_pred_proba_whole = model_whole.predict_proba(X_test_whole)[:, 1]  
    accuracy_whole = accuracy_score(y_test_whole, y_pred_whole)
    auc_score_whole = roc_auc_score(y_test_whole, y_pred_proba_whole)
    print(f"\nTrained on whole dataset, Accuracy: {accuracy_whole:.2f}, AUC: {auc_score_whole:.2f}")
    

    # Step 2: Split the shuffled dataset into subsets
    Split_N = 10
    subsets_X = np.array_split(X, Split_N)
    subsets_y = np.array_split(y, Split_N)
 
    all_feature_importances = []     # Initialize a list to store feature importance scores

    # Step 3: Iterate over the subsets
    random_seed = np.random.randint(100000)
    for i in range(Split_N):
        print(f"\nProcessing subset {i + 1}...")
        # Create training and test sets
        X_split = subsets_X[i]
        y_split = subsets_y[i]

        # Step 4: Tune hyperparameters on the training set
        if load_existing_best_params:
            with open(f"{model_save_dir}/best_params_subset_{i}.pkl", "rb") as file:
                best_params = pickle.load(file)
        else:
            best_params = tune_hyperparameters(X_split, y_split, i, model_save_dir)
        
        X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.2, random_state=random_seed)
        if scale:
            X_train, X_test = scale_data(X_train, X_test)

        # Step 5: Train a Random Forest model with the best parameters
        if model_name == 'random_forest':
            model = RandomForestClassifier(**best_params["random_forest"], random_state=random_seed)
            model.fit(X_train, y_train)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(**best_params["gradient_boosting"], random_state=random_seed)
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)        
        y_pred_proba = model.predict_proba(X_test)[:, 1]  
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"Split {i}, Accuracy: {accuracy:.2f}, AUC: {auc_score:.2f}")

        # Aggregate to determine feature importance
        tree_feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Feature Importance': model.feature_importances_
        })

        all_feature_importances.append(tree_feature_importances)  

    average_importance = pd.concat(all_feature_importances).groupby("Feature").mean().sort_values(by="Feature Importance", ascending=False)

    # Combine feature importance with votes
    Top_N = 10 #  it means as long as the feature is in the top 3 of any subset, it will be voted 
    feature_votes = {}
    for importance_df in all_feature_importances:
        top_features = importance_df.nlargest(Top_N, 'Feature Importance')['Feature']
        for feature in top_features:
            feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
    votes_df = pd.DataFrame(list(feature_votes.items()), columns=['Feature', 'Votes'])

    combined_df = votes_df.merge(average_importance, on='Feature', how='left')

    # Sort by Votes (descending) and Importance (descending)
    combined_df = combined_df[combined_df['Votes'] >= 1] # filter out features with low votes
    combined_df = combined_df.sort_values(by=['Votes', 'Feature Importance'], ascending=[False, False])
    print("\nFinal Feature Importance Values Ranking (Combined Votes and Feature Importance):")
    print(combined_df)

    combined_df = combined_df.head(5) 
    combined_df = combined_df.sort_values(by=['Votes', 'Feature Importance'], ascending=[True, True])
   
    # --------------------------------- Plotting ---------------------------------
    plot_feature_importance(combined_df, model_name, model_save_dir, Top_N, Split_N)

    
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python feature_importance_random_split_validation.py <file_path> <model_save_dir>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_save_dir = sys.argv[2]
    
    

    model_name = 'random_forest' # [random_forest, gradient_boosting]
    
        # Select features, dropping non-relevant columns
    drop_columns = ['count_frequency_inequality_words', 'AI_label', 'label_status', 'target', 'title', 'paper_abstract', # lables
                    'mixed', 'other', 'native_americans',
                    'acad_ineq_t-0', 'acad_ineq_t-1', 'acad_ineq_t-2', 
                    'news_ineq_t-0', 'news_ineq_t-1', 'news_ineq_t-2',
                    # 'acad_ineq_t-3', 'news_ineq_t-3',
                    'acad_ineq_3yr_avg', 'news_ineq_3yr_avg',
                    'female_score_min', 'female_score_max', 'first_author_female_score', 
                    # 'female_score_mean'
                    'year'
                    ]
    

    
    load_existing_best_params = True
    
    add_SOI = False # whether to add second-order interaction features
    
    scale = True

    main(file_path, load_existing_best_params, model_name, model_save_dir, drop_columns, add_SOI, scale)
    
    