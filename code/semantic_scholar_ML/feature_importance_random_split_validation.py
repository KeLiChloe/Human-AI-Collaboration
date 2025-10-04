import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.cm as cm
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score

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
        # "gradient_boosting": (GradientBoostingClassifier(random_state=42), param_grid_gb),
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
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Create the target variable
    # df['target'] = df['count_frequency_inequality_words'].apply(lambda x: 1 if x > 0 else 0)
    df['target'] = df["AI_label"].apply(lambda x: 1 if x == 1 else 0)
    
    y = df['target']


    # Select features, dropping non-relevant columns
    drop_columns = ['count_frequency_inequality_words', 'AI_label', 'label_status', 'target', 'title', 'paper_abstract', # lables
                    # 'first_author_race_native_americans', 'first_author_race_mixed',
                    # 'first_author_race_other', # 'first_author_race_native_hawaiian_or_other_pacific_islander',
                    'mixed', 'other', 'native_americans',
                    'acad_ineq_t-0', 'acad_ineq_t-1', 'acad_ineq_t-2', 'acad_ineq_3yr_avg',
                    'news_ineq_t-0', 'news_ineq_t-1', 'news_ineq_t-2', 'news_ineq_3yr_avg',
                    # 'acad_ineq_t-3', 'news_ineq_t-3',
                    'year'
                    ]

    # Select features 
    X = df.drop(columns=drop_columns) 
    print(f"Initial number of features: {X.shape[1]}")
    
    # shuffle the dataset
    random_seed = np.random.randint(100000)
    dataset = pd.concat([X, y], axis=1).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    X = dataset.iloc[:, :-1]  # Features
    y = dataset.iloc[:, -1]   # Target

    return X, y, df

def add_second_order_interactions(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X)
    interaction_feature_names = poly.get_feature_names_out(input_features=X.columns)
    X = pd.DataFrame(X_interactions, columns=interaction_feature_names)
    return X

def main(file_path, load_existing_best_params, model_name, model_save_dir):
    """
    Find robust feature importance rankings using shuffled and split subsets.

    Parameters:
        file_path (str): Path to the dataset.
        model_save_dir (str): Directory to save model parameters.

    Returns:
        DataFrame: Robust feature importance ranking.
    """
    # Step 1: Load and prepare data with second-order interactions
    X, y, _ = load_and_prepare_data(file_path)

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
            with open(f"{model_save_dir}/best_params.pkl", "rb") as file:
                best_params = pickle.load(file)
        else:
            best_params = tune_hyperparameters(X_split, y_split, i, model_save_dir)
        
        X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.2, random_state=random_seed)

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
    combined_df = combined_df[combined_df['Votes'] > 1] # filter out features with low votes
    combined_df = combined_df.sort_values(by=['Votes', 'Feature Importance'], ascending=[False, False])
    print("\nFinal Feature Importance Values Ranking (Combined Votes and Feature Importance):")
    print(combined_df)

    combined_df = combined_df.head(10) # top 5 features
    combined_df = combined_df.sort_values(by=['Votes', 'Feature Importance'], ascending=[True, True])
   
    
    colors = cm.viridis(np.linspace(0, 1, len(combined_df)))

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    # Bar plot for Votes with gradient colors
    bars = ax.barh(combined_df["Feature"], combined_df["Votes"], color=colors, label="Votes")

    # 设置标签
    ax.set_xlabel("Votes", fontsize=20)
    # ax.set_ylabel("Features", fontsize=20)
    ax.set_title(f"{model_save_dir.split('/')[1].upper()} Inequality \n({model_name})", fontsize=16, fontweight='bold')

    # 确保 y 轴标签显示完整
    ax.set_yticks(np.arange(len(combined_df["Feature"])))  
    ax.set_yticklabels(combined_df["Feature"], fontsize=25, rotation=0)

    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{model_save_dir}/feature_importance_votes.png")
    
    
    
    # part 2: checking directions with/without second order interactions 
    # X_filtered = X[combined_df['Feature'].tolist()] # select features with high votes (> 1)
    # X_filtered = add_second_order_interactions(X_filtered)
    X_filtered = X

    # 'C' is inverse of alpha; lower values increase regularization
    logistic = LogisticRegression(penalty='l1', solver='liblinear', C=10, random_state=random_seed) 
    # logistic = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.05, random_state=42)
 
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=random_seed)
    logistic.fit(X_train, y_train)


    print("\nLogistic Coefficients:")
    lasso_coefficients = pd.DataFrame({
        'Feature': X_filtered.columns,
        'Coefficient': logistic.coef_[0]
    }).sort_values(by='Coefficient', key=lambda x: abs(x), ascending=True)

    # Predict on test set
    y_pred = logistic.predict(X_test)
    y_pred_prob = logistic.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate positive and negative coefficients
    # only keep top 15 absolute coefficients
    lasso_coefficients = lasso_coefficients.reindex(lasso_coefficients['Coefficient'].abs().sort_values(ascending=True).tail(10).index)
    lasso_coefficients['Sign'] = np.where(lasso_coefficients['Coefficient'] > 0, 1, -1)

    # Bar plot with direction
    for index, row in lasso_coefficients.iterrows():
        if row['Coefficient'] != 0:
            coef = row['Coefficient']
            feature = row['Feature']
            color = 'blue' if coef > 0 else 'red'  # Choose colors for positive and negative
            ax.barh(feature, coef, color=color)
    
    ax.tick_params(axis='y', labelsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)

    # Add a vertical line at 0 to separate positive and negative contributions
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)

    # Add labels and title
    ax.set_xlabel('Coefficient Value', fontsize=20)
    # ax.set_ylabel('Features', fontsize=14)
    ax.set_title(f"{model_save_dir.split('/')[1].upper()} Logistic Regression", fontsize=16, fontweight='bold')

    # Improve layout and readability
    plt.tight_layout()
    plt.savefig(f"{model_save_dir}/logit_lasso_feature_importance.png")


    
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python feature_importance_random_split_validation.py <file_path> <model_save_dir>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_save_dir = sys.argv[2]
    
    load_existing_best_params = False

    model_name = 'random_forest' # [random_forest, gradient_boosting]

    main(file_path, load_existing_best_params, model_name, model_save_dir)
    
    