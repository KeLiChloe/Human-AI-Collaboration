import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.neural_network import MLPClassifier

def initial_screen_features_RF(X, y, threshold):
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
    random_seed = np.random.randint(10000)
    rf_model = RandomForestClassifier(random_state=random_seed)
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
    print(importance_df.head(10))
    print(f"\nSelected {len(selected_features)} features based on importance > {threshold}")

    return X_filtered

def initial_screen_features_lasso(X, y, threshold):
    """
    Select important features based on LASSO coefficients.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        threshold (float): Regularization strength (LASSO parameter).

    Returns:
        DataFrame: Filtered feature matrix.
        list: Selected feature names.
    """

    # Train LASSO model
    random_seed = np.random.randint(10000)
    lasso_model = Lasso(alpha=threshold, random_state=random_seed)
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

def tune_hyperparameters(X_train, y_train, model_save_dir, add_SOI):
    """
    Tune hyperparameters for models using GridSearchCV.

    Parameters:
        X_train (array-like): Features for training.
        y_train (array-like): Labels for training.

    Returns:
        dict: Best hyperparameters for each model.
    """
    # ================== 1. Random Forest：适中搜索空间 ==================
    # 针对 100k / 10–15 feat / 4:6，给每个参数 1~3 个合理取值
    param_grid_rf = {
        "n_estimators": [200, 400],      # 树数：中等 & 稍大各一档
        "criterion": ["gini"],           # 不展开，保持稳定
        "max_depth": [12, 16, 24],       # 不用特别深，给 3 档
        "min_samples_split": [5, 10],    # 防过拟合，两档
        "min_samples_leaf": [3, 5],      # 叶子里 3 或 5 个样本
        "bootstrap": [True],
        "max_features": ["sqrt", 0.8],   # 两种常见特征采样方式
        "n_jobs": [-1],
    }

    # ================== 2. XGBoost：小而精的网格 ==================
    # 控制组合数量，同时覆盖“稳 & 稍激进”两种感觉
    param_grid_xgb = {
        "n_estimators": [300, 600],      # 轮数：中等 & 偏多
        "learning_rate": [0.05, 0.1],    # 小学习率 vs 稍大一点
        "max_depth": [4, 5],             # 深度适中，避免过拟合
        "subsample": [0.8],              # 稍微抽样，增加泛化
        "colsample_bytree": [0.8],       # 特征子采样
        "min_child_weight": [1, 5],      # 叶子最小权重：更灵活 vs 更保守
        "reg_lambda": [1.0],             # L2 正则固定住
        "reg_alpha": [0.0, 0.5],         # 是否加一点 L1
    }

    # ================== 3. MLP：结构 & 正则小范围搜索 ==================
    # 使用前记得对 X 标准化（StandardScaler）
    param_grid_mlp = {
        "hidden_layer_sizes": [
            (64,),
            (128,),
            (128, 64),    # 再多一层
        ],
        "activation": ["relu"],
        "learning_rate_init": [0.001, 0.0005],   # 稍快 & 稍慢两档
        "alpha": [0.0001, 0.0005, 0.001],        # L2 正则强度三档
        "batch_size": [256],
        "max_iter": [200, 300],                 # 提前停用不上就自动停
        "early_stopping": [True],
        "solver": ["adam"],
        "learning_rate": ["adaptive"],          # 让它自己调
        "validation_fraction": [0.1],
        "n_iter_no_change": [10],
        "shuffle": [True],
    }

    # ================== 4. 初始化模型（保持原来写法） ==================
    random_seed = np.random.randint(10000)
    models = {
        "random_forest": (
            RandomForestClassifier(random_state=random_seed),
            param_grid_rf
        ),
        "xgboost": (
            XGBClassifier(random_state=random_seed, eval_metric="logloss"),
            param_grid_xgb
        ),
        "mlp": (
            MLPClassifier(random_state=random_seed),
            param_grid_mlp
        )
    }

    # ================== 5. GridSearch 不变 ==================
    best_params = {}
    for model_name, (model, param_grid) in models.items():
        print(f"Tuning {model_name}...")
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_params[model_name] = grid_search.best_params_
    
    # save best_params
    with open(
        f"{model_save_dir}/best_params.pkl",
        "wb"
    ) as file:
        pickle.dump(best_params, file)
    
    return best_params


# Function to load and prepare data
def load_and_prepare_data(file_path, feature_columns):
    df = pd.read_csv(file_path)
    
    # Create the target variable
    df['target'] = df['count_inequality_words'].apply(lambda x: 1 if x > 0 else 0)
    # df['target'] = df["AI_label"].apply(lambda x: 1 if int(x) == 1 else 0)

    y = df['target']
    print(f"Positive class ratio: {y.sum()}/{len(y)} = {y.mean():.4f}")

    # Select features 
    X = df[feature_columns]
    return X, y, df

def add_second_order_interactions(X):
    # PolynomialFeatures with degree 2 for second-order interactions
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X)
    interaction_feature_names = poly.get_feature_names_out(input_features=X.columns)
    X = pd.DataFrame(X_interactions, columns=interaction_feature_names)
    
    return X

# Function to scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将 ndarray 转回 DataFrame，并保留列名
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled

# Function to train and predict models
def train_and_predict_models(X_train_scaled, y_train, X_test_scaled, best_params=None):
    """
    Train models using the best parameters and make predictions.

    Parameters:
        X_train_scaled (array-like): Scaled training features.
        y_train (array-like): Training labels.
        X_test_scaled (array-like): Scaled test features.
        best_params (dict): Tuned hyperparameters for each model (optional).

    Returns:
        dict: Predicted probabilities for each model.
    """
    random_seed = np.random.randint(10000)
    models = {
        "logistic_regression": LogisticRegression(max_iter=800),
        "random_forest": RandomForestClassifier(random_state=random_seed),
        "xgboost": XGBClassifier(random_state=random_seed,  eval_metric="logloss"),
        "mlp": MLPClassifier(random_state=random_seed),
        # "cart": DecisionTreeClassifier(random_state=random_seed),
    }

    trained_models = {}
    predictions = {}

    # Update models with tuned hyperparameters
    if best_params:
        for model_name in models.keys():
            if model_name in best_params:
                models[model_name].set_params(**best_params[model_name])

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        predictions[model_name] = model.predict_proba(X_test_scaled)[:, 1]

    return trained_models, predictions


# Function to calculate metrics for varying thresholds
def calculate_metrics(y_test, y_pred_proba):
    thresholds = np.arange(0.0, 1.05, 0.05)
    accuracy, precision, recall, f1 = [], [], [], []
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        accuracy.append(accuracy_score(y_test, y_pred_thresh))
        precision.append(precision_score(y_test, y_pred_thresh, zero_division=0))
        recall.append(recall_score(y_test, y_pred_thresh))
        f1.append(f1_score(y_test, y_pred_thresh))
    return thresholds, accuracy, precision, recall, f1

# Function to perform cross-validation on the train set
def cross_validate_with_metrics(X_train, y_train, n_splits=5):
    random_seed = np.random.randint(10000)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Initialize dictionaries to store metrics and ROC data

    cv_results = {
    "logistic_regression": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},
    "random_forest": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},
    "xgboost": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},
    "mlp": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},
    # "cart": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},  # Add CART here
    }

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        

        _, predictions = train_and_predict_models(X_train_fold, y_train_fold, X_val_fold)

        for model_name, y_pred_proba in predictions.items():
            _, acc, prec, rec, f1 = calculate_metrics(y_val_fold, y_pred_proba)
            cv_results[model_name]["accuracy"].append(acc)
            cv_results[model_name]["precision"].append(prec)
            cv_results[model_name]["recall"].append(rec)
            cv_results[model_name]["f1"].append(f1)
            cv_results[model_name]["auc"].append(roc_auc_score(y_val_fold, y_pred_proba))

    # Average metrics across folds
    for model_name, metrics in cv_results.items():
        for metric, values in metrics.items():
            cv_results[model_name][metric] = np.mean(values, axis=0)

    return cv_results

# Function to plot final metrics and ROC curves
def plot_metrics_and_roc(predictions, y_test, save_dir, add_SOI, feature_num):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    thresholds = np.arange(0.0, 1.05, 0.05)
    for ax, (model_name, y_pred_proba) in zip(axes[:3], predictions.items()):
        _, acc, prec, rec, f1 = calculate_metrics(y_test, y_pred_proba)
        ax.plot(thresholds, acc, label="Accuracy")
        ax.plot(thresholds, prec, label="Precision")
        ax.plot(thresholds, rec, label="Recall")
        ax.plot(thresholds, f1, label="F1 Score")
        ax.set_title(f"{model_name} (AUC = {roc_auc_score(y_test, y_pred_proba):.3f}) {'With SOI' if add_SOI else ''}(feature # = {feature_num})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Metric Value")
        ax.legend()
        ax.grid()

    # Plot ROC Curve
    ax_roc = axes[3]
    for model_name, y_pred_proba in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    ax_roc.set_title(f"ROC Curve for All Models (feature # = {feature_num})")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_roc.grid()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_and_roc{'_soi' if add_SOI else ''}.jpg")

    plt.show()
    

# Main function
def main(file_path, model_save_dir, add_SOI, use_best_params, feature_columns, scale):

    # Step 1: Load and prepare data
    X, y, _ = load_and_prepare_data(file_path, feature_columns)
    
    if add_SOI:
        X = add_second_order_interactions(X)
        X = initial_screen_features_lasso(X, y, threshold=0.005)

    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    if scale:
        X_train, X_test = scale_data(X_train, X_test)

    feature_names = X_train.columns.tolist()
    print(f"Number of features: {len(feature_names)}")

    # Step 4: Train on full train set and predict on test set
    if use_best_params:
        with open(f"{model_save_dir}/best_params.pkl", "rb") as file:
            best_params = pickle.load(file)
    else:
        print("Tuning hyperparameters...")
        best_params = tune_hyperparameters(X_train, y_train, model_save_dir, add_SOI)
        print("\nBest Hyperparameters:")
    for model_name, params in best_params.items():
        print(f"{model_name}: {params}")
    
    train_models, predictions = train_and_predict_models(X_train, y_train, X_test, best_params)
    
    print("\nTest Results:")
    for model_name, y_pred_proba in predictions.items():
        print(f"\nModel: {model_name}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        # print(f"Accuracy: {accuracy_score(y_test, y_pred_proba >= 0.5):.4f}")
        # print(f"Precision: {precision_score(y_test, y_pred_proba >= 0.5):.4f}")
        # print(f"Recall: {recall_score(y_test, y_pred_proba >= 0.5):.4f}")
        # print(f"F1 Score: {f1_score(y_test, y_pred_proba >= 0.5):.4f}")

    # Step 5: Plot results based on test set
    plot_metrics_and_roc(predictions, y_test, model_save_dir, add_SOI, X.shape[1])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python code/semantic_scholar_ML/new/step6_ml_classification.py <file_path> <model_save_dir>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_save_dir = sys.argv[2]
    
    add_SOI = False
    
    use_best_params = False
    
    scale = True
    
    feature_columns =  [
                       'female_mean', # 'female_max', 'female_min','first_author_female_score',
                       'natural_sciences', 'engineering_and_technology', 'social_sciences',
                       'country_race_shannon_entropy_mean', #'country_race_simpson_index_mean', 'country_race_inverse_dominance_mean',
                       'paper_race_shannon_entropy', #'paper_race_simpson_index', 'paper_race_inverse_dominance',
                        'white_composition', 'asian_composition', 'black_composition', 'hispanic_composition',
                        # 'acad_ineq_t-0', 'acad_ineq_t-1', 'acad_ineq_t-2', 'acad_ineq_t-3', 
                        'acad_ineq_3yr_avg', 
                        # 'news_ineq_t-0', 'news_ineq_t-1', 'news_ineq_t-2', 'news_ineq_t-3', 
                        'news_ineq_3yr_avg', 
                        # 'news_gender_ineq_t-0', 'news_gender_ineq_t-1', 'news_gender_ineq_t-2', 'news_gender_ineq_t-3', 'news_gender_ineq_3yr_avg', 
                        # 'news_econ_ineq_t-0', 'news_econ_ineq_t-1', 'news_econ_ineq_t-2', 'news_econ_ineq_t-3', 'news_econ_ineq_3yr_avg', 
                        # 'news_race_ineq_t-0', 'news_race_ineq_t-1', 'news_race_ineq_t-2', 'news_race_ineq_t-3', 'news_race_ineq_3yr_avg'
                        ]
    
    main(file_path, model_save_dir, add_SOI, use_best_params, feature_columns, scale)

