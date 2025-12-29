# -*- coding: utf-8 -*-
"""
Car Price Prediction and Sale Probability Model
Authors: Ahmet Alp Malkoç
CMPE-442 Car Price Prediction and Sale Probability
"""

# Standard library imports
import os
import glob
import zipfile
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score
)
from sklearn.inspection import PartialDependenceDisplay

# Model imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Optional imports for explainability (install with: pip install shap lime)
try:
    import shap
    import lime.lime_tabular
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP and LIME not installed. Install with: pip install shap lime")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_extract_data(zip_path="dataset.zip", extract_dir="dataset"):
    """
    Load and extract dataset from zip file.
    
    Args:
        zip_path: Path to the zip file containing the dataset
        extract_dir: Directory to extract the data to
        
    Returns:
        None (extracts files to directory)
    """
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Extracted {zip_path} to {extract_dir}")
    else:
        print(f"⚠️  {zip_path} not found. Assuming data is already extracted.")


def find_csv_files(data_dir="dataset"):
    """
    Find all CSV files in the dataset directory.
    
    Args:
        data_dir: Directory to search for CSV files
        
    Returns:
        List of CSV file paths
    """
    # Try non-recursive first
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # Try recursive if no files found
    if not csv_files:
        csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    
    print(f"📁 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   └── {f}")
    
    return csv_files


def load_dataframe(csv_files):
    """
    Load and combine CSV files into a single DataFrame with Brand column.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        Combined pandas DataFrame
    """
    if not csv_files:
        raise ValueError("No CSV files found. Check your path and folder structure.")
    
    df_list = []
    for f in csv_files:
        brand = os.path.basename(f).split('.')[0]
        temp_df = pd.read_csv(f)
        temp_df['Brand'] = brand
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Combined DataFrame shape: {df.shape}")
    return df


def preprocess_data(df):
    """
    Preprocess the DataFrame: handle missing values and clean columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    # Drop duplicate 'tax(£)' column if it exists
    df = df.drop(columns=['tax(£)'], errors='ignore')
    
    # Impute missing numerical values using median
    df['tax'] = df['tax'].fillna(df['tax'].median())
    df['mpg'] = df['mpg'].fillna(df['mpg'].median())
    
    print("✅ Data preprocessing completed")
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    
    return df


def create_preprocessor(num_features, cat_features):
    """
    Create a preprocessing pipeline for numerical and categorical features.
    
    Args:
        num_features: List of numerical feature names
        cat_features: List of categorical feature names
        
    Returns:
        ColumnTransformer preprocessor
    """
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    return preprocessor


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def evaluate_models(X_train, y_train, preprocessor, sample_size=10000, random_state=42):
    """
    Evaluate multiple models using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing pipeline
        sample_size: Number of samples to use for faster evaluation
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with model evaluation results
    """
    # Sample for faster evaluation
    X_train_sample = X_train.sample(min(sample_size, len(X_train)), random_state=random_state)
    y_train_sample = y_train.loc[X_train_sample.index]
    
    # Define models
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_state),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=random_state),
        'XGBoost': XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=random_state),
        'LightGBM': LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=random_state),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        pipe = make_pipeline(preprocessor, model)
        rmse_scores = cross_val_score(
            pipe, X_train_sample, y_train_sample, 
            cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        results[name] = {
            'RMSE (CV mean)': -np.mean(rmse_scores),
            'RMSE (std)': np.std(rmse_scores)
        }
    
    results_df = pd.DataFrame(results).T.sort_values('RMSE (CV mean)')
    print("\nModel Evaluation Results:")
    print(results_df)
    return results_df


def train_price_model(X_train, y_train, X_test, y_test, preprocessor, 
                      sample_size=10000, random_state=42):
    """
    Train the best model for price prediction using grid search.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        preprocessor: Preprocessing pipeline
        sample_size: Number of samples to use for grid search
        random_state: Random state for reproducibility
        
    Returns:
        Trained model and evaluation metrics
    """
    # Sample for faster grid search
    X_train_sample = X_train.sample(min(sample_size, len(X_train)), random_state=random_state)
    y_train_sample = y_train.loc[X_train_sample.index]
    
    # Define pipeline
    pipe = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('regressor', XGBRegressor(random_state=random_state))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__subsample': [0.8, 1.0]
    }
    
    # Grid search
    print("Running grid search for price prediction model...")
    grid_search = GridSearchCV(
        pipe, param_grid, cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_sample, y_train_sample)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best RMSE (CV): {-grid_search.best_score_:.2f}")
    
    # Fit on full training set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return best_model, {'rmse': rmse, 'mae': mae, 'r2': r2}


# ============================================================================
# SALE PROBABILITY CALCULATION
# ============================================================================

def calculate_sale_probability(df, alpha=2, beta=1):
    """
    Calculate sale probability for each car based on price deviation and popularity.
    
    Args:
        df: DataFrame with car data
        alpha: Weight for price deviation
        beta: Weight for popularity score
        
    Returns:
        DataFrame with added sale probability columns
    """
    # Brand reliability scores (0 to 1 scale)
    brand_reliability = {
        'Toyota': 0.9, 'Honda': 0.85, 'Ford': 0.7, 'Mercedes': 0.5,
        'Jaguar': 0.4, 'BMW': 0.6, 'Hyundai': 0.75, 'Kia': 0.8,
        'VW': 0.65, 'Vauxhall': 0.7, 'Audi': 0.55, 'Nissan': 0.78
    }
    
    # Step 1: Price deviation from group average
    group_avg_price = df.groupby(['Brand', 'model', 'year'])['price'].transform('mean')
    df['price_deviation'] = (group_avg_price - df['price']) / group_avg_price
    
    # Step 2: Popularity score based on reliability & mileage
    df['brand_score'] = df['Brand'].map(brand_reliability).fillna(0.6)
    df['mileage_normalized'] = 1 - (df['mileage'] / df['mileage'].max())
    df['popularity_score'] = 0.6 * df['brand_score'] + 0.4 * df['mileage_normalized']
    
    # Step 3: Final sale probability using sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    df['raw_sale_score'] = (alpha * df['price_deviation']) + (beta * df['popularity_score'])
    df['sale_probability'] = sigmoid(df['raw_sale_score'])
    
    print("\nSale Probability Statistics:")
    print(df['sale_probability'].describe())
    
    return df


def train_sale_probability_model(X_train, y_train, X_test, y_test, preprocessor,
                                  sample_size=10000, random_state=42):
    """
    Train model to predict sale probability.
    
    Args:
        X_train: Training features
        y_train: Training target (sale probability)
        X_test: Test features
        y_test: Test target
        preprocessor: Preprocessing pipeline
        sample_size: Number of samples to use for grid search
        random_state: Random state for reproducibility
        
    Returns:
        Trained model and evaluation metrics
    """
    # Sample for faster grid search
    X_train_sample = X_train.sample(min(sample_size, len(X_train)), random_state=random_state)
    y_train_sample = y_train.loc[X_train_sample.index]
    
    # Define pipeline
    pipe_prob = make_pipeline(preprocessor, XGBRegressor(random_state=random_state))
    
    # Hyperparameter grid
    param_grid_prob = {
        'xgbregressor__n_estimators': [100, 200],
        'xgbregressor__max_depth': [3, 5, 7],
        'xgbregressor__learning_rate': [0.05, 0.1],
        'xgbregressor__subsample': [0.8, 1.0]
    }
    
    # Grid search
    print("Running grid search for sale probability model...")
    grid_search_prob = GridSearchCV(
        pipe_prob, param_grid_prob, cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search_prob.fit(X_train_sample, y_train_sample)
    
    print(f"\nBest parameters: {grid_search_prob.best_params_}")
    print(f"Best RMSE (CV): {-grid_search_prob.best_score_:.4f}")
    
    # Fit on full training set
    best_model = grid_search_prob.best_estimator_
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Classification metrics (using 0.5 threshold)
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_test_binary = (y_test > 0.5).astype(int)
    
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    
    print(f"\nClassification Metrics (threshold=0.5):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    metrics = {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'precision': precision, 'recall': recall, 'f1': f1
    }
    
    return best_model, metrics


# ============================================================================
# MODEL EXPLAINABILITY
# ============================================================================

def generate_shap_explanations(model, X_test, preprocessor, model_type='price'):
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model (pipeline)
        X_test: Test features
        preprocessor: Preprocessing pipeline (extracted from model)
        model_type: Type of model ('price' or 'sale_probability')
        
    Returns:
        SHAP explainer and values
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return None, None
    
    # Extract components from pipeline
    if model_type == 'price':
        regressor = model.named_steps['regressor']
        preprocessor_used = model.named_steps['preprocessing']
    else:
        regressor = model.named_steps['xgbregressor']
        preprocessor_used = model.named_steps['columntransformer']
    
    # Transform test data
    X_test_transformed = preprocessor_used.transform(X_test)
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    
    feature_names = preprocessor_used.get_feature_names_out()
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    # Create SHAP explainer
    explainer = shap.Explainer(regressor, X_test_df)
    shap_values = explainer(X_test_df)
    
    # Generate summary plot
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.savefig(f"shap_summary_{model_type}.png", bbox_inches='tight')
    plt.close()
    print(f"✅ Saved SHAP summary plot: shap_summary_{model_type}.png")
    
    return explainer, shap_values


def generate_lime_explanation(model, X_test, preprocessor, instance_idx=12):
    """
    Generate LIME explanation for a specific instance.
    
    Args:
        model: Trained model (pipeline)
        X_test: Test features
        preprocessor: Preprocessing pipeline
        instance_idx: Index of instance to explain
        
    Returns:
        LIME explanation object
    """
    if not SHAP_AVAILABLE:
        print("LIME not available. Install with: pip install lime")
        return None
    
    # Extract components
    regressor = model.named_steps['xgbregressor']
    preprocessor_used = model.named_steps['columntransformer']
    
    # Transform test data
    X_test_transformed = preprocessor_used.transform(X_test)
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    
    feature_names = preprocessor_used.get_feature_names_out()
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    # Create LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test_df.values,
        feature_names=X_test_df.columns.tolist(),
        mode='regression'
    )
    
    # Explain instance
    lime_exp = lime_explainer.explain_instance(
        X_test_df.iloc[instance_idx].values,
        regressor.predict
    )
    
    return lime_exp


def plot_partial_dependence(model, X_test, features=['mileage', 'year', 'engineSize']):
    """
    Plot partial dependence plots for specified features.
    
    Args:
        model: Trained model
        X_test: Test features
        features: List of feature names to plot
    """
    PartialDependenceDisplay.from_estimator(model, X_test, features)
    plt.savefig("partial_dependence.png", bbox_inches='tight')
    plt.close()
    print("✅ Saved partial dependence plot: partial_dependence.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("Car Price Prediction and Sale Probability Model")
    print("Authors: Ahmet Alp Malkoç, Ceren Kızılırmak")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = "dataset"
    ZIP_PATH = "dataset.zip"
    RANDOM_STATE = 42
    SAMPLE_SIZE = 10000
    
    # Step 1: Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    load_and_extract_data(ZIP_PATH, DATA_DIR)
    csv_files = find_csv_files(DATA_DIR)
    df = load_dataframe(csv_files)
    df = preprocess_data(df)
    
    # Step 2: Prepare features for price prediction
    print("\n[2/6] Preparing features for price prediction...")
    num_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    cat_features = ['model', 'transmission', 'fuelType', 'Brand']
    preprocessor = create_preprocessor(num_features, cat_features)
    
    X = df.drop(columns=['price'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Step 3: Evaluate models and train price prediction model
    print("\n[3/6] Training price prediction model...")
    evaluate_models(X_train, y_train, preprocessor, SAMPLE_SIZE, RANDOM_STATE)
    price_model, price_metrics = train_price_model(
        X_train, y_train, X_test, y_test, preprocessor, SAMPLE_SIZE, RANDOM_STATE
    )
    
    # Step 4: Calculate sale probability
    print("\n[4/6] Calculating sale probability...")
    df = calculate_sale_probability(df)
    
    # Step 5: Train sale probability model
    print("\n[5/6] Training sale probability model...")
    X_prob = df.drop(columns=['price', 'sale_probability', 'sale_likely', 'raw_sale_score'], 
                     errors='ignore')
    y_prob = df['sale_probability']
    
    Xpr_train, Xpr_test, ypr_train, ypr_test = train_test_split(
        X_prob, y_prob, test_size=0.2, random_state=RANDOM_STATE
    )
    
    sale_model, sale_metrics = train_sale_probability_model(
        Xpr_train, ypr_train, Xpr_test, ypr_test, preprocessor, SAMPLE_SIZE, RANDOM_STATE
    )
    
    # Step 6: Generate explanations (optional)
    print("\n[6/6] Generating model explanations...")
    if SHAP_AVAILABLE:
        print("Generating SHAP explanations...")
        generate_shap_explanations(price_model, X_test, preprocessor, 'price')
        generate_shap_explanations(sale_model, Xpr_test, preprocessor, 'sale_probability')
        
        # Generate LIME explanation
        print("Generating LIME explanation...")
        lime_exp = generate_lime_explanation(sale_model, Xpr_test, preprocessor)
        if lime_exp:
            print("✅ LIME explanation generated")
    else:
        print("Skipping SHAP/LIME explanations (not installed)")
    
    # Plot partial dependence
    plot_partial_dependence(price_model, X_test)
    
    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)
    
    return price_model, sale_model, df


if __name__ == "__main__":
    price_model, sale_model, df = main()
