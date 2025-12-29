# Car Price Prediction and Sale Probability Model

## Project Information

**Course:** CMPE-442 Machine Learning  
**Student:** Ahmet Alp Malkoç  
**Supervisor:** Venera Atagoziev

## Overview

This project implements machine learning models to predict car prices and estimate sale probability based on various car features. The solution uses ensemble methods, specifically XGBoost, to build regression models for both price prediction and sale probability estimation.

The project uses machine learning to predict the selling price and sale probability of used cars in the UK, based on their specifications. It combines predictive modeling with explainability tools to ensure fairness, transparency, and actionable insights for both sellers and buyers.

**Dataset Source:** Kaggle - UK Used Car Dataset  
**Dataset Size:** ~100,000 entries after cleaning  
**Features:** Brand, Model, Year, Mileage, MPG, Engine Size, Transmission, Fuel Type, Tax

## Objectives

1. Predict car selling price (regression)
2. Estimate probability of sale (regression + classification)
3. Make the models explainable using SHAP and LIME

## Features

- **Price Prediction Model**: Predicts car prices using features such as year, mileage, tax, mpg, engine size, model, transmission, fuel type, and brand
- **Sale Probability Model**: Estimates the probability that a car will be sold based on price deviation from market average and popularity scores
- **Model Explainability**: Includes SHAP and LIME explanations for model interpretability

## Requirements

### Python Version
Python 3.7 or higher

### Required Packages

```
numpy
pandas
scikit-learn
xgboost
lightgbm
matplotlib
```

### Optional Packages (for model explainability)

```
shap
lime
```

Install all packages using:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib shap lime
```

## Dataset

The project expects a dataset in the following format:
- A zip file named `dataset.zip` containing CSV files
- Each CSV file should be named after the car brand (e.g., `Toyota.csv`, `BMW.csv`)
- Each CSV file should contain car data with columns: year, mileage, tax, mpg, engineSize, model, transmission, fuelType, price

## Usage

1. Place your `dataset.zip` file in the project directory

2. Run the training script:

```bash
python training_carprice&saleprob.py
```

The script will:
- Extract and load the dataset
- Preprocess the data (handle missing values, add brand information)
- Train and evaluate multiple models for price prediction
- Calculate sale probability scores
- Train a model to predict sale probability
- Generate model explanations (if SHAP/LIME are installed)
- Save visualization plots

## Project Structure

```
MachineLearningProject/
├── training_carprice&saleprob.py  # Main training script
├── dataset.zip                     # Dataset file (user provided)
└── README.md                       # This file
```

## Model Details

### Price Prediction Model
- **Best Model:** XGBoost
- Uses XGBoost regressor with hyperparameter tuning via GridSearchCV
- Preprocessing includes standardization of numerical features and one-hot encoding of categorical features
- Evaluated using RMSE, MAE, and R² metrics
- **Price Prediction R²:** 0.9579

### Sale Probability Model
- Calculates sale probability based on:
  - Price deviation from group average (Brand, Model, Year)
  - Popularity score combining brand reliability and normalized mileage
- Uses sigmoid function to convert raw scores to probabilities
- Trained using XGBoost regressor
- Evaluated using both regression metrics (RMSE, MAE, R²) and classification metrics (Precision, Recall, F1)
- **Sale Probability R²:** 0.6939
- **F1 Score:** 0.9900 (binary sale classification)

### Preprocessing
- One-hot encoding for categorical features
- StandardScaler for numerical features
- Median/Mode imputation for missing values
- Custom sale_probability target using price deviation and brand/mileage-based popularity

### Explainability
- **SHAP:** Used for global, local, and class-wise explanations
- **LIME:** Provided localized reasoning for specific predictions
- **Key Insights:**
  - Newer, fuel-efficient cars with low mileage are more likely to sell
  - Functional attributes outweigh brand in model decisions

## Output Files

The script generates the following output files:
- `shap_summary_price.png` - SHAP summary plot for price prediction model
- `shap_summary_saleprob.png` - SHAP summary plot for sale probability model
- `partial_dependence.png` - Partial dependence plots for key features

## Notes

- The script uses a sample of 10,000 records for faster model evaluation during development
- Final models are trained on the full training dataset
- Random state is set to 42 for reproducibility
- If SHAP/LIME are not installed, the script will skip explainability sections with a warning
