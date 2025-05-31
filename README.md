This project uses machine learning to predict the selling price and sale probability of used cars in the UK, based on their specifications. It combines predictive modeling with explainability tools to ensure fairness, transparency, and actionable insights for both sellers and buyers.
	•	Source: Kaggle - UK Used Car Dataset
	•	Size: ~100,000 entries after cleaning
	•	Features: Brand, Model, Year, Mileage, MPG, Engine Size, Transmission, Fuel Type, Tax

 Objectives:
 	1.	Predict car selling price (regression)
	2.	Estimate probability of sale (regression + classification)
	3.	Make the models explainable using SHAP and LIME

  Modeling
	•	Best Model: XGBoost
	•	Price Prediction R²: 0.9579
	•	Sale Probability R²: 0.6939
	•	F1 Score: 0.9900 (binary sale classification)

 Explainability
	•	SHAP: Used for global, local, and class-wise explanations
	•	LIME: Provided localized reasoning for specific predictions
	•	Insights:
	•	Newer, fuel-efficient cars with low mileage are more likely to sell
	•	Functional attributes outweigh brand in model decisions

 Preprocessing
	•	One-hot encoding for categorical features
	•	StandardScaler for numericals
	•	Median/Mode imputation for missing values
	•	Custom sale_probability target using price deviation and brand/mileage-based popularity
