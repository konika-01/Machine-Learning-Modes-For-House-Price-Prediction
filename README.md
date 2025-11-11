### Machine-Learning-Modes-For-House-Price-Prediction

---
### `Abstract`

A key challenge for property sellers and buyers is determining the accurate sale price of a property. Predicting the property value helps investors and homebuyers plan their finances according to market trends. Property prices depend on various features such as property area, basement square footage, year built, number of bedrooms, and more.

This project applies supervised machine learning regression techniques to predict house prices based on multiple property attributes.

---

### `Problem Statement`

Use regression analysis to predict the sale price of a property using given features. The goal is to build a model that minimizes prediction error and generalizes well to unseen data.

--- 
### `Dataset`

Source: (Add source name, e.g., Kaggle — House Prices: Advanced Regression Techniques)

Rows: (add number if known)

Columns: Property features such as LotArea, YearBuilt, TotalBsmtSF, GrLivArea, BedroomAbvGr, etc.

---

### `Tools & Libraries Used`

1. Python
2. Pandas — Data cleaning and manipulation
3. NumPy — Numerical computations
4. Matplotlib / Seaborn — Data visualization and EDA
5. Scikit-learn (sklearn) — Model building and evaluation
6. Statsmodels — OLS regression analysis

---

### `Data Preprocessing Steps`

1. Handled null values — Imputed or dropped missing data.
2. Removed duplicates — Ensured data consistency.
3. Fixed data types — Converted incorrect data formats.
4. Separated numerical and categorical features — For suitable transformations.
5. Performed EDA (Exploratory Data Analysis) 
   Distribution plots and outlier detection.
   Correlation heatmaps to identify key predictors.
   Relationship analysis between features and target variable (SalePrice).

---

### `Model Building`

Models implemented:

1. Ordinary Least Squares (OLS) Regression
2. Linear Regression (Baseline Model)
3. Ridge Regression (Regularized Model)
4. Lasso Regression (Feature Shrinkage Model)
5. Train-Test Split: 70% training and 30% testing
6. x_train, x_test, y_train, y_test = train_test_split(x3, y3, test_size=0.3, random_state=42)

--- 

### `Model Performance`
Model	Train R²	Test R²	MAE	MSE	RMSE	Observation
Linear Regression	0.914	0.8856	16078.30	774,647,037.85	27,832.48	Slight overfitting observed
Ridge Regression (α=0.743)	0.9137	0.8863	—	—	—	Overfitting slightly minimized
Lasso Regression (α=0.306)	0.7599	0.7663	—	—	—	Underfits the data significantly

--- 

### `Inference`

The Linear Regression model performs well with an R² score of 0.8856 on the test data.
Ridge Regression slightly improves generalization by reducing overfitting.
Lasso Regression shows higher bias (underfitting), indicating it removes too many features.
Overall, Ridge Regression provides the most balanced performance.

--- 

✍️ Konika Malik.
