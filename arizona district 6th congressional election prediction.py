import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 1: Data Preparation
# Creating the DataFrame based on the provided data
data = {
    'Year': [2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022],
    'Republican_Percentage': [65.94, 79.38, 74.8, 62.42, 66.42, 61.3, 64.86, 62.13, 55.19, 52.2, 50.73],
    'Democratic_Percentage': [31.57, 0, 0, 34.55, 29.12, 33.31, 35.14, 37.87, 44.81, 47.8, 49.24]
}

df = pd.DataFrame(data)

# Step 2: Feature Engineering
# Adding feature columns for trends over time
df['Year_Difference'] = df['Year'] - df['Year'].min()
df['Republican_Change'] = df['Republican_Percentage'].diff().fillna(0)
df['Democratic_Change'] = df['Democratic_Percentage'].diff().fillna(0)

# Define features (X) and targets for vote percentage predictions
X = df[['Year_Difference', 'Republican_Change', 'Democratic_Change']]
y_rep = df['Republican_Percentage']
y_dem = df['Democratic_Percentage']

# Step 3: Train-Test Split
# Splitting data for training and testing
X_train, X_test, y_train_rep, y_test_rep = train_test_split(X, y_rep, test_size=0.2, random_state=42)
X_train, X_test, y_train_dem, y_test_dem = train_test_split(X, y_dem, test_size=0.2, random_state=42)

# Step 4: Model Training for Vote Percentage Prediction
# Using Random Forest Regressor to predict vote percentages for each party
model_rep = RandomForestRegressor(n_estimators=100, random_state=42)
model_dem = RandomForestRegressor(n_estimators=100, random_state=42)

# Fitting the models
model_rep.fit(X_train, y_train_rep)
model_dem.fit(X_train, y_train_dem)

# Step 5: Evaluating Models
# Making predictions on the test set
rep_predictions = model_rep.predict(X_test)
dem_predictions = model_dem.predict(X_test)

# Calculating Mean Absolute Error
rep_mae = mean_absolute_error(y_test_rep, rep_predictions)
dem_mae = mean_absolute_error(y_test_dem, dem_predictions)

print(f"Mean Absolute Error for Republican predictions: {rep_mae:.2f}")
print(f"Mean Absolute Error for Democratic predictions: {dem_mae:.2f}")

# Step 6: Predicting for 2024
# Preparing the input features for the year 2024
X_2024 = pd.DataFrame({
    'Year_Difference': [2024 - df['Year'].min()],
    'Republican_Change': [df['Republican_Change'].mean()],  # Average change for simplicity
    'Democratic_Change': [df['Democratic_Change'].mean()]   # Average change for simplicity
})

# Predicting vote percentages for 2024
rep_2024_pred = model_rep.predict(X_2024)[0]
dem_2024_pred = model_dem.predict(X_2024)[0]

# Determining the winner based on predicted vote percentages
winner_2024 = 'Republican' if rep_2024_pred > dem_2024_pred else 'Democratic'

print(f"\nPredicted Republican Vote Percentage in 2024: {rep_2024_pred:.2f}%")
print(f"Predicted Democratic Vote Percentage in 2024: {dem_2024_pred:.2f}%")
print(f"Predicted Winner in 2024: {winner_2024}")
