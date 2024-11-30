import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Data Preparation
# Manually create the DataFrame based on the image data
data = {
    'Year': [2020, 2016, 2012, 2008, 2004, 2000, 1996, 1992, 1988, 1984, 1980, 1976],
    'Democratic_Percentage': [49.4, 45.1, 44.6, 45.1, 44.4, 44.7, 46.5, 36.5, 38.7, 32.5, 28.2, 39.8],
    'Republican_Percentage': [49.1, 48.7, 53.7, 53.6, 54.9, 51.0, 44.3, 38.5, 60.0, 66.4, 60.6, 56.4]
}

df = pd.DataFrame(data)

# 2. Feature Engineering
# Calculate additional features based on historical data
df['Year_Difference'] = df['Year'] - df['Year'].min()
df['Democratic_Change'] = df['Democratic_Percentage'].diff().fillna(0)
df['Republican_Change'] = df['Republican_Percentage'].diff().fillna(0)

# Define the features (X) and target (y)
X = df[['Year_Difference', 'Democratic_Change', 'Republican_Change']]
y_democratic = df['Democratic_Percentage']
y_republican = df['Republican_Percentage']

# 3. Train-Test Split
X_train, X_test, y_train_dem, y_test_dem = train_test_split(X, y_democratic, test_size=0.2, random_state=42)
X_train, X_test, y_train_rep, y_test_rep = train_test_split(X, y_republican, test_size=0.2, random_state=42)

# 4. Model Training and Prediction

# Initialize models
model_democratic = RandomForestRegressor(n_estimators=100, random_state=42)
model_republican = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
model_democratic.fit(X_train, y_train_dem)
model_republican.fit(X_train, y_train_rep)

# Evaluate models
dem_predictions = model_democratic.predict(X_test)
rep_predictions = model_republican.predict(X_test)

# Calculate the error
dem_mae = mean_absolute_error(y_test_dem, dem_predictions)
rep_mae = mean_absolute_error(y_test_rep, rep_predictions)

print(f"Mean Absolute Error for Democratic predictions: {dem_mae}")
print(f"Mean Absolute Error for Republican predictions: {rep_mae}")

# 5. Predict for 2024
# Create the input for 2024
X_2024 = pd.DataFrame({
    'Year_Difference': [2024 - df['Year'].min()],
    'Democratic_Change': [df['Democratic_Change'].mean()],  # You may use more sophisticated methods
    'Republican_Change': [df['Republican_Change'].mean()]
})

# Predict 2024 vote percentages
dem_2024_pred = model_democratic.predict(X_2024)[0]
rep_2024_pred = model_republican.predict(X_2024)[0]

print(f"Predicted Democratic Vote Percentage in 2024: {dem_2024_pred:.2f}%")
print(f"Predicted Republican Vote Percentage in 2024: {rep_2024_pred:.2f}%")
