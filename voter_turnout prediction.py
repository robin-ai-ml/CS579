import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# 1. Data Preparation

# Create the DataFrame
data = {
    'Year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'Population': [737884, 737185, 749808, 764776, 783087, 783621, 814971, 794611, 786168, 790643],
    'Citizenship_percentage': [92.7, 92.5, 92.7, 92.6, 92.4, 92.5, 92.6, 93.0, 93.1, 96.5],
    '>18_years_over': [582555, 584944, 595726, 607829, 622324, 627417, 660059, 649297.5, 638536, 639148],
    'Eligible_voters': [540028.485, 541073.2, 552238.002, 562849.654, 575027.376, 580360.725, 611214.634, 
                        603846.675, 594477.016, 616777.82],
    'Total_votes': [np.nan, 199776, np.nan, 324444, np.nan, 313699, np.nan, 417427, np.nan, 349283],
    'Voter_turnout_percent': [np.nan, 36.9, np.nan, 57.6, np.nan, 54.1, np.nan, 69.1, np.nan, 56.6]
}

df = pd.DataFrame(data)

# Display the DataFrame
print("Initial DataFrame:")
print(df)

# Handling Missing Values
# Since 'Voter_turnout_percent' is our target, we need to drop rows where it's missing
df_clean = df.dropna(subset=['Voter_turnout_percent']).reset_index(drop=True)

# Alternatively, if you want to include all data and predict missing values, you can handle it differently.
# For this example, we'll proceed with rows where 'Voter_turnout_percent' is available.

print("\nCleaned DataFrame (Rows with Voter Turnout Percent):")
print(df_clean)

# 2. Exploratory Data Analysis (Optional but recommended)

# Visualize correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(df_clean)
plt.show()

# 3. Feature Selection

# Define feature columns and target
feature_cols = ['Year', 'Population', 'Citizenship_percentage', '>18_years_over', 
                'Eligible_voters', 'Total_votes']
target_col = 'Voter_turnout_percent'

# Check if there are any missing values in feature columns
print("\nMissing values in feature columns:")
print(df_clean[feature_cols].isnull().sum())

# Handle missing values in features (Total_votes has some missing)
# Since 'Total_votes' is highly correlated with 'Voter_turnout_percent', it might be essential.
# We can choose to drop rows with missing 'Total_votes' or impute them.
# For simplicity, we'll drop rows with missing 'Total_votes'.
df_final = df_clean.dropna(subset=['Total_votes']).reset_index(drop=True)

print("\nFinal DataFrame after dropping rows with missing 'Total_votes':")
print(df_final)

# Define features and target
X = df_final[feature_cols]
y = df_final[target_col]

# 4. Model Training

# Split the data into training and testing sets
# Given the small dataset, we'll use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n{name} Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

# Select the best model based on R² Score
# For simplicity, let's choose Random Forest if it has better R²
# Otherwise, use Linear Regression
if r2_score(y_test, models['Random Forest Regressor'].predict(X_test)) > \
   r2_score(y_test, models['Linear Regression'].predict(X_test)):
    best_model = models['Random Forest Regressor']
    print("\nSelected Model: Random Forest Regressor")
else:
    best_model = models['Linear Regression']
    print("\nSelected Model: Linear Regression")

# 5. Prediction for 2024

# Prepare the input features for 2024
# You need to provide the expected or estimated values for the features in 2024.
# Since we don't have actual data, we'll make assumptions or use trends to estimate them.

# Example: Estimating 2024 features based on previous trends (simple linear extrapolation)

# Function to estimate next year's value based on previous years
def estimate_next_year(df, column):
    # Using linear trend: (last value - first value) / number of intervals
    n = len(df)
    slope = (df[column].iloc[-1] - df[column].iloc[0]) / (df['Year'].iloc[-1] - df['Year'].iloc[0])
    next_year_value = df[column].iloc[-1] + slope
    return next_year_value

# Estimate features for 2024
year_2024 = 2024
population_2024 = estimate_next_year(df_final, 'Population')
citizenship_percentage_2024 = estimate_next_year(df_final, 'Citizenship_percentage')
over_18_2024 = estimate_next_year(df_final, '>18_years_over')
eligible_voters_2024 = estimate_next_year(df_final, 'Eligible_voters')
total_votes_2024 = estimate_next_year(df_final, 'Total_votes')

# Create a DataFrame for the 2024 data
data_2024 = pd.DataFrame({
    'Year': [year_2024],
    'Population': [population_2024],
    'Citizenship_percentage': [citizenship_percentage_2024],
    '>18_years_over': [over_18_2024],
    'Eligible_voters': [eligible_voters_2024],
    'Total_votes': [total_votes_2024]
})

print("\nEstimated Features for 2024:")
print(data_2024)

# Predict voter turnout for 2024
voter_turnout_2024 = best_model.predict(data_2024)[0]

print(f"\nPredicted Voter Turnout Percentage for 2024: {voter_turnout_2024:.2f}%")

# Optional: Visualizing the prediction

# Combine the existing data with the prediction
df_prediction = df_clean.copy()
df_prediction = df_prediction.dropna(subset=['Total_votes']).reset_index(drop=True)
df_prediction = df_prediction.append({
    'Year': year_2024,
    'Population': population_2024,
    'Citizenship_percentage': citizenship_percentage_2024,
    '>18_years_over': over_18_2024,
    'Eligible_voters': eligible_voters_2024,
    'Total_votes': total_votes_2024,
    'Voter_turnout_percent': voter_turnout_2024
}, ignore_index=True)

# Plotting the historical and predicted voter turnout
plt.figure(figsize=(10,6))
sns.lineplot(data=df_prediction, x='Year', y='Voter_turnout_percent', marker='o')
plt.title('Voter Turnout Percentage Over Years')
plt.xlabel('Year')
plt.ylabel('Voter Turnout %')
plt.axvline(x=2024, color='red', linestyle='--', label='2024 Prediction')
plt.legend()
plt.show()
