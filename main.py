import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset 
df = pd.read_csv('Salary_Data.csv')

# Print the first few rows of the dataframe to understand its structure
print("Preview of the data:")
print(df.head())

# Basic cleaning: Remove rows with missing values (NaN)
# For an advanced project, we would replace with the mean, but here we remove to simplify.
df = df.dropna()

# ==========================================
# 3. PREPROCESSING
# ==========================================

# A. Encoding categorical variables (e.g., City, Gender)
# pandas.get_dummies transforms text into columns of 0s and 1s
# Replace the columns below with those in your dataset that contain text
text_columns = ['Gender', 'Education Level', 'Job Title']  
df = pd.get_dummies(df, columns=text_columns, drop_first=True)

# B. Definition of Features (X) and Target (y)
target = 'Salary'  

X = df.drop(target, axis=1) # All except the target
y = df[target]              # Only the target
# C. Train/Test Split
# Keep 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# D. Normalization (Scaling)
scaler = StandardScaler()

# Fit the scaler on the training data and apply it to both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. MODELING
# ==========================================

# --- Model 1: Linear Regression ---
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train) # Training
y_pred_lr = model_lr.predict(X_test_scaled) # Prediction

# --- Model 2: Decision Tree ---
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train_scaled, y_train) # Training
y_pred_dt = model_dt.predict(X_test_scaled) # Prediction

# ==========================================
# 5. EVALUATION AND COMPARISON
# ==========================================

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- Performance of {name} ---")
    print(f"MSE (Mean Squared Error) : {mse:.2f}")
    print(f"R2 Score (Closeness to reality) : {r2:.4f}")
    print("-" * 30)

# Display results
print("\n=== FINAL RESULTS ===\n")
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Decision Tree", y_test, y_pred_dt)

# TODO : Write a concluding sentence here to say which model is better.