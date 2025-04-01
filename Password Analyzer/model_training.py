import numpy as np 
import pandas as pd
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

dataPath = r"C:\Users\shalu\Downloads\rockyou.txt\rockyou.txt"
passwords = pd.read_csv(dataPath, header=None, names=['password'], encoding='ISO-8859-1', on_bad_lines='skip')
passwords.drop_duplicates(inplace=True)
passwords.dropna(inplace=True)
passwords.reset_index(drop=True, inplace=True)

def calculate_entropy(password):
    if not password:
        return 0
    counts = Counter(password)
    length = len(password)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts.values())
    return entropy

passwords['entropy'] = passwords['password'].apply(calculate_entropy)
passwords['length'] = passwords['password'].str.len()
passwords['unique_chars'] = passwords['password'].apply(lambda x: len(set(x)))
passwords['uppercase_count'] = passwords['password'].apply(lambda x: sum(1 for char in x if char.isupper()))
passwords['lowercase_count'] = passwords['password'].apply(lambda x: sum(1 for char in x if char.islower()))
passwords['digit_count'] = passwords['password'].apply(lambda x: sum(1 for char in x if char.isdigit()))
passwords['special_count'] = passwords['password'].apply(lambda x: sum(1 for char in x if not char.isalnum()))

X = passwords[['length', 'unique_chars', 'uppercase_count', 'lowercase_count', 'digit_count', 'special_count']]
y = passwords['entropy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)

print("Random Forest Regressor Performance:")
print(f"Mean Squared Error: {rf_mse:.4f}")
print(f"R² Score: {rf_r2:.4f}")
print("\nXGBoost Regressor Performance:")
print(f"Mean Squared Error: {xgb_mse:.4f}")
print(f"R² Score: {xgb_r2:.4f}")

joblib.dump(rf_model, r"C:\Users\shalu\Downloads\random_forest_model.pkl")
joblib.dump(xgb_model, r"C:\Users\shalu\Downloads\xgboost_model.pkl")
joblib.dump(scaler, r"C:\Users\shalu\Downloads\scaler.pkl")
print("Models and scaler saved successfully.")
