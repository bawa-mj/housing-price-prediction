import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load Data
df = pd.read_csv("Housing (1).csv")

# Binary Encoding
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Feature Engineering
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['area_per_room'] = df['area'] / df['total_rooms']
df.replace([np.inf, -np.inf], 0, inplace=True)

# Prepare Features
X = df.drop("price", axis=1)
y = np.log1p(df["price"])

# Preprocessor
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'total_rooms', 'area_per_room']
categorical_features = ['furnishingstatus']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/house_price_model.pkl")
print("Pipeline saved to models/house_price_model.pkl")