import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Generating synthetic real estate dataset...")
np.random.seed(42)
n_samples = 1000

# Generate realistic property features
sqft = np.random.randint(800, 5000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)
location = np.random.randint(1, 11, n_samples)

# Calculate a logical price baseline with some random market noise
base_price = 100000
price = (base_price + (sqft * 160) + (bedrooms * 12000) + 
         (bathrooms * 8000) - (age * 1500) + (location * 25000))
price = price + np.random.normal(0, 15000, n_samples) # Add market volatility

# Create DataFrame matching our expected feature names exactly
X = pd.DataFrame({
    'sqft': sqft, 
    'bedrooms': bedrooms, 
    'bathrooms': bathrooms, 
    'age_of_house': age, 
    'location_score': location
})
y = price

print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Export the trained model for the FastAPI server to use
joblib.dump(model, 'real_estate_model.joblib')
print("Model successfully trained and saved to 'real_estate_model.joblib'")