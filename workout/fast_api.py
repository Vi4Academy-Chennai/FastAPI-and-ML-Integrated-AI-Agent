from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI(title="Vi4 Academy ML Predictor")

# Load the trained model into memory on startup
model = joblib.load("real_estate_model.joblib")

# Define the exact data schema expected from the AI Agent tool call
class PropertyFeatures(BaseModel):
    sqft: int
    bedrooms: int
    bathrooms: int
    age_of_house: int
    location_score: int

@app.post("/predict-price")
def predict_price(features: PropertyFeatures):
    # Convert the validated JSON payload into a 2D numpy array for Scikit-Learn
    input_data = np.array([[
        features.sqft,
        features.bedrooms,
        features.bathrooms,
        features.age_of_house,
        features.location_score
    ]])
    
    # Run the model inference
    prediction = model.predict(input_data)[0]
    
    # Return the exact JSON structure the AI Agent expects to read
    return {"estimated_price": int(prediction)}