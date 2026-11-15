import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Create API application
app = FastAPI()

# Load our trained model
model = joblib.load("house_price_model.joblib")

# Define the expected input
from pydantic import BaseModel, Field

class House(BaseModel):
    size_m2: float = Field(gt=0)

# Define prediction endpoint
@app.post("/predict")
def predict_price(house: House):

    # Prepare input
    X = pd.DataFrame({
        "size_m2": [house.size_m2]
    })

    # Make prediction
    prediction = model.predict(X)

    return {"predicted_price": float(prediction[0])}