import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

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
    logger.info("Prediction requested for %.2f m2", house.size_m2)
    if house.size_m2 < 50 or house.size_m2 > 200:
        logger.warning(
            "House size %.2f m2 is outside the training range!",
            house.size_m2
    )
    input_data = pd.DataFrame({"size_m2": [house.size_m2]})
    prediction = model.predict(input_data)

    logger.info("Predicted price: $%.2f", prediction[0])

    return {"predicted_price": float(prediction[0])}
    # Prepare input
    X = pd.DataFrame({
        "size_m2": [house.size_m2]
    })

    # Make prediction
    prediction = model.predict(X)

    return {"predicted_price": float(prediction[0])}