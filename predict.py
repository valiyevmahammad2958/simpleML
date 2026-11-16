import joblib
import pandas as pd

# Load the trained model
model = joblib.load("house_price_model.joblib")

# Create a new house
new_house = pd.DataFrame({
    "size_m2": [120]
})

# Predict its price
prediction = model.predict(new_house)

print(f"Predicted price: ${prediction[0]:,.2f}")