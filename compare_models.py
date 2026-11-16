
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Load both models
model_v1 = joblib.load("house_price_model.joblib")
model_v2 = joblib.load("house_price_model_v2.joblib")

# Create a shared evaluation dataset
np.random.seed(456)

sizes = np.random.uniform(150, 300, 100)
actual_prices = 2000 * sizes + np.random.normal(0, 20000, 100)

X_test = pd.DataFrame({"size_m2": sizes})

# Predict using both models
pred_v1 = model_v1.predict(X_test)
pred_v2 = model_v2.predict(X_test)

# Evaluate on the SAME dataset
mae_v1 = mean_absolute_error(actual_prices, pred_v1)
mae_v2 = mean_absolute_error(actual_prices, pred_v2)

print(f"Original model MAE: ${mae_v1:.2f}")
print(f"Retrained model MAE: ${mae_v2:.2f}")

if mae_v2 < mae_v1:
    print("Retrained model performs better!")
else:
    print("Original model performs better!")
