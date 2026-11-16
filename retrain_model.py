
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Load original data
old_data = pd.read_csv("houses.csv")

# 2. Simulate new houses with known prices
np.random.seed(123)

new_sizes = np.random.uniform(150, 300, 100)
new_prices = 2000 * new_sizes + np.random.normal(0, 20000, 100)

new_data = pd.DataFrame({
    "size_m2": new_sizes,
    "price": new_prices
})

# 3. Combine old and new data
combined_data = pd.concat(
    [old_data, new_data],
    ignore_index=True
)

# 4. Prepare features and target
X = combined_data[["size_m2"]]
y = combined_data["price"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train a new model
new_model = LinearRegression()
new_model.fit(X_train, y_train)

# 7. Evaluate
predictions = new_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Retrained model MAE: ${mae:.2f}")

# 8. Save as a separate model
joblib.dump(new_model, "house_price_model_v2.joblib")

print("Retrained model saved successfully!")
