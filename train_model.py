import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import joblib
# Load our dataset
df = pd.read_csv("houses.csv")

# Define input and target
X = df[["size_m2"]]
y = df["price"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training houses:", len(X_train))
print("Testing houses:", len(X_test))

from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model using training data
model.fit(X_train, y_train)

# Display learned parameters
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict prices for unseen houses
y_pred = model.predict(X_test)

# Compare actual and predicted prices
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print(results.head())

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

import os
import matplotlib

if os.getenv("CI") == "true":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plot actual test data
plt.scatter(X_test["size_m2"], y_test, label="Actual prices")

# Plot model predictions
plt.plot(
    X_test["size_m2"].sort_values(),
    model.predict(X_test.sort_values("size_m2")),
    color="red",
    label="Regression line"
)

plt.xlabel("House size (m²)")
plt.ylabel("House price ($)")
plt.title("House Price Prediction")
plt.legend()
plt.show()


import mlflow

def log_experiment(mae, rmse, r2):
    mlflow.set_experiment("House Price Prediction")

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

    print("Experiment logged successfully!")


log_experiment(mae, rmse, r2)


# Save the trained model
joblib.dump(model, "house_price_model.joblib")

print("Model saved successfully!")