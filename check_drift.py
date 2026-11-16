
import pandas as pd
import numpy as np

# Load original training data
training_data = pd.read_csv("houses.csv")

# Simulate new incoming data
np.random.seed(123)
new_sizes = np.random.uniform(150, 300, 100)

# Compare average house sizes
training_mean = training_data["size_m2"].mean()
new_mean = new_sizes.mean()

change_percent = (
    abs(new_mean - training_mean) / training_mean
) * 100

print(f"Training mean: {training_mean:.2f} m2")
print(f"New data mean: {new_mean:.2f} m2")
print(f"Change: {change_percent:.2f}%")

# Simple illustrative drift alert
if change_percent > 20:
    print("WARNING: Potential data drift detected!")
else:
    print("No significant mean shift detected.")
