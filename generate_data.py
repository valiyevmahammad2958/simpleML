import numpy as np
import pandas as pd

# Make random numbers reproducible
np.random.seed(42)

# Generate 100 house sizes between 50 and 200 m²
sizes = np.random.uniform(50, 200, 100)

# Generate random price variation
noise = np.random.normal(0, 20000, 100)

# Generate house prices
prices = 2000 * sizes + noise

# Create a DataFrame
df = pd.DataFrame({
    "size_m2": sizes,
    "price": prices
})

# Save the dataset
df.to_csv("houses.csv", index=False)

# Display first 5 rows
print(df.head())