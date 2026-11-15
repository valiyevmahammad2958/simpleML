# 1. Start with an image containing Python
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependency list
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy our API and trained model
COPY app.py .
COPY house_price_model.joblib .

# 6. Document the port used by our API
EXPOSE 8000

# 7. Start FastAPI when the container runs
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]