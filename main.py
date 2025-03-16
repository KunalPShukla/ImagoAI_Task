import numpy as np
import pickle
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import MeanSquaredError
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": " FastAPI is running successfully!"}

# Paths for scaler and model
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "best_neural_network.h5")

# Load scaler and model
try:
    if not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Scaler is not of type StandardScaler")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

    logging.info(" Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    raise RuntimeError(f"Initialization failed: {e}")

# Input schema
class InputData(BaseModel):
    features: list[float]

@app.post("/predict/")
async def predict(input_data: InputData):
    try:
        # Ensure correct input shape
        new_input = np.array(input_data.features).reshape(1, -1)
        logging.info(f" Received input: {new_input}")

        # Standardize input
        new_input_scaled = scaler.transform(new_input)

        # Make prediction
        prediction = model.predict(new_input_scaled)
        logging.info(f" Prediction result: {prediction}")

        return {"predicted_target": float(prediction[0, 0])}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))  # Get port from environment or default to 8000
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
