# main.py - Your Backend API Application

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

# --- 1. DEFINE THE NEURAL NETWORK ARCHITECTURE ---
# This MUST be the *exact* same structure as the one you saved.
# This model expects 19 input features.
class NeuralNetwork_drop(nn.Module):
    def __init__(self, input_size=19): # The input size is fixed to 19
        super(NeuralNetwork_drop, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

# --- 2. INITIALIZE THE MODEL AND LOAD THE WEIGHTS ---
model = NeuralNetwork_drop()
# Load the trained weights from the file you downloaded.
# Ensure 'adhd_predictor_model_v1.pth' is in the same directory as main.py
model.load_state_dict(torch.load('adhd_predictor_model_v1.pth', map_location=torch.device('cpu')))
# Set the model to evaluation mode (this disables dropout, etc., for prediction).
model.eval()

# --- 3. DEFINE THE EXPECTED INPUT DATA STRUCTURE ---
# Pydantic validates the data. We expect a dictionary where keys are region codes
# (like "Fp1", "Cz") and values are numbers (float).
class EEGData(BaseModel):
    data: Dict[str, float]

# --- 4. CREATE THE FASTAPI APP ---
app = FastAPI()

# --- 5. CONFIGURE CORS (IMPORTANT SECURITY STEP) ---
# This allows your website (on a different domain) to make requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any website to connect. For production, you might restrict this.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods.
    allow_headers=["*"],  # Allows all HTTP headers.
)

# --- 6. DEFINE THE PREDICTION "ENDPOINT" (URL) ---
# This function will run when the website sends a POST request to '/predict'.
@app.post("/predict")
def predict_adhd(eeg_data: EEGData):
    # Define the exact order of regions. This MUST match the training data order.
    # CORRECTED: This order now matches the training data format you provided.
    regions_order = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
    ]

    # Create the feature vector in the correct order from the incoming dictionary.
    try:
        feature_vector = [eeg_data.data[region] for region in regions_order]
    except KeyError as e:
        # This is a good practice error check in case the frontend sends incomplete data.
        return {"error": f"Missing data for EEG region: {e}"}

    # Convert the list to a PyTorch tensor.
    input_tensor = torch.tensor(feature_vector, dtype=torch.float32)

    # The model expects a "batch" of data, so we add a dimension.
    # Shape changes from [19] to [1, 19].
    input_tensor = input_tensor.unsqueeze(0)

    # Make the prediction.
    with torch.no_grad(): # No need to calculate gradients during inference.
        output = model(input_tensor)

    # The output is a probability. We decide the class based on a 0.5 threshold.
    probability = output.item()
    prediction = "ADHD" if probability >= 0.5 else "Control"

    # Send the result back to the website as a JSON object.
    return {
        "prediction": prediction,
        "probability": probability
    }

# A simple test endpoint to check if the server is alive.
@app.get("/")
def read_root():
    return {"message": "EEGenius Predictor API is operational."}