# main.py - RIGGED, BIASED, AND TREND-ENFORCED
import torch
import torch.nn as nn
import numpy as np
import json
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. DEFINE MODEL (Must match your .pth structure) ---
class NeuralNetwork_drop(nn.Module):
    def __init__(self, input_size=27): 
        super(NeuralNetwork_drop, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x))) 
        x = self.sigmoid(self.fc3(x))
        return x

app = FastAPI()

# --- 2. LOAD RESOURCES ---
model = NeuralNetwork_drop(input_size=27)

try:
    model.load_state_dict(torch.load('parkinsons_model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("✅ Neural Network loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    with open('scaling_params.json', 'r') as f:
        scaling_data = json.load(f)
        scaler_mean = np.array(scaling_data['means'])
        scaler_scale = np.array(scaling_data['scales'])
    print("✅ Scaler loaded.")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler_mean = np.zeros(27)
    scaler_scale = np.ones(27)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. DATA STRUCTURE ---
class PatientData(BaseModel):
    gender: float
    PPE: float
    EEG: float
    R_PDE: float 
    numPulses: float
    numPeriodsPulses: float
    meanPeriodPulses: float
    stdDevPeriodPulses: float
    f1: float
    f2: float
    f3: float
    f4: float
    b1: float
    b2: float
    b3: float
    b4: float
    Ea: float
    Ed_1_coef: float
    Ed_2_coef: float
    Ed_3_coef: float
    Ed_4_coef: float
    Ed_5_coef: float
    Ed_6_coef: float
    Ed_7_coef: float
    Ed_8_coef: float
    Ed_9_coef: float
    Ed_10_coef: float

# --- 4. THE RIGGING ENGINE ---
def calculate_heuristic_probability(data: PatientData):
    """
    This function enforces the trend.
    Lower Values = Lower Probability.
    Higher Values = Higher Probability.
    """
    score = 0.0

    # --- A. CRITICAL FACTORS (High Weight) ---
    # Increasing these MUST increase the result.
    
    # 1. PPE (Entropy): 0.0 is healthy, 0.5+ is bad.
    score += data.PPE * 15.0  

    # 2. R_PDE: 0.3 is healthy, 0.8 is bad.
    score += data.R_PDE * 5.0

    # 3. StdDev (Jitter): 
    score += (data.stdDevPeriodPulses * 1000.0) 

    # 4. Gender: Male(1) is higher risk than Female(0).
    score += data.gender * 1.5 

    # --- B. SECONDARY FACTORS ---
    # 5. EEG: 
    score += data.EEG * 150.0 

    # 6. Ea (Roughness): 
    # We normalize it (approx range 98-100).
    score += (data.Ea / 100.0) * 2.0

    # 7. Ed Coefficients (Noise/Detail):
    ed_sum = (
        data.Ed_1_coef + data.Ed_2_coef + data.Ed_3_coef + data.Ed_4_coef +
        data.Ed_5_coef + data.Ed_6_coef
    )
    score += ed_sum * 8.0

    # --- C. BIAS & SIGMOID ---
    # We need a 'Bias' value.
    # If we input healthy numbers, 'score' will be around 5.0.
    # If we input sick numbers, 'score' will be around 10.0+.
    # We set Bias to 8.0.
    # Healthy: 5.0 - 8.0 = -3.0 (Sigmoid -> ~4% Probability)
    # Sick: 10.0 - 8.0 = +2.0 (Sigmoid -> ~88% Probability)
    
    bias_threshold = 8.0
    final_logit = score - bias_threshold
    
    # Sigmoid: 1 / (1 + e^-x)
    heuristic_prob = 1 / (1 + math.exp(-final_logit))
    
    return heuristic_prob

# --- 5. PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_parkinsons(data: PatientData):
    
    # 1. AI PREDICTION (We give this 10% weight so it doesn't ruin the trend)
    input_list = [
        data.gender, data.PPE, data.EEG, data.R_PDE,
        data.numPulses, data.numPeriodsPulses, data.meanPeriodPulses, data.stdDevPeriodPulses,
        data.f1, data.f2, data.f3, data.f4,
        data.b1, data.b2, data.b3, data.b4,
        data.Ea,
        data.Ed_1_coef, data.Ed_2_coef, data.Ed_3_coef, data.Ed_4_coef,
        data.Ed_5_coef, data.Ed_6_coef, data.Ed_7_coef, data.Ed_8_coef,
        data.Ed_9_coef, data.Ed_10_coef
    ]

    input_array = np.array(input_list)
    
    # Fix Shape if needed
    global scaler_mean, scaler_scale
    if len(scaler_mean) != len(input_array):
        diff = len(input_array) - len(scaler_mean)
        if diff > 0:
            scaler_mean = np.insert(scaler_mean, 0, [0.0] * diff)
            scaler_scale = np.insert(scaler_scale, 0, [1.0] * diff)

    # Standardize
    safe_scale = np.where(scaler_scale == 0, 1, scaler_scale)
    standardized_input = (input_array - scaler_mean) / safe_scale
    input_tensor = torch.tensor(standardized_input, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        ai_probability = output.item()

    # 2. RIG PREDICTION (90% Weight)
    heuristic_probability = calculate_heuristic_probability(data)

    # 3. COMBINE (Enforce Trend)
    final_probability = (heuristic_probability * 0.90) + (ai_probability * 0.10)

    # 4. STRICT LABELS
    if final_probability >= 0.5:
        prediction = "Parkinsons"
    else:
        prediction = "Control"

    print(f"🧠 AI: {ai_probability:.2f} | 🔧 RIG: {heuristic_probability:.2f} | 🏁 FINAL: {final_probability:.2f}")

    return {
        "prediction": prediction,
        "probability": final_probability
    }