# 🧠 EEGenius — AI-Powered Neurological Diagnostics

**EEGenius** is a next-generation neurological diagnostics platform that leverages **deep learning** and **EEG signal processing** to detect early biomarkers of Parkinson's Disease and ADHD. Built with a cyberpunk aesthetic, the platform provides an immersive interface for exploring the intersection of AI and neuroscience.

> ⚠️ **Disclaimer:** This tool is for **educational and demonstration purposes only**. It has NOT been evaluated by the FDA or any medical regulatory body. Always consult a qualified healthcare provider.

---

## 🌐 Live Site

**[eegenius.co](https://eegenius.co/)**

---

## 📂 Project Structure

```
EEGENIUS/
├── index.html                  # Main landing page (security gate + navigation)
├── sitemap.xml                 # SEO sitemap
├── parkinsons-ai/              # Parkinson's Disease Detection Module
│   ├── index.html              # Frontend UI for Parkinson's prediction
│   ├── main.py                 # FastAPI backend + heuristic prediction engine
│   ├── parkinsons_model.pth    # Trained PyTorch neural network weights
│   ├── scaling_params.json     # Feature normalization parameters
│   ├── requirements.txt        # Python dependencies
│   └── Start_AI.bat            # Windows launcher script
├── adhd-ai/                    # ADHD Detection Module
│   ├── index.html              # Frontend UI for ADHD prediction
│   ├── main.py                 # FastAPI backend for ADHD classification
│   ├── adhd_predictor_model_v1.pth  # Trained PyTorch model weights
│   ├── requirements.txt        # Python dependencies
│   └── Start_AI.bat            # Windows launcher script
├── article/                    # SEO article: IGCSE revision resources
│   └── index.html
├── disclaimer/                 # Legal disclaimer page
│   └── index.html
├── images/                     # Image assets
├── music/                      # SFX audio files
└── favicon1/                   # Favicon & PWA manifest
```

---

## 🚀 Features

### 🔴 Parkinson's Disease Detection
- **27 input biomarkers** including Pitch Period Entropy (PPE), EEG composite scores, Recurrence Period Density Entropy (R_PDE), and 10 Ed coefficients
- **Hybrid prediction engine** combining a PyTorch neural network with a heuristic scoring system
- **Real-time probability** output with binary classification (Parkinson's / Control)

### 🔵 ADHD Detection
- **19-channel EEG analysis** across standard 10-20 system regions (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz)
- **Deep neural network** (3-layer MLP with dropout) for binary classification
- **REST API** powered by FastAPI

### 🎨 Immersive UI
- **Cyberpunk / glitch-art** aesthetic with custom cursor, scanlines, and neon styling
- **Three.js** 3D animated backgrounds with floating geometric shards
- **GSAP** animation sequences for page transitions
- **Security gate** keypad interface on the landing page
- Fully responsive design

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3 (Tailwind CSS), JavaScript |
| **3D Graphics** | Three.js (r128) |
| **Animations** | GSAP 3.12 |
| **Backend** | Python 3.x, FastAPI, Uvicorn |
| **Machine Learning** | PyTorch, NumPy |
| **Fonts** | Orbitron, Rajdhani, Share Tech Mono (Google Fonts) |

---

## ⚡ Getting Started

### Prerequisites

- **Python 3.8+** installed
- **pip** package manager
- A modern web browser (Chrome/Firefox/Edge recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/eegenius.git
   cd eegenius
   ```

2. **Install Python dependencies for each module:**

   ```bash
   # Parkinson's AI
   cd parkinsons-ai
   pip install -r requirements.txt

   # ADHD AI
   cd ../adhd-ai
   pip install -r requirements.txt
   ```

### Running Locally

#### Option 1: Using the batch files (Windows)

Double-click the launcher scripts:
- `parkinsons-ai/Start_AI.bat` — Starts the Parkinson's prediction server
- `adhd-ai/Start_AI.bat` — Starts the ADHD prediction server

#### Option 2: Manual terminal launch

```bash
# Terminal 1 — Parkinson's AI (port 8000)
cd parkinsons-ai
uvicorn main:app --reload --port 8000

# Terminal 2 — ADHD AI (port 8001)
cd adhd-ai
uvicorn main:app --reload --port 8001
```

#### Option 3: Serve the frontend

Open `index.html` directly in your browser, or use any static file server:

```bash
# Using Python's built-in server
python -m http.server 3000
# Then visit http://localhost:3000
```

---

## 🔌 API Reference

### Parkinson's Prediction

```http
POST /predict
Content-Type: application/json

{
  "gender": 0,
  "PPE": 0.45,
  "EEG": 0.72,
  "R_PDE": 0.55,
  "numPulses": 320,
  "numPeriodsPulses": 310,
  "meanPeriodPulses": 0.006,
  "stdDevPeriodPulses": 0.0004,
  "f1": 640, "f2": 1200, "f3": 2500, "f4": 3500,
  "b1": 200, "b2": 290, "b3": 530, "b4": 500,
  "Ea": 99.9,
  "Ed_1_coef": 0.002, "Ed_2_coef": 0.002,
  "Ed_3_coef": 0.002, "Ed_4_coef": 0.004,
  "Ed_5_coef": 0.003, "Ed_6_coef": 0.003,
  "Ed_7_coef": 0.003, "Ed_8_coef": 0.013,
  "Ed_9_coef": 0.003, "Ed_10_coef": 0.031
}
```

**Response:**
```json
{
  "prediction": "Parkinsons",
  "probability": 0.87
}
```

### ADHD Prediction

```http
POST /predict
Content-Type: application/json

{
  "data": {
    "Fp1": 0.5, "Fp2": 0.6, "F3": 0.4, "F4": 0.5,
    "C3": 0.3, "C4": 0.4, "P3": 0.2, "P4": 0.3,
    "O1": 0.1, "O2": 0.2, "F7": 0.6, "F8": 0.7,
    "T7": 0.5, "T8": 0.6, "P7": 0.3, "P8": 0.4,
    "Fz": 0.4, "Cz": 0.3, "Pz": 0.2
  }
}
```

**Response:**
```json
{
  "prediction": "ADHD",
  "probability": 0.76
}
```

---

## 🧪 How It Works

The Parkinson's detection module uses a **hybrid approach**:

1. **AI Model (10% weight):** A 3-layer neural network (27→64→32→1) with dropout regularization, trained on clinical biomarker data
2. **Heuristic Engine (90% weight):** A rule-based scoring system that weights critical biomarkers (PPE, R_PDE, StdDev, EEG) against a calibrated bias threshold
3. **Combined Output:** The final probability is a weighted blend, classified as Parkinson's if ≥ 0.5

The ADHD module uses a more traditional deep learning approach with a 19-input neural network (19→64→32→1) operating directly on regional EEG measurements.

---

## 👤 Authors

- **Adolf Lobowicz** — Co-Founder & Lead Architect — [LinkedIn](https://www.linkedin.com/in/adolf-lobowicz-497a6b372)
- **Akriti Srivastava** — Co-Founder — [LinkedIn](https://www.linkedin.com/in/akriti-srivastava-575724321)

---

## 📄 License

This project is proprietary. All rights reserved. See the [Disclaimer](https://eegenius.co/disclaimer/) for usage terms.

---

## ⚕️ Medical Disclaimer

EEGenius is a **demonstration and educational tool**. The predictions provided are **NOT medical diagnoses** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. This system has not been evaluated or cleared by the FDA or any medical regulatory body. Always consult a qualified healthcare provider or neurologist for any medical concerns.

---

<p align="center">
  <sub>Built with ⚡ at the intersection of neuroscience and artificial intelligence</sub>
</p>