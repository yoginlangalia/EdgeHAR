```
 _____    _            _   _    _    ____  
| ____|__| | __ _  ___| | | |  / \  |  _ \ 
|  _| / _` |/ _` |/ _ \ |_| | / _ \ | |_) |
| |__| (_| | (_| |  __/  _  |/ ___ \|  _ < 
|_____\__,_|\__, |\___|_| |_/_/   \_\_| \_\
            |___/                           
```

# EdgeHAR — Edge Human Activity Recognition

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **Real-time Human Activity Recognition using CNN-LSTM with Edge Deployment on ESP32 and Streamlit Dashboard.**

Classify 6 human activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) from accelerometer and gyroscope sensor data using a CNN-LSTM deep learning model. Deploy with a real-time Streamlit dashboard and optionally push to ESP32 for edge inference.

---

## 📑 Table of Contents

- [Architecture](#-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)
- [Full Usage Guide](#-full-usage-guide)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [ESP32 Setup](#-esp32-setup)
- [Docker](#-docker)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🏗 Architecture

```
┌──────────────┐    ┌────────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
│   IMU Data   │───▶│ Preprocessing  │───▶│   CNN-LSTM   │───▶│ Classifier │───▶│  Dashboard   │
│  (6-ch, 128) │    │  (Normalize)   │    │   Encoder    │    │  (6 class) │    │  (Streamlit) │
└──────────────┘    └────────────────┘    └──────────────┘    └────────────┘    └──────────────┘
       │                                                                              │
       │                    ┌───────────────────┐                                     │
       └────────────────────│   ESP32 Sensor     │─────────────────────────────────────┘
                            │   (Serial JSON)    │
                            └───────────────────┘
```

---

## ✨ Features

- 🧠 **CNN-LSTM Architecture** — 3 Conv blocks + 2-layer LSTM for robust temporal feature extraction
- 📊 **Real-Time Dashboard** — Streamlit-based monitoring with live predictions, sensor plots, and confusion matrix
- 📱 **ESP32 Integration** — Arduino sketch simulates 6-axis IMU data over serial at 125Hz
- 🚀 **Model Export** — ONNX and TorchScript export for cross-platform deployment
- 🐳 **Docker Ready** — One-command deployment with Docker Compose
- 📈 **Full ML Pipeline** — Data → Train → Evaluate → Export → Deploy, all automated

---

## 📁 Project Structure

```
EdgeHAR/
├── data/
│   └── download_data.py          # Download UCI HAR Dataset
├── notebooks/
│   └── exploration.ipynb         # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── dataset.py                # PyTorch Dataset + DataLoaders
│   ├── model.py                  # CNN-LSTM architecture
│   ├── train.py                  # Training with early stopping
│   ├── evaluate.py               # Metrics + confusion matrix
│   └── export.py                 # ONNX + TorchScript export
├── esp32/
│   └── sensor_simulator.ino      # Arduino IMU simulator
├── dashboard/
│   └── app.py                    # Streamlit real-time dashboard
├── models/                       # Saved checkpoints (gitignored)
├── outputs/                      # Plots and metrics (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/EdgeHAR.git
cd EdgeHAR

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python data/download_data.py      # Download dataset
python src/train.py               # Train model
streamlit run dashboard/app.py    # Launch dashboard
```

---

## 📖 Full Usage Guide

### Step 1: Download Dataset

```bash
python data/download_data.py
```

Downloads the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) (~60MB) and extracts it to `data/`.

### Step 2: Train the Model

```bash
python src/train.py --epochs 50 --lr 0.001 --batch_size 64 --patience 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Maximum training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch_size` | 64 | Batch size |
| `--patience` | 10 | Early stopping patience |

Saves best checkpoint to `models/best_model.pth` and training curves to `outputs/`.

### Step 3: Evaluate

```bash
python src/evaluate.py
```

Generates classification report, confusion matrix heatmap, per-class F1 scores, and inference timing.

### Step 4: Export Model

```bash
python src/export.py
```

Exports to ONNX (`models/har_model.onnx`) and TorchScript (`models/har_model_scripted.pt`), then verifies outputs match.

### Step 5: Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Opens a real-time monitoring dashboard at `http://localhost:8501`. In **Live Simulation** mode, it auto-generates sensor data and runs ONNX inference continuously.

### Docker Deployment

```bash
# Launch dashboard
docker-compose up dashboard

# Run training in container
docker-compose run --rm trainer
```

---

## 🧠 Model Architecture

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Input | (B, 6, 128) | — |
| Conv1d(6→64) + BN + ReLU + MaxPool | (B, 64, 64) | ~1.3K |
| Conv1d(64→128) + BN + ReLU + MaxPool | (B, 128, 32) | ~24.8K |
| Conv1d(128→256) + BN + ReLU | (B, 256, 32) | ~99.1K |
| LSTM(256→128, 2 layers) | (B, 32, 128) | ~395.3K |
| Linear(128→64) + ReLU + Dropout | (B, 64) | ~8.3K |
| Linear(64→6) | (B, 6) | ~0.4K |
| **Total** | | **~529K** |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~92%+ |
| Macro F1 Score | ~0.91+ |
| Inference Time | <1ms/sample (CPU) |
| Model Size (ONNX) | ~2 MB |

> *Results may vary slightly across runs. Train with `seed=42` for reproducibility.*

---

## 📱 ESP32 Setup

### Hardware Required
- ESP32 development board (or any Arduino-compatible board)
- USB cable for serial connection

### Setup Steps

1. **Open** `esp32/sensor_simulator.ino` in Arduino IDE
2. **Select** your board: Tools → Board → ESP32 Dev Module
3. **Upload** the sketch
4. **Open Serial Monitor** at 115200 baud to verify JSON output
5. **Connect** to EdgeHAR Dashboard via the serial port selector

The simulator generates 6-channel IMU data at 125Hz with distinct patterns for each activity, cycling every 5 seconds.

**Serial Output Format:**
```json
{"ax":0.12,"ay":0.98,"az":0.05,"gx":0.01,"gy":-0.02,"gz":0.00,"ts":12345}
```

---

## 🐳 Docker

```bash
# Build and launch dashboard
docker-compose up --build dashboard

# Train model in container
docker-compose --profile training run --rm trainer

# Stop services
docker-compose down
```

---

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | PyTorch 2.0+ |
| Model Export | ONNX, TorchScript |
| Dashboard | Streamlit, Plotly |
| Data Science | NumPy, Pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Edge Device | ESP32 (Arduino) |
| Containerization | Docker, Docker Compose |
| Serial Communication | PySerial |

---

## 📄 License

This project is licensed under the **MIT License** — see below:

```
MIT License

Copyright (c) 2025 EdgeHAR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Made with ❤️ for the edge AI community
</p>
