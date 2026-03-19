# 🔒 Privacy-Preserving Federated Image Classification System

An end-to-end federated learning system that enables multiple clients to collaboratively train an image classification model **without sharing raw data**. The system uses a centralized aggregation server to combine locally trained models via Federated Averaging (FedAvg), with differential privacy (DP-SGD) to protect against inference attacks on model updates.

## ✨ Key Features

- **Federated Learning** — Multiple simulated clients train locally; a central server aggregates updates via FedAvg
- **Differential Privacy** — DP-SGD (via Opacus) adds controlled noise to gradients during training
- **IID & Non-IID Simulation** — Evaluate model performance under realistic heterogeneous data distributions
- **Centralized vs Federated Comparison** — Baseline analysis to quantify the cost of decentralization
- **Privacy–Utility Tradeoff Analysis** — Experiment with varying privacy budgets (ε) and measure accuracy impact
- **Experiment Tracking** — Weights & Biases integration for logging metrics across rounds and settings
- **API-Based Control** — FastAPI backend to programmatically manage training workflows
- **Interactive Dashboard** — Streamlit-based visualization of performance metrics

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch, Torchvision |
| Federated Learning | Flower (flwr) |
| Differential Privacy | Opacus (DP-SGD) |
| Experiment Tracking | Weights & Biases (wandb) |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit, Matplotlib, Seaborn |
| Deployment | Docker |

## 📁 Project Structure

```
FederatedLearning/
├── config/
│   └── default.yaml          # Central configuration (hyperparams, FL, DP settings)
├── src/
│   ├── data/                  # Dataset loading & IID/Non-IID partitioning
│   ├── models/                # CNN model definition
│   ├── training/              # Centralized & federated training loops
│   ├── privacy/               # Opacus/DP integration
│   ├── federated/             # Flower client & server logic
│   └── utils/                 # Helpers — config loader, metrics, logging
├── api/                       # FastAPI application
├── dashboard/                 # Streamlit application
├── experiments/               # Saved experiment results & configs
├── scripts/                   # CLI scripts for running training
└── docker/                    # Dockerfiles & docker-compose
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd FederatedLearning
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; import flwr; import opacus; print('All imports successful!')"
   ```

## 📄 License

This project is for educational and research purposes.
