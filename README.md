# Privacy-Preserving Federated Learning for Medical Image Classification

A full-stack federated learning system where multiple simulated hospitals collaboratively train a **colorectal cancer tissue classifier** on PathMNIST — **without sharing patient data**. The system uses Federated Averaging (FedAvg) for model aggregation and Differential Privacy (DP-SGD via Opacus) to mathematically guarantee that no individual patient's data can be reverse-engineered from model updates.

## Key Results

| Configuration | Test Accuracy | Privacy Level |
|---|---|---|
| Centralized (all data in one place) | **85.3%** | None |
| Federated IID (no DP) | **81.9%** | Data stays local |
| Federated IID + DP (noise=0.5) | **62.3%** | Mathematical guarantee (epsilon=3.96) |
| Federated Non-IID (2 cls/client) | **17.3%** | Data stays local |
| Federated Non-IID (4 cls/client) | **35.9%** | Data stays local |

**Key Finding:** Federation costs only 3.4% accuracy (85.3% to 81.9%), while adding differential privacy costs an additional 19.6% — quantifying the exact privacy-utility tradeoff that organizations face when protecting sensitive medical data.

![Accuracy Comparison](experiments/comparison_chart.png)

![Training Curves](experiments/training_curves.png)

![Privacy-Utility Tradeoff](experiments/privacy_tradeoff.png)

## How It Works

```
1. SERVER creates a random model and sends weights to selected clients

2. Each CLIENT (hospital):
   - Receives global weights
   - Trains on LOCAL patient images (data never leaves)
   - Optionally applies DP-SGD (clip gradients + add noise)
   - Sends ONLY updated weights back (NOT images)

3. SERVER averages all client weights (FedAvg) → improved global model

4. Repeat for N rounds → model improves each round

5. Only MODEL WEIGHTS travel between client and server — never patient data
```

## Features

- **Federated Learning** — 5 simulated hospital clients train locally; central server aggregates via FedAvg
- **Differential Privacy** — DP-SGD (Opacus) clips per-sample gradients and adds Gaussian noise
- **Privacy Budget Tracking** — Epsilon monitoring to track cumulative privacy expenditure
- **IID & Non-IID Simulation** — Test with balanced vs heterogeneous data distributions
- **Automated Experiments** — Scripts to run the full experiment matrix and generate comparison plots
- **REST API** — FastAPI backend to programmatically control training
- **Interactive Dashboard** — Streamlit UI with training controls, metric visualization, and epsilon tracker
- **W&B Integration** — Optional Weights & Biases logging for cloud experiment tracking

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Deep Learning | PyTorch, Torchvision | Model training and evaluation |
| Medical Dataset | MedMNIST (PathMNIST) | Colorectal cancer tissue classification (9 classes) |
| Differential Privacy | Opacus | DP-SGD with per-sample gradient clipping and noise |
| Experiment Tracking | Weights & Biases | Cloud-based metric logging and dashboards |
| Backend API | FastAPI + Uvicorn | REST endpoints for experiment orchestration |
| Dashboard | Streamlit, Matplotlib | Interactive visualization and training controls |

## Project Structure

```
FederatedLearning/
├── config/
│   └── default.yaml              # Central configuration (model, FL, DP, data settings)
├── src/
│   ├── datasets_partition/
│   │   ├── dataset.py            # PathMNIST loader with normalization
│   │   └── partition.py          # IID & Non-IID data partitioning
│   ├── models/
│   │   └── cnn.py                # SimpleCNN (GroupNorm, Opacus-compatible)
│   ├── training/
│   │   └── train.py              # Centralized training baseline
│   ├── federated/
│   │   ├── client.py             # Federated client (local training + DP)
│   │   └── server.py             # FedAvg aggregation + simulation loop
│   ├── privacy/
│   │   └── dp_utils.py           # Opacus DP-SGD wrapper utilities
│   └── utils/
│       ├── config_loader.py      # YAML configuration loader
│       └── logger.py             # W&B experiment logging helpers
├── api/
│   └── main.py                   # FastAPI endpoints
├── dashboard/
│   └── app.py                    # Streamlit visualization dashboard
├── experiments/                   # Experiment results (JSON) and plots (PNG)
├── scripts/
│   ├── run_experiments.py        # Automated experiment runner (5 configs)
│   ├── run_noniid_improved.py    # Improved Non-IID experiments
│   └── plot_results.py           # Generate comparison charts
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.9+
- Conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/NehaJaiswal15/FederatedLearning.git
cd FederatedLearning

# Create conda environment
conda create -n fedLearn python=3.10 -y
conda activate fedLearn

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Test dataset loading (downloads PathMNIST ~200MB first time)
python -m src.datasets_partition.dataset

# 2. Run centralized baseline
python -m src.training.train

# 3. Run federated training (no DP)
python -m src.federated.server

# 4. Run all experiments and generate plots
python scripts/run_experiments.py
python scripts/plot_results.py
```

### API & Dashboard

```bash
# Terminal 1: Start the API
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start the Dashboard
streamlit run dashboard/app.py
```

## Dataset: PathMNIST

[PathMNIST](https://medmnist.com/) contains **89,996 training** and **7,180 test** images of colorectal cancer tissue (28x28 RGB, resized to 32x32). The 9 classes represent different tissue types found in colorectal cancer biopsies:

| Class | Tissue Type |
|---|---|
| 0 | Adipose |
| 1 | Background |
| 2 | Debris |
| 3 | Lymphocytes |
| 4 | Mucus |
| 5 | Smooth Muscle |
| 6 | Normal Colon Mucosa |
| 7 | Cancer-Associated Stroma |
| 8 | Colorectal Adenocarcinoma Epithelium |

**Why this dataset:** Medical imaging is the primary real-world application of federated learning — hospitals must comply with HIPAA/GDPR regulations and cannot centralize patient data.

## Model Architecture

**SimpleCNN** (~1.1M parameters):
- 2 convolutional blocks: Conv2d → GroupNorm → ReLU → Conv2d → GroupNorm → ReLU → MaxPool2d
- Fully connected: Linear(4096→256) → ReLU → Dropout(0.5) → Linear(256→9)
- **GroupNorm** instead of BatchNorm (required for Opacus compatibility)
- **Non-inplace ReLU** (required for per-sample gradient computation)

## Configuration

All hyperparameters are centralized in `config/default.yaml`:

```yaml
model:
  num_classes: 9

training:
  epochs: 1
  batch_size: 32
  learning_rate: 0.01

federated:
  num_clients: 5
  num_rounds: 10
  fraction_fit: 0.5

privacy:
  enable_dp: false        # Toggle DP-SGD
  target_epsilon: 10.0    # Privacy budget
  noise_multiplier: 0.5   # Noise level
  max_grad_norm: 1.0      # Gradient clipping bound

data:
  dataset_name: PathMNIST
  iid: true               # IID or Non-IID partitioning
```

## Key Design Decisions

| Decision | Reason |
|---|---|
| GroupNorm over BatchNorm | Opacus cannot track per-sample gradients through BatchNorm's running statistics |
| Non-inplace ReLU | Opacus needs original activations preserved for backpropagation |
| Fresh model per DP round | Prevents Opacus "hooks attached twice" error when reusing models |
| Manual FedAvg (no Flower runtime) | Avoids Ray dependency issues on Windows while maintaining identical aggregation |
| PathMNIST over CIFAR-10 | Real-world medical use case with strong privacy narrative |
| Resize 28→32 | Keeps CNN architecture unchanged from CIFAR-10 baseline |

## License

This project is for educational and research purposes.

## References

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg, 2017)
- Abadi et al., "Deep Learning with Differential Privacy" (DP-SGD, 2016)
- Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification" (2023)
- Opacus: https://opacus.ai/
