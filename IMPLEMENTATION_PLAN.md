# 🔒 Privacy-Preserving Federated Image Classification — Implementation Plan

> **Purpose of this document:** This is a complete project reference. If you're an AI assistant helping a contributor, read this fully before writing any code. If you're a developer joining the project, this tells you everything you need to know.

---

## 1. Project Overview

### What We're Building
An end-to-end **federated learning system** where multiple simulated clients collaboratively train a CNN image classifier on CIFAR-10 **without sharing raw data**. A central server aggregates model updates via **Federated Averaging (FedAvg)**. **Differential privacy (DP-SGD)** is applied to local training to protect against model update inference attacks.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                   │
│         (visualize metrics, trigger experiments)        │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│                     FASTAPI SERVER                      │
│    (API layer: start training, fetch results, config)   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              FLOWER FEDERATED SERVER                    │
│         (orchestrates rounds, FedAvg aggregation)       │
└──────┬───────────┬───────────┬──────────────────────────┘
       │           │           │
┌──────▼──┐  ┌─────▼───┐  ┌───▼───────┐
│ Client 1│  │ Client 2│  │ Client N  │   ← Simulated FL Clients
│ PyTorch │  │ PyTorch │  │ PyTorch   │
│ + Opacus│  │ + Opacus│  │ + Opacus  │   ← DP-SGD for privacy
└─────────┘  └─────────┘  └───────────┘
       │           │           │
   [Local Data] [Local Data] [Local Data]  ← Data never leaves client
```

### Tech Stack

| Component | Technology | Why |
|---|---|---|
| Deep Learning | PyTorch + Torchvision | Industry standard, Opacus compatible |
| Federated Learning | Flower (flwr) | Most mature FL framework |
| Differential Privacy | Opacus (DP-SGD) | Meta's official DP library for PyTorch |
| Experiment Tracking | Weights & Biases (wandb) | Production-grade metric logging |
| Backend API | FastAPI + Uvicorn | Modern async Python API |
| Dashboard | Streamlit + Matplotlib + Seaborn | Rapid interactive visualization |
| Deployment | Docker + docker-compose | Containerized multi-service deployment |

---

## 2. Project Structure

```
FederatedLearning/
├── config/
│   └── default.yaml              # Central config (hyperparams, FL, DP settings)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # ✅ DONE — CIFAR-10 loader with normalization
│   │   └── partition.py          # ✅ DONE — IID & Non-IID data partitioning
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py                # ✅ DONE — SimpleCNN (GroupNorm, Opacus-ready)
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py              # 📌 TODO (Phase 4) — Centralized training loop
│   ├── privacy/
│   │   ├── __init__.py
│   │   └── dp_utils.py           # 📌 TODO (Phase 6) — Opacus DP wrapper
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── client.py             # 📌 TODO (Phase 5) — Flower FL client
│   │   └── server.py             # 📌 TODO (Phase 5) — Flower FL server
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py      # ✅ DONE — YAML config loader
├── api/
│   ├── __init__.py
│   └── main.py                   # 📌 TODO (Phase 8) — FastAPI app
├── dashboard/
│   ├── __init__.py
│   └── app.py                    # 📌 TODO (Phase 9) — Streamlit dashboard
├── experiments/                   # Saved experiment results
├── scripts/                       # CLI scripts to run training
├── docker/                        # Dockerfiles and compose
├── requirements.txt               # ✅ DONE
├── README.md                      # ✅ DONE
├── .gitignore                     # ✅ DONE
└── IMPLEMENTATION_PLAN.md         # ← This file
```

---

## 3. What's Already Built (Phases 1–3) ✅

### `src/utils/config_loader.py`
- **Function:** `load_config(config_path="config/default.yaml") → dict`
- Reads YAML config, returns nested Python dictionary
- Has error handling for missing files

### `config/default.yaml`
Central configuration with 5 sections:
```yaml
model:       { name: "SimpleCNN", num_classes: 10 }
training:    { epochs: 1, batch_size: 32, learning_rate: 0.01, optimizer: "SGD" }
federated:   { num_clients: 5, num_rounds: 10, fraction_fit: 0.5 }
privacy:     { enable_dp: false, target_epsilon: 10.0, max_grad_norm: 1.0, noise_multiplier: 0.5 }
data:        { dataset_name: "CIFAR10", data_dir: "./data", iid: true }
```

### `src/data/dataset.py`
- **Function:** `get_cifar10(data_dir="./data") → (train_dataset, test_dataset)`
- Downloads CIFAR-10 (50k train + 10k test, 32×32 RGB)
- Applies `ToTensor()` + `Normalize(mean, std)` using CIFAR-10 channel statistics
- **Constant:** `CIFAR10_CLASSES` — list of 10 class names

### `src/data/partition.py`
- **Function:** `partition_data(dataset, num_clients, iid=True, classes_per_client=2) → list[Subset]`
- **IID mode:** Random shuffle → split into equal chunks → each client gets all 10 classes
- **Non-IID mode:** Group by class → create shards → each client gets only `classes_per_client` classes
- Returns `torch.utils.data.Subset` objects (views into the original dataset)

### `src/models/cnn.py`
- **Class:** `SimpleCNN(num_classes=10)`
- Architecture: 2 conv blocks (Conv2d → GroupNorm → ReLU → Conv2d → GroupNorm → ReLU → MaxPool2d) → Flatten → Linear(4096→256) → ReLU → Dropout(0.5) → Linear(256→10)
- **GroupNorm** instead of BatchNorm (required for Opacus/DP compatibility)
- **No inplace=True** on ReLU (required for Opacus per-sample gradients)
- ~1.1M trainable parameters
- Output: raw logits (no softmax — `CrossEntropyLoss` handles it)

---

## 4. Phase Plan — What's Left to Build

### Phase 4: Centralized Training Baseline 👈 TEAMMATE'S FIRST TASK
**Folder:** `src/training/train.py`
**Branch:** `feature/centralized-training`

Build a standard (non-federated) training loop as a performance baseline.

**What the file should contain:**
1. A `train_one_epoch(model, dataloader, optimizer, criterion, device)` function:
   - Loop over batches → forward pass → compute loss → backward pass → optimizer step
   - Return average loss and accuracy for the epoch
2. An `evaluate(model, dataloader, criterion, device)` function:
   - Loop over batches with `torch.no_grad()` → compute loss and accuracy
   - Return average loss and accuracy
3. A `run_centralized_training(config)` function:
   - Load config using `load_config()`
   - Load full CIFAR-10 using `get_cifar10()`
   - Create `DataLoader` objects with `batch_size` from config
   - Initialize `SimpleCNN`, `CrossEntropyLoss`, and `SGD` optimizer
   - Train for `config['training']['epochs']` epochs (use 5-10 for baseline)
   - Print loss/accuracy per epoch
4. An `if __name__ == "__main__"` block to run it

**Key imports from existing code:**
```python
from src.utils.config_loader import load_config
from src.data.dataset import get_cifar10
from src.models.cnn import SimpleCNN
```

**Expected centralized baseline accuracy:** ~70-75% after 10 epochs on CPU

---

### Phase 5: Flower Federated Client & Server
**Folder:** `src/federated/client.py`, `src/federated/server.py`
**Owner:** Person A (repo owner)

**client.py:**
- Extend `flwr.client.NumPyClient`
- Implement `get_parameters()`, `set_parameters()`, `fit()`, `evaluate()`
- `fit()` trains locally on the client's partition for `config['training']['epochs']` epochs
- `evaluate()` tests on local data and returns loss/accuracy

**server.py:**
- Configure Flower server with `FedAvg` strategy
- Set `fraction_fit` and `num_rounds` from config
- Start server with `flwr.server.start_server()`

---

### Phase 6: Differential Privacy (Opacus)
**Folder:** `src/privacy/dp_utils.py`
**Owner:** Person A

- Wrap model + optimizer + dataloader with Opacus `PrivacyEngine`
- Use `max_grad_norm` and `noise_multiplier` from config
- Track privacy budget (epsilon) spent per round
- Conditionally enabled via `config['privacy']['enable_dp']`

---

### Phase 7: W&B Experiment Tracking
**Folder:** `src/utils/` (add logging helpers)
**Owner:** Person B (teammate)

- Initialize W&B run with config dict
- Log per-epoch metrics: loss, accuracy
- Log per-round FL metrics: global accuracy, per-client accuracy
- Log privacy budget (epsilon) when DP is enabled

---

### Phase 8: FastAPI Backend
**Folder:** `api/main.py`
**Owner:** Person B

Endpoints to build:
- `POST /train/centralized` — start centralized training
- `POST /train/federated` — start federated training
- `GET /results` — fetch latest experiment results
- `GET /config` — return current config
- `POST /config` — update config parameters

---

### Phase 9: Streamlit Dashboard
**Folder:** `dashboard/app.py`
**Owner:** Person B

Visualizations:
- Training loss/accuracy curves (per round)
- Centralized vs Federated comparison charts
- Per-client data distribution (IID vs Non-IID)
- Privacy budget (ε) tracker
- Controls to adjust config and trigger training via API

---

### Phase 10: Privacy–Utility Tradeoff Experiments
**Owner:** Person A

Run experiments with varying:
- `target_epsilon`: [0.1, 1.0, 5.0, 10.0, 50.0]
- `noise_multiplier`: [0.1, 0.5, 1.0, 2.0]
- IID vs Non-IID
- Number of clients: [3, 5, 10]

Generate comparison plots and analysis.

---

### Phase 11: Docker Deployment
**Owner:** Both (together)

- `Dockerfile` for the combined system
- `docker-compose.yml` for multi-container setup (server, clients, API, dashboard)

---

## 5. Work Division — Who Owns What

### Person A (Repo Owner)
| Phase | Folder | Status |
|---|---|---|
| Phase 1-3 | Setup, Data, Model | ✅ Done |
| Phase 5 | `src/federated/` | 📌 Next |
| Phase 6 | `src/privacy/` | 📌 |
| Phase 10 | `experiments/`, `scripts/` | 📌 |

### Person B (Teammate)
| Phase | Folder | Status |
|---|---|---|
| Phase 4 | `src/training/` | 📌 Next |
| Phase 7 | `src/utils/` (logging) | 📌 |
| Phase 8 | `api/` | 📌 |
| Phase 9 | `dashboard/` | 📌 |

### Together
| Phase | Folder |
|---|---|
| Phase 11 | `docker/` |

**Rule:** Each person works in **different folders** — zero merge conflicts.

---

## 6. Git Workflow

- **Person A** (repo owner): pushes directly to `main`
- **Person B** (teammate): creates feature branches → opens PR → Person A reviews → merge

```bash
# Teammate's workflow
git clone https://github.com/NehaJaiswal15/FederatedLearning.git
cd FederatedLearning
git checkout -b feature/centralized-training

# ... do Phase 4 work in src/training/ ...

git add .
git commit -m "Phase 4: Centralized training baseline"
git push origin feature/centralized-training
# Then open a Pull Request on GitHub
```

---

## 7. How to Run Existing Code

```bash
# Setup
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Test config loader
python -m src.utils.config_loader

# Test dataset (downloads ~170MB first time)
python -m src.data.dataset

# Test partitioning
python -m src.data.partition

# Test model
python -m src.models.cnn
```

---

## 8. Key Design Decisions & Constraints

| Decision | Reason |
|---|---|
| **GroupNorm** over BatchNorm | Opacus cannot work with BatchNorm (tracks running stats across batch, leaks privacy) |
| **No inplace ReLU** | Opacus needs original activations for per-sample gradient computation |
| **CIFAR-10** as dataset | Industry-standard FL benchmark, auto-downloads, trains fast on CPU |
| **CrossEntropyLoss** | Model outputs raw logits (no softmax layer) — CE applies softmax internally |
| **YAML config** | Keeps hyperparameters out of code; makes experiments reproducible |
| **Subset** for partitions | Memory-efficient views into original dataset (no data duplication) |
| **Non-IID: 2 classes/client** | Realistic simulation — each client has biased local data |

---

## 9. Dependencies (requirements.txt)

```
torch>=2.1.0
torchvision>=0.16.0
flwr>=1.5.0
opacus>=1.4.0
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
wandb>=0.16.0
pyyaml>=6.0
```
