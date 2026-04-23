"""
FastAPI Backend (Phase 8)

REST API for triggering training experiments and retrieving results.
Provides endpoints to run centralized and federated training,
view/update configuration, and fetch experiment results.

Usage:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /train/centralized  - Start centralized training
    POST /train/federated    - Start federated training
    GET  /results            - Fetch latest experiment results
    GET  /config             - Return current config
    POST /config             - Update config parameters
"""

import copy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.utils.config_loader import load_config


# ── Initialize FastAPI app ──
app = FastAPI(
    title="Federated Learning API",
    description="REST API for Privacy-Preserving Federated Image Classification",
    version="1.0.0",
)

# Allow requests from the Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory storage for results ──
latest_results = {
    "centralized": None,
    "federated": None,
}

# ── Load current config ──
current_config = load_config()


# ── Pydantic models for request/response validation ──

class TrainingRequest(BaseModel):
    """Optional overrides for training parameters."""
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None

class FederatedRequest(BaseModel):
    """Optional overrides for federated training parameters."""
    num_clients: Optional[int] = None
    num_rounds: Optional[int] = None
    fraction_fit: Optional[float] = None
    enable_dp: Optional[bool] = None
    noise_multiplier: Optional[float] = None

class ConfigUpdate(BaseModel):
    """Partial config update. Only provided fields are changed."""
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    num_clients: Optional[int] = None
    num_rounds: Optional[int] = None
    fraction_fit: Optional[float] = None
    enable_dp: Optional[bool] = None
    noise_multiplier: Optional[float] = None
    max_grad_norm: Optional[float] = None
    iid: Optional[bool] = None


# ── Helper: apply overrides to config ──
def apply_overrides(config, overrides):
    """
    Apply request overrides to a copy of the config.

    Args:
        config (dict): Base config dictionary.
        overrides: Pydantic model with optional fields.

    Returns:
        dict: Config with overrides applied.
    """
    cfg = copy.deepcopy(config)
    override_dict = overrides.dict(exclude_none=True)

    # Map flat override fields to nested config structure
    field_map = {
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "learning_rate": ("training", "learning_rate"),
        "num_clients": ("federated", "num_clients"),
        "num_rounds": ("federated", "num_rounds"),
        "fraction_fit": ("federated", "fraction_fit"),
        "enable_dp": ("privacy", "enable_dp"),
        "noise_multiplier": ("privacy", "noise_multiplier"),
        "max_grad_norm": ("privacy", "max_grad_norm"),
        "iid": ("data", "iid"),
    }

    for field, value in override_dict.items():
        if field in field_map:
            section, key = field_map[field]
            cfg[section][key] = value

    return cfg


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "project": "Privacy-Preserving Federated Learning",
        "endpoints": [
            "POST /train/centralized",
            "POST /train/federated",
            "GET /results",
            "GET /config",
            "POST /config",
        ],
    }


@app.post("/train/centralized")
def train_centralized(request: TrainingRequest = TrainingRequest()):
    """
    Start centralized (non-federated) training.

    Optionally override epochs, batch_size, or learning_rate.
    Returns training results with per-epoch metrics.
    """
    from src.training.train import run_centralized_training

    config = apply_overrides(current_config, request)

    try:
        results = run_centralized_training(config=config)
        latest_results["centralized"] = results
        return {
            "status": "completed",
            "mode": "centralized",
            "final_test_accuracy": results["final_test_accuracy"],
            "epochs": len(results["train_losses"]),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/federated")
def train_federated(request: FederatedRequest = FederatedRequest()):
    """
    Start federated training with FedAvg.

    Optionally override num_clients, num_rounds, fraction_fit,
    enable_dp, and noise_multiplier.
    Returns training history with per-round metrics.
    """
    from src.federated.server import run_federated_training

    config = apply_overrides(current_config, request)

    try:
        history = run_federated_training(config=config)
        latest_results["federated"] = history
        return {
            "status": "completed",
            "mode": "federated",
            "final_test_accuracy": history["test_accuracy"][-1],
            "best_test_accuracy": max(history["test_accuracy"]),
            "rounds": len(history["round"]),
            "dp_enabled": config["privacy"]["enable_dp"],
            "results": history,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results")
def get_results():
    """
    Fetch the latest experiment results.

    Returns results from the most recent centralized and/or
    federated training runs.
    """
    if latest_results["centralized"] is None and latest_results["federated"] is None:
        return {
            "status": "no_results",
            "message": "No training has been run yet. Use POST /train/centralized or /train/federated.",
        }

    response = {"status": "ok"}

    if latest_results["centralized"] is not None:
        response["centralized"] = {
            "final_test_accuracy": latest_results["centralized"]["final_test_accuracy"],
            "epochs": len(latest_results["centralized"]["train_losses"]),
            "train_accuracies": latest_results["centralized"]["train_accuracies"],
            "test_accuracies": latest_results["centralized"]["test_accuracies"],
        }

    if latest_results["federated"] is not None:
        response["federated"] = {
            "final_test_accuracy": latest_results["federated"]["test_accuracy"][-1],
            "best_test_accuracy": max(latest_results["federated"]["test_accuracy"]),
            "rounds": len(latest_results["federated"]["round"]),
            "test_accuracies": latest_results["federated"]["test_accuracy"],
        }

    return response


@app.get("/config")
def get_config():
    """Return the current configuration."""
    return {
        "status": "ok",
        "config": current_config,
    }


@app.post("/config")
def update_config(update: ConfigUpdate):
    """
    Update configuration parameters.

    Only the fields you provide will be changed.
    Other fields remain at their current values.
    """
    global current_config
    updated = apply_overrides(current_config, update)
    current_config = updated

    return {
        "status": "updated",
        "config": current_config,
    }
