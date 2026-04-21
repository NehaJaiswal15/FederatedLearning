"""
Differential Privacy Utilities (Phase 6)

Wraps model + optimizer + dataloader with Opacus PrivacyEngine
to enable DP-SGD (Differentially Private Stochastic Gradient Descent).

How DP-SGD works:
1. CLIP:  Each sample's gradient is clipped to max_grad_norm
          (limits how much one sample can influence the model)
2. NOISE: Gaussian noise is added to the clipped gradients
          (makes it impossible to tell if a specific sample was used)
3. TRACK: Privacy budget (epsilon) is tracked across training steps
          (lower epsilon = stronger privacy guarantee)

Why this matters:
Without DP, an attacker could analyze model weight updates to infer
whether a specific person's data was used for training (membership
inference attack). DP-SGD mathematically prevents this.

Usage:
    from src.privacy.dp_utils import make_private, get_epsilon
"""

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def validate_model(model):
    """
    Check if the model is compatible with Opacus (DP-SGD).

    Opacus requires:
    - No BatchNorm (use GroupNorm instead)
    - No inplace operations (inplace=False on ReLU)
    - All layers must support per-sample gradients

    Our SimpleCNN was designed from the start to pass this check.

    Args:
        model (nn.Module): The model to validate

    Returns:
        nn.Module: The validated (and possibly auto-fixed) model
    """
    errors = ModuleValidator.validate(model, strict=False)

    if errors:
        print(f"  [DP] Model has {len(errors)} compatibility issue(s):")
        for err in errors:
            print(f"       - {err}")
        print("  [DP] Attempting automatic fix...")
        model = ModuleValidator.fix(model)
        print("  [DP] Model fixed successfully.")
    else:
        print("  [DP] Model is Opacus-compatible. No fixes needed.")

    return model


def make_private(model, optimizer, dataloader, config):
    """
    Wrap model + optimizer + dataloader with Opacus PrivacyEngine.

    This is the core function that enables differential privacy.
    After calling this, every optimizer.step() will:
    1. Clip per-sample gradients to max_grad_norm
    2. Add calibrated Gaussian noise
    3. Update the privacy budget (epsilon)

    Args:
        model (nn.Module): The neural network model
        optimizer (Optimizer): SGD or Adam optimizer
        dataloader (DataLoader): Training data loader
        config (dict): Privacy config with:
            - max_grad_norm (float): Maximum L2 norm for gradient clipping
            - noise_multiplier (float): How much noise to add (higher = more private)
            - target_epsilon (float): Privacy budget target

    Returns:
        tuple: (dp_model, dp_optimizer, dp_dataloader, privacy_engine)
    """
    privacy_cfg = config["privacy"]

    max_grad_norm = privacy_cfg["max_grad_norm"]
    noise_multiplier = privacy_cfg["noise_multiplier"]

    print(f"  [DP] Enabling Differential Privacy (DP-SGD)")
    print(f"       Max Grad Norm:    {max_grad_norm}")
    print(f"       Noise Multiplier: {noise_multiplier}")
    print(f"       Target Epsilon:   {privacy_cfg['target_epsilon']}")

    # Step 1: Validate and fix model if needed
    model = validate_model(model)

    # Step 2: Create and attach PrivacyEngine
    privacy_engine = PrivacyEngine()

    dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    print(f"  [DP] PrivacyEngine attached successfully.")

    return dp_model, dp_optimizer, dp_dataloader, privacy_engine


def get_epsilon(privacy_engine, delta=1e-5):
    """
    Get the current privacy budget (epsilon) spent so far.

    Epsilon measures how much privacy has been "used up":
    - epsilon < 1:   Very strong privacy (model barely learned)
    - epsilon 1-10:  Good privacy-utility balance
    - epsilon > 10:  Weaker privacy but better accuracy

    The delta parameter is the probability that the privacy
    guarantee fails. 1e-5 is standard practice.

    Args:
        privacy_engine (PrivacyEngine): The attached privacy engine
        delta (float): Privacy failure probability (default: 1e-5)

    Returns:
        float: Current epsilon value
    """
    return privacy_engine.get_epsilon(delta=delta)
