"""
Streamlit Dashboard (Phase 9)

Interactive visualization dashboard for the Federated Learning project.
Provides controls to run experiments and visualize results.

Features:
    - Training loss/accuracy curves
    - Centralized vs Federated comparison charts
    - Per-client data distribution visualization (IID vs Non-IID)
    - Privacy budget (epsilon) tracker
    - Controls to adjust config and trigger training via API

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import json
import os

# Page Configuration 
st.set_page_config(
    page_title="Federated Learning Dashboard",
    page_icon="FL",
    layout="wide",
)

#  Styling 
sns.set_theme(style="darkgrid")

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

#  SIDEBAR: Configuration & Controls

st.sidebar.title("Configuration")

st.sidebar.subheader("Training Settings")
epochs = st.sidebar.slider("Epochs (local)", 1, 20, 1)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.01,
)

st.sidebar.subheader("Federated Settings")
num_clients = st.sidebar.slider("Number of Clients", 2, 10, 5)
num_rounds = st.sidebar.slider("Number of Rounds", 1, 50, 10)
fraction_fit = st.sidebar.slider("Fraction Fit", 0.1, 1.0, 0.5, step=0.1)
iid = st.sidebar.toggle("IID Data Distribution", value=True)

st.sidebar.subheader("Privacy Settings")
enable_dp = st.sidebar.toggle("Enable Differential Privacy", value=False)
noise_multiplier = st.sidebar.slider("Noise Multiplier", 0.1, 2.0, 0.5, step=0.1)
max_grad_norm = st.sidebar.slider("Max Gradient Norm", 0.1, 5.0, 1.0, step=0.1)


# Helper: check if API is running
def api_available():
    try:
        r = requests.get(f"{API_URL}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# Helper: plot training curves 
def plot_training_curves(results, title, x_label="Epoch"):
    """Plot loss and accuracy curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = range(1, len(results["train_losses"]) + 1)

    # Loss plot
    ax1.plot(x, results["train_losses"], "o-", label="Train Loss", color="#FF6B6B", linewidth=2)
    ax1.plot(x, results["test_losses"], "s-", label="Test Loss", color="#4ECDC4", linewidth=2)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(x, results["train_accuracies"], "o-", label="Train Acc", color="#FF6B6B", linewidth=2)
    ax2.plot(x, results["test_accuracies"], "s-", label="Test Acc", color="#4ECDC4", linewidth=2)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_federated_curves(history, title="Federated Training"):
    """Plot federated training metrics per round."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    rounds = history["round"]

    # Loss plot
    ax1.plot(rounds, history["train_losses"], "o-", label="Avg Train Loss", color="#FF6B6B", linewidth=2)
    ax1.plot(rounds, history["test_loss"], "s-", label="Test Loss", color="#4ECDC4", linewidth=2)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(rounds, history["train_accuracies"], "o-", label="Avg Train Acc", color="#FF6B6B", linewidth=2)
    ax2.plot(rounds, history["test_accuracy"], "s-", label="Test Acc", color="#4ECDC4", linewidth=2)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison(centralized_results, federated_results):
    """Plot centralized vs federated accuracy comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Centralized\nBaseline"]
    accuracies = [centralized_results["final_test_accuracy"]]
    colors = ["#4ECDC4"]

    if federated_results is not None:
        fed_acc = federated_results["test_accuracy"][-1]
        dp_label = " (with DP)" if federated_results.get("dp_enabled", False) else ""
        categories.append(f"Federated{dp_label}")
        accuracies.append(fed_acc)
        colors.append("#FF6B6B")

    bars = ax.bar(categories, accuracies, color=colors, width=0.5, edgecolor="white", linewidth=2)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=14)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Centralized vs Federated Comparison")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_data_distribution(num_clients, iid):
    """Visualize how data is distributed across clients."""
    fig, ax = plt.subplots(figsize=(10, 5))

    classes = ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus",
               "Smooth Muscle", "Normal Mucosa", "Stroma", "Epithelium"]

    if iid:
        # IID: each client gets equal samples of all classes
        data = np.ones((num_clients, 9)) * (5000 // num_clients // 9)
    else:
        # Non-IID: each client gets only 2 classes
        data = np.zeros((num_clients, 9))
        for i in range(num_clients):
            cls1 = (i * 2) % 9
            cls2 = (i * 2 + 1) % 9
            data[i, cls1] = 5000 // num_clients // 2
            data[i, cls2] = 5000 // num_clients // 2

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(9))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels([f"Client {i}" for i in range(num_clients)])
    ax.set_title(f"Data Distribution ({'IID' if iid else 'Non-IID'})")

    plt.colorbar(im, ax=ax, label="Samples per class")
    plt.tight_layout()
    return fig


def plot_epsilon_tracker(epsilons, target_epsilon=10.0):
    """Plot privacy budget (epsilon) over training rounds."""
    fig, ax = plt.subplots(figsize=(8, 4))

    rounds = range(1, len(epsilons) + 1)
    ax.plot(rounds, epsilons, "o-", color="#9B59B6", linewidth=2, label="Epsilon (spent)")
    ax.axhline(y=target_epsilon, color="#E74C3C", linestyle="--", linewidth=2, label=f"Target (e={target_epsilon})")

    ax.fill_between(rounds, epsilons, alpha=0.2, color="#9B59B6")
    ax.set_xlabel("Round")
    ax.set_ylabel("Epsilon")
    ax.set_title("Privacy Budget Tracker")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# MAIN PAGE

st.title("Privacy-Preserving Federated Learning")
st.markdown("Interactive dashboard for training, comparing, and analyzing federated learning experiments with differential privacy.")

# Features & How to Use Guide
with st.expander("Features & How to Use", expanded=False):
    st.markdown("""
### Sidebar - Configuration Panel (left side)
Adjust all settings here **before** clicking a training button.

**Training Settings:**
- **Epochs (local):** How many times the model processes the full dataset per round. More epochs = better learning but slower.
- **Batch Size:** Images processed per step. Larger = faster but uses more memory.
- **Learning Rate:** How aggressively the model updates. Too high = overshoots, too low = learns too slowly.

**Federated Settings:**
- **Number of Clients:** Simulated devices that each hold a portion of the data (2-10).
- **Number of Rounds:** How many times the server aggregates client updates (1-50).
- **Fraction Fit:** Percentage of clients selected each round. 0.5 = half participate per round.
- **IID Data Distribution:** ON = data evenly split across clients. OFF = each client only has 2 of 10 classes (harder, more realistic).

**Privacy Settings:**
- **Enable Differential Privacy:** Adds calibrated noise to gradients (DP-SGD) to protect individual data points.
- **Noise Multiplier:** Controls noise amount. Higher = stronger privacy but lower accuracy.
- **Max Gradient Norm:** Clips gradients to limit any single sample's influence.

---

### Dashboard Sections

**1. Data Distribution Heatmap**
Visualizes how PathMNIST tissue classes are split across clients. In Non-IID mode, each client only sees 2 classes, making federated training harder.

**2. Training Controls**
Two buttons to trigger training via the API:
- *Run Centralized Training* - Trains on all data in one place (baseline).
- *Run Federated Training* - Trains across multiple clients using FedAvg.
- Buttons are disabled if the API server is not running.

**3. Training Results (Charts)**
Appear after training completes:
- **Loss curves** (should decrease) and **Accuracy curves** (should increase).
- Red line = training performance, Teal line = test performance.

**4. Comparison Bar Chart**
Appears after running both centralized AND federated training. Directly compares their test accuracies.

**5. Privacy Budget (Epsilon) Tracker**
Appears when DP is enabled. Shows how the privacy budget is consumed over training rounds. When epsilon exceeds the target, privacy guarantees weaken.

**6. Summary Table**
A quick-glance comparison of all completed experiments with final accuracy, loss, and number of epochs/rounds.
    """)

# API Status
api_status = api_available()
if api_status:
    st.success("API is running at " + API_URL)
else:
    st.warning("API is not running. Start it with: `uvicorn api.main:app --reload --port 8000`")

st.markdown("---")

# Data Distribution Visualization
st.header("Data Distribution Across Clients")
st.markdown("Shows how PathMNIST data is split across clients. IID gives each client all 9 classes; Non-IID restricts each client to only 2 classes.")

fig_dist = plot_data_distribution(num_clients, iid)
st.pyplot(fig_dist)
plt.close()

st.markdown("---")

# Training Controls
st.header("Run Training")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Centralized Training")
    st.markdown("Train on all data in one place (baseline).")
    if st.button("Run Centralized Training", disabled=not api_status, type="primary"):
        with st.spinner("Training centralized model..."):
            try:
                response = requests.post(
                    f"{API_URL}/train/centralized",
                    json={"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate},
                    timeout=600,
                )
                if response.status_code == 200:
                    result = response.json()
                    st.session_state["centralized_results"] = result["results"]
                    st.success(f"Done! Test Accuracy: {result['final_test_accuracy']:.2f}%")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

with col2:
    st.subheader("Federated Training")
    dp_text = " with DP-SGD" if enable_dp else ""
    st.markdown(f"Train across {num_clients} clients using FedAvg{dp_text}.")
    if st.button("Run Federated Training", disabled=not api_status, type="primary"):
        with st.spinner(f"Running {num_rounds} rounds of federated training..."):
            try:
                response = requests.post(
                    f"{API_URL}/train/federated",
                    json={
                        "num_clients": num_clients,
                        "num_rounds": num_rounds,
                        "fraction_fit": fraction_fit,
                        "enable_dp": enable_dp,
                        "noise_multiplier": noise_multiplier,
                    },
                    timeout=1800,
                )
                if response.status_code == 200:
                    result = response.json()
                    st.session_state["federated_results"] = result["results"]
                    st.session_state["federated_dp"] = enable_dp
                    st.success(f"Done! Test Accuracy: {result['final_test_accuracy']:.2f}%")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

st.markdown("---")

# Results Visualization
st.header("Training Results")

# Centralized results
if "centralized_results" in st.session_state:
    st.subheader("Centralized Training Curves")
    fig_central = plot_training_curves(
        st.session_state["centralized_results"],
        "Centralized Training"
    )
    st.pyplot(fig_central)
    plt.close()

# Federated results
if "federated_results" in st.session_state:
    st.subheader("Federated Training Curves")
    fig_fed = plot_federated_curves(st.session_state["federated_results"])
    st.pyplot(fig_fed)
    plt.close()

# Comparison chart
if "centralized_results" in st.session_state and "federated_results" in st.session_state:
    st.subheader("Centralized vs Federated Comparison")
    fig_comp = plot_comparison(
        st.session_state["centralized_results"],
        st.session_state["federated_results"],
    )
    st.pyplot(fig_comp)
    plt.close()

# Epsilon tracker (placeholder for when DP results are available)
if "federated_results" in st.session_state and st.session_state.get("federated_dp", False):
    st.subheader("Privacy Budget (Epsilon) Tracker")
    fed_results = st.session_state["federated_results"]
    # If epsilon data is available in the results
    if "epsilons" in fed_results:
        fig_eps = plot_epsilon_tracker(fed_results["epsilons"])
        st.pyplot(fig_eps)
        plt.close()
    else:
        st.info("Epsilon tracking data will be available when DP is enabled during federated training.")

st.markdown("---")

# Metrics Summary Table 
if "centralized_results" in st.session_state or "federated_results" in st.session_state:
    st.header("Summary")

    summary_data = []
    if "centralized_results" in st.session_state:
        cr = st.session_state["centralized_results"]
        summary_data.append({
            "Mode": "Centralized",
            "Final Test Accuracy (%)": f"{cr['final_test_accuracy']:.2f}",
            "Final Train Loss": f"{cr['train_losses'][-1]:.4f}",
            "Epochs/Rounds": len(cr["train_losses"]),
        })
    if "federated_results" in st.session_state:
        fr = st.session_state["federated_results"]
        dp_status = "Yes" if st.session_state.get("federated_dp", False) else "No"
        summary_data.append({
            "Mode": f"Federated (DP: {dp_status})",
            "Final Test Accuracy (%)": f"{fr['test_accuracy'][-1]:.2f}",
            "Final Train Loss": f"{fr['train_losses'][-1]:.4f}",
            "Epochs/Rounds": len(fr["round"]),
        })

    st.table(summary_data)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Privacy-Preserving Federated Image Classification | "
    "Built with PyTorch, Flower, Opacus, FastAPI & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
