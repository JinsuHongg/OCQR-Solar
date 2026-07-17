# OCQR-Solar: Ordinal Conformalized Quantile Regression for Solar Flare Forecasting

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)

OCQR-Solar is a specialized uncertainty quantification (UQ) and conformal prediction (CP) framework developed for high-stakes ordinal classification. The primary application of this architecture is space weather forecasting, specifically solar flare severity prediction.

## The Label Space

Solar flares exhibit a highly skewed, heavy-tailed distribution in severity. We model this physical phenomenon as a 5-class ordinal classification problem:

*   **Classes:** $K=5$ ordinal levels mapped to integers: `{"FQ/A": 0, "B": 1, "C": 2, "M": 3, "X": 4}`.
*   **Natural Ordering:** $0 < 1 < 2 < 3 < 4$.
*   **The Zero-Disjoint Axiom:** Prediction sets must never omit intermediate ordinal states. For instance, a prediction interval of `[0, 1, 2]` (FQ, B, C) is logically valid. Conversely, a disjoint set such as `[0, 4]` (FQ, X) or `[1, 3]` (B, M) physically violates domain constraints and represents a fatal algorithmic failure.

## Repository Structure

The repository is modularized into dedicated PyTorch Lightning components. Below is a high-level overview of the architectural structure:

```text
OCQR-Solar/
├── assets/                     # Persistent storage for model checkpoints (.ckpt), Wandb telemetry, and generated evaluation artifacts.
├── configs/                    # Hydra YAML configurations controlling hyperparameters for backbone models and conformal experiments.
│   ├── cls/                    # Configurations for baseline nominal classification backbones (ResNet).
│   └── qr/                     # Configurations for continuous quantile regression backbones (Pinball loss).
├── scripts/                    # Entry points for execution.
│   └── experiments/            # Scripts for initiating model training and executing conformal calibration loops.
└── src/ocqr_solar/             # Core Python package housing the primary logic.
    ├── datamodules/            # PyTorch Lightning DataModules (handles dynamic batching, cross-validation splits, and memory pinning).
    ├── datasets/               # Native PyTorch Dataset classes handling disk I/O for Adience, Retina-MNIST, and Solar Flare tensors.
    ├── explainability/         # Implementation of Mondrian conformal score computations and quantile thresholding operations.
    ├── metrics/                # Vectorized evaluators for contiguity (SFS, MDJ, CCR) and probability density.
    ├── models/                 # Neural architectures including base regressors, classifiers, and Lightning Module wrappers.
    └── utils/                  # Telemetry hooks, callback definitions, and helper functions.
```

## Core Methodology: The 3-Step Pipeline

Our framework enforces a 3-step hybrid interval methodology to guarantee contiguous predictions.

### Step 1: Continuous Interval Estimation
The backbone architecture (e.g., ResNet-18 or 3D CNNs) predicts two continuous quantile boundaries: Lower Bound $L(X)$ and Upper Bound $U(X)$ optimized via Pinball Loss. Monte Carlo Dropout (MCD) or Laplace approximations are integrated during inference to model epistemic uncertainty.

### Step 2: Mondrian (Class-Conditional) Conformal Calibration
Due to severe class imbalance (where severe M and X class flares represent <1% of the distribution), marginal calibration yields unstable coverage. The algorithm computes class-specific non-conformity scores $s_i = \max(L(X_i) - Y_i, Y_i - U(X_i))$ to establish conditional quantile correction factors $\hat{q}_k$ for each class $k \in \{0, 1, 2, 3, 4\}$.

### Step 3: Boundary-Based Ordinal Mapping
The calibrated continuous interval is defined as:

$$ \hat{I}(X) = [L(X) - \hat{q}_{y_{pred}}, U(X) + \hat{q}_{y_{pred}}] $$

Given predefined continuous domain thresholds $\{\tau_0, \tau_1, \dots, \tau_K\}$, a discrete ordinal class $k$ is included in the final prediction set $C(X)$ if and only if its continuous domain bin $[\tau_{k-1}, \tau_k)$ overlaps with the calibrated interval $\hat{I}(X)$. This continuous-to-discrete mapping mathematically guarantees zero disjoint gaps.

## Evaluation Metrics

Standard conformal prediction metrics, such as marginal coverage and set size, fail to penalize disjoint anomalies. OCQR-Solar evaluates model integrity using structural metrics:

*   **CCR (Contiguous Coverage Rate):** The primary benchmark metric. Defines the proportion of samples where the prediction set is strictly contiguous and successfully captures the ground truth label $Y$.
*   **SFS (Set Fragmentation Score):** Quantifies the number of disconnected sub-segments within a prediction set. The target value is exactly `1.0`. Any value `> 1.0` indicates a fundamental contiguity violation.
*   **MDJ (Maximum Disjoint Jump):** Quantifies the maximum magnitude of omitted intermediate classes (e.g., predicting `[1, 4]` produces an MDJ of `2`). The target value is `0.0`.

## Installation & Execution

### 1. Environment Initialization

Dependencies are rigidly managed via conda/mamba.
```bash
conda env create -f environment.yml
conda activate ocqr_solar
```

### 2. Available Benchmark Datasets

The repository supports multiple distinct DataModules to facilitate rigorous unit testing and ablation studies:
- **`FlareSuryaBench`**: The primary operational space-weather dataset of solar flare image sequences.
- **`Retina-MNIST`**: 5-class ordinal medical imaging benchmark for accelerated local algorithm validation.
- **`Adience`**: 8-class biological dataset emphasizing continuous physical quantity estimation.

### 3. Model Training

Initiate training for a chosen architecture using Hydra configuration overrides:
```bash
# Train the baseline ordinal classification model
python scripts/experiments/training.py --config-path ../../configs/cls --config-name CLS_resnet18_binomial_train_adience

# Train the continuous quantile regression backbone
python scripts/experiments/training.py --config-path ../../configs/qr --config-name QR_resnet18_train_adience
```

### 4. Conformal Calibration Validation

Following model convergence, extract calibrated prediction sets and analyze structural contiguity:
```bash
python scripts/experiments/calibration.py --config-path ../../configs/qr --config-name QR_resnet18_calibration_adience
```
