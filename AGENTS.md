# Antigravity Agent Context & Architectural Guardrails
**Project Name:** `OCQR-Solar` (Ordinal Conformalized Quantile Regression for Solar Flare Forecasting)  
**Primary Target:** IEEE Transactions on Geoscience and Remote Sensing (TGRS) / Machine Learning Top-Tier Journals  

---

## 1. Project Overview & Core Mission
`OCQR-Solar` is a specialized uncertainty quantification (UQ) and conformal prediction (CP) codebase designed for **high-stakes ordinal classification**, with a primary focus on space weather (solar flare severity forecasting). 

The fundamental mission of this codebase is to guarantee **strictly contiguous (zero-disjoint) prediction sets** over naturally ordered label spaces while maintaining statistically valid class-conditional coverage ($1 - \alpha$).

### The Label Space (Solar Flares)
* **Classes:** $K=5$ ordinal classes mapped to integers: `{"FQ/A": 0, "B": 1, "C": 2, "M": 3, "X": 4}`.
* **Natural Ordering:** $0 < 1 < 2 < 3 < 4$.
* **The "Zero-Disjoint" Rule:** Prediction sets must NEVER skip intermediate ordinal levels. A prediction set like `[0, 1, 2]` (FQ, B, C) is logically valid. A disjoint set like `[0, 4]` (FQ, X) or `[1, 3]` (B, M) physically violates domain constraints and is considered a **fatal algorithmic failure**.

---

## 2. Core Methodology (The 3-Step Pipeline)
When implementing, refactoring, or evaluating models in this repository, you MUST adhere to the following 3-step hybrid interval framework:

1. **Step 1: Continuous Interval Estimation via CQR + MCD**
   * The backbone model (e.g., ResNet-18) predicts two continuous quantile boundaries: Lower Bound $L(X)$ and Upper Bound $U(X)$ using Pinball Loss.
   * Monte Carlo Dropout (MCD) is integrated during inference to account for epistemic uncertainty.
2. **Step 2: Mondrian (Class-Conditional) Conformal Calibration**
   * Due to extreme class imbalance (rare M and X class flares), marginal calibration is insufficient.
   * You must compute class-specific non-conformity scores $s_i = \max(L(X_i) - Y_i, Y_i - U(X_i))$ and establish class-specific quantile correction factors $\hat{q}_k$ for each class $k \in \{0, 1, 2, 3, 4\}$.
3. **Step 3: Boundary-Based Ordinal Mapping**
   * The calibrated continuous interval is $\hat{I}(X) = [L(X) - \hat{q}_{y_{pred}}, U(X) + \hat{q}_{y_{pred}}]$.
   * Given predefined continuous domain thresholds $\{\tau_0, \tau_1, \dots, \tau_K\}$, a discrete ordinal class $k$ is included in the final prediction set $C(X)$ **if and only if** its domain bin $[\tau_{k-1}, \tau_k)$ overlaps with $\hat{I}(X)$. This continuous-to-discrete bridge mathematically guarantees zero disjoint gaps.

---

## 3. Custom Evaluation Metrics
When writing validation, test, or evaluation loops, you must calculate and log the following custom metrics located in `src/ocqr_solar/metrics/`:

* **`SFS` (Set Fragmentation Score):** Measures the number of disconnected segments in a set. Target value is exactly `1.0`. Any value `> 1.0` indicates a contiguity violation.
* **`MDJ` (Maximum Disjoint Jump):** Measures the maximum gap size of omitted intermediate classes (e.g., `[1, 4]` has an MDJ of `2`). Target value is `0.0`.
* **`CCR` (Contiguous Coverage Rate):** The primary benchmark metric. The proportion of samples where the prediction set is strictly contiguous (`SFS == 1`) AND successfully contains the ground truth label $Y$.

---

## 4. Strict Engineering Rules & Coding Standards
As an AI agent contributing to this repository, you must strictly enforce the following senior-level PyTorch/PyTorch Lightning engineering rules:

### RULE 1: Zero CPU-GPU Synchronization Bottlenecks
* **NEVER** use `.item()`, `.numpy()`, or `.cpu()` inside batch loops or vector processing steps during training, validation, or calibration.
* **NEVER** use Python `list.append()` inside forward loops to collect per-sample scores if vectorization is possible.
* Use PyTorch matrix broadcasting, `torch.where()`, `torch.gather()`, and cumulative operations (`torch.cumsum`) entirely within CUDA memory.

### RULE 2: Safe Tensor Dimension Squeezing
* **NEVER** call `.squeeze()` without arguments on model predictions or targets (e.g., `preds.squeeze()`). When `batch_size == 1`, this collapses 1D tensors into 0D scalars, causing fatal runtime exceptions in downstream loops.
* **ALWAYS** use `.view(-1)` or explicitly specify the dimension to squeeze (e.g., `.squeeze(-1)`).

### RULE 3: Safe Laplace Approximation (`laplace-torch`)
* Raw PyTorch Lightning modules conflict with Kronecker-factored Hessian approximations in `laplace-torch` due to structural parameter routing.
* **ALWAYS** wrap the extracted backbone network in `SafeLaplaceModel` (inheriting directly from `nn.Module`) before passing it to `Laplace()`.

### RULE 4: Out-of-Bounds Index Protection in Small Samples
* In Mondrian calibration, rare classes may have very few calibration samples ($n_k$).
* When computing empirical quantile indices via order statistics, **ALWAYS** clamp the integer index within safe boundaries: `q_idx = min(max(int(np.ceil((n + 1) * (1.0 - alpha))), 1), n) - 1`.

### RULE 5: use subagents "plan" and "build" agents.
* invoke those two agents for all tasks.

---

## 5. Auxiliary Benchmarks for Unit Testing
For fast local debugging and integration tests without loading heavy solar image datasets, utilize the lightweight datasets configured in `src/ocqr_solar/datasets/`:
* **`Retina-MNIST`:** 5 ordinal classes, 1,600 samples ($28 \times 28$ images). The primary sandbox for testing 5-level Mondrian calibration and zero-disjoint guarantees.
* **`Wine Quality`:** 7 ordinal classes, tabular data. Used for sub-second unit testing of calibration logic and metric calculations.

### RULE 6: Conda Environment Default
* **ALWAYS** activate the `ocqr_solar` conda/mamba environment before executing any scripts, running tests, or installing packages.

