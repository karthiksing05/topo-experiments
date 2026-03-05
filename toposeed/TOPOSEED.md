# TopoSeed — Technical Reference

## Table of Contents

1. [Core Methodology](#1-core-methodology)
2. [File Overview](#2-file-overview)
3. [`utils.py`](#3-utilspy)
4. [`sheet.py`](#4-sheetpy)
5. [`buffers.py`](#5-bufferspy)
6. [`expansion.py`](#6-expansionpy)
7. [`regularizers.py`](#7-regularizerspy)
8. [`layer.py`](#8-layerpy)
9. [Training Loop Contract](#9-training-loop-contract)
10. [Hyperparameter Guide](#10-hyperparameter-guide)

---

## 1. Core Methodology

TopoSeed is a PyTorch module that induces **brain-like topographic organisation** in neural
network layers through a *seed-based growth* process on a 2D cortical sheet.

### The Cortical Sheet

Every weight matrix in a standard network is a flat 2D tensor `(out, in)`.  TopoSeed
*re-interprets* this as a 3D spatial volume:

```
Linear:   W ∈ R^(out_features × in_features)
          → cortical sheet C ∈ R^(H × W × D)
             H × W = out_features  (area  = output neurons)
             D     = in_features   (depth = input dimensionality)

Conv2d:   W ∈ R^(c_out × c_in × kH × kW)
          → C ∈ R^(H × W × D)
             H × W = c_out
             D     = c_in × kH × kW
```

`H` and `W` are chosen to be as close to square as possible (minimising perimeter)
via `compute_sheet_dimensions`.  All spatial operations — neighbour lookup, patch
membership, expansion footprint, seed placement — are computed in this `H×W` space.

### Seeds

At initialisation the sheet is divided into a **G×G** grid of equal-area cells.
One *seed* neuron is placed at the centre of each cell, giving `G²` starting neurons.
All remaining `H×W − G²` neurons are **dormant**: their weights exist in the parameter
tensor but are multiplied to zero by a binary `active_mask`, so they:

- contribute nothing to the forward pass
- receive no gradient

Seeds are not privileged; they can die like any other neuron.

### Spatially Correlated Initialisation

Seed weights are initialised with Kaiming-scale noise that is spatially correlated
across the sheet:

```
shared_noise     = gaussian_blur(randn(H, W, D), sigma=grid_spacing × 0.5)
independent_noise = randn(H, W, D) × 0.3
seed_weights      = kaiming_base + shared_noise × 0.2 + independent_noise × 0.8
```

Nearby seeds share ~20–30 % of their initialisation variance; distant seeds are
independent.

### Lifecycle: Evidence → Expansion → Death

Each training step, three signals are tracked **per neuron**:

| Signal | Buffer | Timescale |
|---|---|---|
| `gradient_magnitude × activation` | `EvidenceBuffer` (fast EMA) | α = 0.05 |
| `gradient_magnitude × activation` | `HealthBuffer` (slow EMA) | α = 0.01 |
| Per-patch residual error | `ResidualBuffer` | α = 0.05 |

**Expansion** fires when a patch's combined signal exceeds `expansion_threshold`:

```
combined_signal = evidence_ema + residual_weight × residual_ema
if combined_signal > expansion_threshold:
    recruit dormant neighbours of each boundary neuron
```

New neurons inherit the boundary neuron's weights plus a small gradient-biased
perturbation, and are assigned to the expanding patch.

**Death** fires when a neuron's health EMA stays below `death_threshold` for
`death_sustained_steps` consecutive steps — the neuron is deactivated and its
weights are zeroed.

Both expansion and death are **gated by a warmup period** (`warmup_steps`) during
which `active_mask` is frozen.

### Patch Membership

Each active neuron belongs to exactly one patch (identified by `patch_id`).
After each expansion/death cycle, `update_membership` reassigns low-confidence
neurons to their spatially nearest seed.

### Regularisation

Two loss terms encourage topographic coherence:

- **Intra-layer smoothness** — penalises cosine dissimilarity between same-patch
  neighbours on the sheet (computed within `forward()`).
- **Cross-layer coherence** — penalises cases where spatially close neurons in
  layer L have divergent outgoing connections to layer L+1 (sampled-pair
  approximation, computed externally by the training loop).

---

## 2. File Overview

```
toposeed/
├── utils.py          Grid/sheet geometry, Gaussian blur, device helpers
├── sheet.py          CorticalSheet — the spatial weight store
├── buffers.py        EvidenceBuffer, HealthBuffer, ResidualBuffer
├── expansion.py      ExpansionManager — lifecycle orchestration
├── regularizers.py   intra_layer_smoothness, cross_layer_coherence
├── layer.py          TopoSeedLayer — public API / nn.Module
└── __init__.py       Package re-exports
```

---

## 3. `utils.py`

### `get_device(preferred="auto") → torch.device`

Returns the best available device.  Priority order: `cuda > mps > cpu`.
Pass a string like `"cuda"`, `"mps"`, or `"cpu"` to force a specific device.

### `compute_sheet_dimensions(n_neurons) → (H, W)`

Finds the most square factorisation of `n_neurons` such that `H × W ≥ n_neurons`
and `H × W` is minimised.  `H ≤ W` always.

### `place_seeds_on_grid(H, W, G) → list[(i, j)]`

Returns `G²` seed positions centred within each cell of a `G×G`
grid overlaid on the `H×W` sheet.

### `spatial_distance_matrix(positions) → Tensor(N, N)`

Pairwise Euclidean distances between a list of `(i, j)` positions.

### `gaussian_blur_2d(tensor, sigma) → Tensor(H, W, D)`

Applies a separable 2D Gaussian blur to a `(H, W, D)` tensor independently
over the D dimension.  Uses grouped `F.conv2d` with input shape `(1, D, H, W)`.
Used during spatially correlated initialisation.

### `correlated_init(H, W, D, seed_positions, grid_spacing, std, device) → Tensor`

Builds the initial `(H, W, D)` weight tensor with spatial correlation as described
in §1.  Non-seed positions initialised to the same formula; masking is applied
separately by `CorticalSheet`.

### `active_mask_expanded(mask, D) → Tensor(H, W, D)`

Broadcasts a `(H, W)` binary mask to `(H, W, D)` for element-wise weight masking.

### `weights_to_matrix(weights, n_out) → Tensor(n_out, D)`

Flattens the cortical sheet `(H, W, D)` to a weight matrix, keeping only the
first `n_out` rows (the sheet may be slightly larger than `n_out`).

### `matrix_to_sheet(matrix, H, W) → Tensor(H, W, D)`

Pads or trims a `(n_out, D)` weight matrix to exactly `H×W` rows and reshapes
to `(H, W, D)`.

---

## 4. `sheet.py`

### `class CorticalSheet(nn.Module)`

Owns all spatial state for one layer.

#### Construction

```python
CorticalSheet(H, W, D, grid_size, n_neurons, std=1.0, device=None)
```

On construction:
- Allocates `self.weights: nn.Parameter` of shape `(H, W, D)` with correlated init
- Registers `active_mask (H, W)` — 1 = active, 0 = dormant
- Registers `patch_id (H, W)` — integer patch index; −1 for dormant neurons
- Registers `confidence (H, W)` — per-neuron patch-membership confidence
- Populates `seed_positions` (list of `(i, j)`, one per patch)

#### `get_active_weight_matrix() → Tensor(n_neurons, D)`

Returns `weights * active_mask_expanded`, reshaped to a standard weight matrix.
The mask multiplication is **autograd-safe** — dormant neurons receive zero gradient
without any in-place operation or `detach`.

#### `get_neighbors(i, j, radius=1) → list[(i, j)]`

Returns dormant neighbours within Manhattan distance `radius` of `(i, j)`.
Used by `ExpansionManager` to find recruitment targets.

#### `activate_neuron(i, j, pid, source_weights, gradient)`

Wakes a dormant neuron:
- Sets `active_mask[i, j] = 1`
- Initialises weights as `source_weights + small_perturbation + gradient_bias`
- Assigns to patch `pid` with initial confidence 0.5

#### `deactivate_neuron(i, j)`

Kills an active neuron: zeros its weights, sets `active_mask = 0`, `patch_id = −1`.

#### `update_membership(beta)`

Reassigns neurons whose `confidence < beta` to their spatially nearest seed
(Euclidean on the sheet).  High-confidence neurons keep their current patch.

#### `active_count() / dormant_count() / num_patches() / mean_patch_size()`

Diagnostic counters used by `get_stats()` and logging.

#### `reset_epoch_counters()`

Resets the per-epoch `expansions_this_epoch` / `deaths_this_epoch` counters.

---

## 5. `buffers.py`

Plain Python objects (not `nn.Module`s) holding tensor state.  Each has a
`.to(device)` method so `TopoSeedLayer._sync_non_module_buffers()` can keep them
on the correct device after model moves.

### `class EvidenceBuffer`

```python
EvidenceBuffer(H, W, window_size=100, ema_alpha=0.05, device=None)
```

Tracks `gradient_magnitude × activation` via exponential moving average.

| Method | Description |
|---|---|
| `update(grad_magnitude, activation, active_mask)` | Updates EMA: `ema = (1−α)·ema + α·(grad×act×mask)` |
| `get_patch_evidence(patch_id, pid) → float` | Mean EMA value for all neurons in patch `pid` |
| `reset_patch(patch_id, pid)` | Zeros EMA for neurons in patch `pid` after an expansion; uses `torch.where` (MPS-safe) |

### `class HealthBuffer`

```python
HealthBuffer(H, W, window_size=500, ema_alpha=0.01, device=None)
```

Slower EMA (α = 0.01) plus a cumulative low-signal counter per neuron.

| Method | Description |
|---|---|
| `update(grad_magnitude, activation, active_mask)` | Updates health EMA; increments `low_health_count` for signal below `1e-5`; resets counter on recovery.  Uses arithmetic ops + `torch.where` (MPS-safe) |
| `get_death_candidates(threshold, sustained_steps) → list[(i,j)]` | Returns positions where `health < threshold` AND `low_health_count > sustained_steps` |

### `class ResidualBuffer`

```python
ResidualBuffer(num_patches, ema_alpha=0.05, device=None)
```

Per-patch scalar residual EMA.  Complementary signal to `EvidenceBuffer`.

| Method | Description |
|---|---|
| `update(patch_residuals)` | `residual_ema = (1−α)·residual_ema + α·patch_residuals` |
| `get_patch_residual(pid) → float` | Current EMA value for patch `pid` |

---

## 6. `expansion.py`

### `class ExpansionManager`

```python
ExpansionManager(
    sheet, evidence_buffer, health_buffer, residual_buffer,
    warmup_steps, expansion_threshold, death_threshold,
    death_sustained_steps, expansion_radius=1,
    residual_weight=0.5, beta=0.7,
)
```

Orchestrates the full lifecycle for one layer.

#### `step_update(grad_magnitude, activation, patch_residuals)`

Called every training step (after `loss.backward()`).  Updates all three buffers.
After warmup, calls `_check_expansions()`, `_check_deaths()`, then
`sheet.update_membership(beta)`.

#### `_check_expansions()`

For each patch, computes:

```
signal = evidence_ema + residual_weight × residual_ema
```

If `signal > expansion_threshold`, calls `_expand_patch(pid)` and resets the
patch's evidence EMA to 0.

#### `_expand_patch(pid)`

Finds the **boundary neurons** of the patch (active neurons with at least one
dormant neighbour), then recruits all dormant neighbours within `expansion_radius`
via `sheet.activate_neuron()`.

#### `_check_deaths()`

Calls `health_buffer.get_death_candidates()` and deactivates each candidate via
`sheet.deactivate_neuron()`.  Never kills the last remaining neuron.

#### `_get_patch_boundary(pid) → list[(i, j)]`

Returns active neurons in patch `pid` that have at least one dormant neighbour
— the expansion frontier.

#### `_get_gradient_at(i, j) → Tensor(D)`

Returns `sheet.weights.grad[i, j]` or zeros if gradients are not yet computed.

---

## 7. `regularizers.py`

### `intra_layer_smoothness(sheet, active_mask) → scalar Tensor`

Penalises representational dissimilarity between neighbouring active neurons in
the same patch.

**Algorithm:**
1. For each of the four axis-aligned neighbours (up/down/left/right), identify
   pairs where both neurons are active and share a patch.
2. Compute `cosine_similarity(w_a, w_b)` for each valid pair.
3. Loss = mean of `(1 − cos_sim)` across all valid neighbour pairs and directions.

Returns a scalar; multiply externally by `lambda_intra`.

### `cross_layer_coherence(sheet_L, sheet_L1, weight_matrix, num_pairs=512) → scalar Tensor`

Penalises cases where spatially close neurons in layer L have dissimilar outgoing
connections to layer L+1.

**Algorithm:**
1. Collect active neuron positions on `sheet_L`.
2. Sample `num_pairs` random pairs `(a, b)`.
3. For each pair:
   - `spatial_dist` = normalised Euclidean distance on the sheet
   - `conn_sim` = cosine similarity of their rows in `weight_matrix`
4. Loss = mean of `(1 − spatial_dist) × (1 − conn_sim)` — nearby neurons with
   divergent connections.

The pair-sampling keeps per-step cost O(`num_pairs`) regardless of layer size.
Returns a scalar; multiply externally by `lambda_cross`.

---

## 8. `layer.py`

### `class TopoSeedLayer(nn.Module)`

Drop-in replacement for `nn.Linear` and `nn.Conv2d`.

#### Construction

```python
# Linear replacement
TopoSeedLayer(
    layer_type='linear',
    in_features=784, out_features=256,
    bias=True,
    grid_size=4,           # G×G = 16 initial seeds
    warmup_steps=500,
    expansion_threshold=0.15,
    death_threshold=0.02,
    death_sustained_steps=300,
    expansion_radius=1,
    residual_weight=0.5,
    beta=0.7,
    lambda_intra=0.01,
)

# Conv replacement
TopoSeedLayer(
    layer_type='conv',
    in_channels=64, out_channels=128, kernel_size=3,
    stride=1, padding=1,
    grid_size=6,
    ...
)
```

#### `forward(x) → Tensor`

1. Calls `sheet.get_active_weight_matrix()` — returns the masked `(n_out, D)` matrix,
   autograd attached.
2. Runs `F.linear` or `F.conv2d` with the masked weights.
3. Computes and stores `intra_layer_smoothness(...) × lambda_intra` as `self._reg_loss`.
4. Forward and backward hooks fire automatically.

#### `_forward_hook` / `_backward_hook`

Registered at construction.

- **Forward hook**: captures mean absolute activation per output neuron → `_last_activation (H, W)`.
- **Backward hook**: captures L2 gradient norm per output neuron → `_last_grad_magnitude (H, W)`.

Both are stored as plain tensors (no grad) for the next `update_buffers()` call.

#### `update_buffers(patch_residuals=None)`

**Must be called after `loss.backward()` and before `optimizer.step()`.**

Feeds the captured grad/activation signals into `ExpansionManager.step_update()`.
If `patch_residuals` is `None`, computes a proxy: the fraction of total activation
*outside* each patch (how much signal the patch is not explaining).

#### `get_reg_loss() → scalar Tensor`

Returns the intra-layer smoothness loss computed during the most recent `forward()`.
Add this to the task loss before `backward()`.

#### `get_stats() → dict`

Returns:
```python
{
    "active_neuron_count":    int,
    "dormant_neuron_count":   int,
    "total_neurons":          int,
    "num_patches":            int,
    "mean_patch_size":        float,
    "mean_evidence":          float,
    "expansions_this_epoch":  int,
    "deaths_this_epoch":      int,
    "warmup_step":            int,
    "in_warmup":              bool,
}
```

#### `reset_epoch_stats()`

Resets per-epoch expansion/death counters in the underlying sheet.

#### Device handling

`_apply(fn, recurse)` is overridden so that all `.to()` / `.cuda()` / `.mps()`
calls — including when invoked transitively by a parent model — also move the
non-`nn.Module` buffer objects (`EvidenceBuffer`, `HealthBuffer`, `ResidualBuffer`)
to the correct device.  A dedicated `.mps()` method is also provided.

#### Correctness checks

| Method | What it asserts |
|---|---|
| `assert_dormant_zero(x)` | Perturbing dormant weights does not change forward output |
| `assert_seed_count()` | `active_mask.sum() == G²` at initialisation |
| `assert_no_expansion_during_warmup()` | Active count unchanged while `step < warmup_steps` |
| `assert_sheet_reshape_roundtrip(H, W, D)` *(static)* | `weights → (H,W,D) → weights` is lossless |

---

## 9. Training Loop Contract

The minimal addition to a standard training loop:

```python
# 1. Build total loss BEFORE backward (reg_loss uses graph from current forward)
task_loss = criterion(logits, labels)
total_loss = task_loss + layer.get_reg_loss()

# 2. Standard backward
total_loss.backward()

# 3. Update TopoSeed buffers — AFTER backward, BEFORE optimizer step
layer.update_buffers()   # triggers expansion/death if past warmup

# 4. Optimizer step as usual
optimizer.step()
```

Cross-layer coherence (optional, needs both sheets):

```python
from toposeed import cross_layer_coherence
cross_loss = cross_layer_coherence(
    layer1.sheet, layer2.sheet,
    layer2.sheet.get_active_weight_matrix(),
) * lambda_cross
total_loss = task_loss + layer1.get_reg_loss() + layer2.get_reg_loss() + cross_loss
```

---

## 10. Hyperparameter Guide

| Parameter | Typical range | Effect |
|---|---|---|
| `grid_size` | 3–8 | Number of initial seeds = `G²`. Smaller → more room to grow; larger → finer spatial tiling from the start. |
| `warmup_steps` | 200–1000 | Steps before any expansion/death can fire.  Scale with dataset size and layer depth. |
| `expansion_threshold` | 0.08–0.25 | Lower → more aggressive expansion.  Start at 0.15 and tune by monitoring `active_neuron_count` growth rate. |
| `death_threshold` | 0.01–0.05 | Higher → more pruning.  Keep low initially; raise if the network over-grows. |
| `death_sustained_steps` | 150–500 | How long a neuron must be dormant before it dies.  Longer = more patient. |
| `expansion_radius` | 1–2 | Manhattan radius of each expansion step.  Radius 1 = 4 neighbours max per boundary neuron. |
| `residual_weight` | 0.3–0.7 | Weight of patch residual in the expansion signal.  Higher = patches expand more when they leave signal unexplained. |
| `beta` | 0.5–0.9 | Membership reassignment threshold.  Lower = more fluid patch boundaries. |
| `lambda_intra` | 0.001–0.05 | Weight of intra-layer smoothness loss.  Increase if the sheet is fragmented; decrease if accuracy degrades. |
| `lambda_cross` | 0.001–0.01 | Weight of cross-layer coherence loss (applied externally). |
