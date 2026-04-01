"""Split-FashionMNIST continual learning experiment.

Five sequential binary tasks (class-incremental, single shared 10-way head):
  Task 0: T-shirt  (0) vs Trouser    (1)
  Task 1: Pullover (2) vs Dress      (3)
  Task 2: Coat     (4) vs Sandal     (5)
  Task 3: Shirt    (6) vs Sneaker    (7)
  Task 4: Bag      (8) vs AnkleBoot  (9)

The shared model (784 → 256 → 10, identical to fmnist_forgetting.py) is trained on each task
sequentially.  The optimizer is NOT reset between tasks (standard CL protocol).
After every epoch within a task, ALL tasks seen so far are re-evaluated so that forgetting
is measured live as it happens.

Five variants compared (identical architecture / losses to fmnist_forgetting.py):
  baseline         — cross-entropy only, no regularisation
  topo_only        — TopoLoss on fc1, no KL / entropy sparsity
  topo_sparsity    — TopoLoss + KL + per-sample entropy on fc1
  topo_auxk        — TopoLoss + AuxK CE loss on the full fc1 unit space
  topo_auxk_pooled — TopoLoss + AuxK CE loss on pooled cortical sheet

Key output metrics:
  acc_matrix[j][i]  — accuracy on task i's val split after training through task j  (i ≤ j)
  bwt               — Backward Transfer = mean(acc_matrix[T-1][i] − acc_matrix[i][i])  for i < T-1
  final_avg_acc     — mean accuracy over all tasks at end of all training

Results saved as:
  outputs/split_fmnist/results/split_fmnist_results_latest.json
  outputs/split_fmnist/results/split_fmnist_results_{timestamp}.json

JSON top-level keys are variant labels.
"""

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from topoloss import LaplacianPyramid, TopoLoss
from topoloss.core import find_cortical_sheet_size

# -- Import shared model / loss definitions from sibling module ----------------

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from fmnist_forgetting import (
    SimpleNN,
    CorticalRegionLock,
    EWC,
    ReplayBuffer,
    FMNIST_CLASSES,
    VARIANT_LABELS,
    DISPLAY_NAMES,
    COLORS,
    MARKERS,
    TOPO_LAYER_NAMES,
    cortical_sparsity_losses,
    cortical_similarity_loss,
    _grad_entropy,
    _activation_entropy,
    _build_topo_loss,
    _variant_config,
)

# -- Constants -----------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "split_fmnist"

# Split-FashionMNIST task definitions: 5 sequential binary classification tasks
TASKS = [
    {"classes": [0, 1], "name": "T-shirt/Trouser"},
    {"classes": [2, 3], "name": "Pullover/Dress"},
    {"classes": [4, 5], "name": "Coat/Sandal"},
    {"classes": [6, 7], "name": "Shirt/Sneaker"},
    {"classes": [8, 9], "name": "Bag/AnkleBoot"},
]

N_TASKS = len(TASKS)

# -- Data helpers --------------------------------------------------------------

def _task_subset(dataset, class_list: list) -> Subset:
    """Return a Subset of *dataset* containing only samples from *class_list*."""
    indices = [i for i, (_, lbl) in enumerate(dataset) if int(lbl) in class_list]
    return Subset(dataset, indices)


@torch.no_grad()
def task_accuracy(model: SimpleNN, loader: DataLoader, device: str) -> float:
    """Top-1 accuracy (%) on *loader* using the full 10-way argmax."""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    model.train()
    return 100.0 * correct / total if total > 0 else float("nan")


# -- Single-task training phase ------------------------------------------------

def run_task_phase(
    label: str,
    task_idx: int,
    model: SimpleNN,
    train_loader: DataLoader,
    task_val_loaders: list,      # val loaders for ALL tasks seen so far (0..task_idx)
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    device: str,
    print_freq: int,
    topo_loss=None,
    layer_cfg: dict = None,
    ckpt_dir: Path = None,
    region_locker: "CorticalRegionLock | None" = None,
    ewc: "EWC | None" = None,
    replay_buffer: "ReplayBuffer | None" = None,
) -> dict:
    """Train on one task; evaluate on all previously seen tasks each epoch.

    Returns
    -------
    dict with keys:
      ce, topo, kl, entropy, grad_entropy, auxk_aux, auxk_dead_frac : list[float] (per epoch)
      val_accs : list[list[float]]  —  val_accs[epoch][sub_idx] for sub_idx in 0..task_idx
    """
    history = {k: [] for k in (
        "ce", "topo", "kl", "entropy", "sim", "grad_entropy",
        "act_entropy",
        "auxk_aux", "auxk_dead_frac",
    )}
    history["val_accs"] = []   # list[list[float]]

    is_auxk  = model.sparsity_mode != "relu"
    best_acc = 0.0
    if is_auxk:
        model.reset_dead_counts()

    for epoch in range(epochs):
        model.train()
        sum_ce = sum_topo = 0.0
        sum_kl  = {n: 0.0 for n in TOPO_LAYER_NAMES}
        sum_ent = {n: 0.0 for n in TOPO_LAYER_NAMES}
        sum_sim = {n: 0.0 for n in TOPO_LAYER_NAMES}
        sum_grad_ent  = 0.0
        sum_act_ent   = 0.0
        sum_auxk_aux  = 0.0
        sum_auxk_dead = 0.0
        n_total = n_steps = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)
            ce     = criterion(logits, labels)

            extra = torch.zeros(1, device=device)
            if ewc is not None:
                extra = extra + ewc.penalty(model)
            if topo_loss is not None:
                topo      = topo_loss.compute(model=model, reduce_mean=True)
                sum_topo += topo.item() * imgs.size(0)
                extra     = extra + topo

                if is_auxk:
                    lc = layer_cfg[TOPO_LAYER_NAMES[0]]
                    L_aux, d_frac = model.auxk_loss(
                        labels, criterion,
                        int(lc.get("dead_threshold", 100)),
                    )
                    alpha = float(lc.get("auxk_alpha", 1 / 32))
                    extra = extra + alpha * L_aux
                    sum_auxk_aux  += L_aux.item() * imgs.size(0)
                    sum_auxk_dead += d_frac
                else:
                    act = getattr(model, "_fc1_acts", None)
                    if act is not None:
                        lc = layer_cfg["fc1"]
                        kl, ent = cortical_sparsity_losses(
                            act, lc["factor_h"], lc["factor_w"],
                            lc.get("temperature", 3.0),
                        )
                        extra = extra + lc["lambda_kl"] * kl + lc["lambda_entropy"] * ent
                        sum_kl["fc1"] += kl.item() * imgs.size(0)
                        sum_ent["fc1"] += ent.item() * imgs.size(0)
                        lambda_sim = float(lc.get("lambda_sim", 0.0))
                        if lambda_sim > 0.0:
                            sim = cortical_similarity_loss(
                                act, lc["factor_h"], lc["factor_w"]
                            )
                            extra           = extra + lambda_sim * sim
                            sum_sim["fc1"] += sim.item() * imgs.size(0)

            (ce + extra).backward()
            if region_locker is not None:
                region_locker.apply_gradient_masks(model)
            sum_grad_ent += _grad_entropy(model)
            # model._fc1_acts is set by SimpleNN.forward() — post-ReLU/TopK,
            # so this correctly reflects actual representation sparsity for
            # ALL variants (including baseline which has no hooks registered).
            sum_act_ent  += _activation_entropy(getattr(model, "_fc1_acts", None))
            optimizer.step()

            # ── Replay: train on a buffered mini-batch separately ──────────
            if replay_buffer is not None and len(replay_buffer) > 0:
                rb = replay_buffer.sample_batch(device)
                if rb is not None:
                    r_imgs, r_labels = rb
                    optimizer.zero_grad(set_to_none=True)
                    r_loss = criterion(model(r_imgs), r_labels)
                    if ewc is not None:
                        r_loss = r_loss + ewc.penalty(model)
                    r_loss.backward()
                    optimizer.step()

            bs       = imgs.size(0)
            sum_ce  += ce.item() * bs
            n_total += bs
            n_steps += 1

        # Evaluate on all tasks seen so far (0..task_idx)
        epoch_val_accs = [task_accuracy(model, vl, device) for vl in task_val_loaders]
        current_acc    = epoch_val_accs[task_idx]

        ce_avg   = sum_ce   / n_total
        topo_avg = sum_topo / n_total
        kl_avg   = sum(sum_kl[n]  for n in TOPO_LAYER_NAMES) / n_total
        ent_avg  = sum(sum_ent[n] for n in TOPO_LAYER_NAMES) / n_total
        sim_avg  = sum(sum_sim[n] for n in TOPO_LAYER_NAMES) / n_total
        ge_avg   = sum_grad_ent / max(n_steps, 1)
        ae_avg   = sum_act_ent  / max(n_steps, 1)

        auxk_aux_avg  = sum_auxk_aux  / n_total if n_total else 0.0
        auxk_dead_avg = sum_auxk_dead / max(n_steps, 1)

        history["ce"].append(ce_avg)
        history["topo"].append(topo_avg)
        history["kl"].append(kl_avg)
        history["entropy"].append(ent_avg)
        history["sim"].append(sim_avg)
        history["grad_entropy"].append(ge_avg)
        history["act_entropy"].append(ae_avg)
        history["auxk_aux"].append(auxk_aux_avg)
        history["auxk_dead_frac"].append(auxk_dead_avg)
        history["val_accs"].append(epoch_val_accs)

        # Save best checkpoint for the current task
        if ckpt_dir is not None and current_acc >= best_acc:
            best_acc = current_acc
            torch.save({
                "epoch":         epoch,
                "task_idx":      task_idx,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "best_acc":      best_acc,
                "sparsity_mode": model.sparsity_mode,
                "hidden_size":   model.hidden_size,
                "k":             model.k,
                "k_aux":         model.k_aux,
                "factor_h":      getattr(model, "factor_h", None),
                "factor_w":      getattr(model, "factor_w", None),
            }, ckpt_dir / f"best_task{task_idx}_{label}.pt")

        if (epoch + 1) % print_freq == 0 or epoch == epochs - 1:
            acc_str = "  ".join(f"T{i}={a:.1f}%" for i, a in enumerate(epoch_val_accs))
            if is_auxk:
                print(
                    f"  [{label}|T{task_idx}] Ep[{epoch+1:2d}/{epochs}]  "
                    f"CE={ce_avg:.4f}  Topo={topo_avg:.6f}  "
                    f"AuxK={auxk_aux_avg:.4f}  Dead={auxk_dead_avg:.1%}  "
                    f"GradH={ge_avg:.4f}  ActH={ae_avg:.4f}  {acc_str}"
                )
            elif topo_loss is not None:
                print(
                    f"  [{label}|T{task_idx}] Ep[{epoch+1:2d}/{epochs}]  "
                    f"CE={ce_avg:.4f}  Topo={topo_avg:.6f}  "
                    f"KL={kl_avg:.4f}  Ent={ent_avg:.4f}  Sim={sim_avg:.4f}  "
                    f"GradH={ge_avg:.4f}  ActH={ae_avg:.4f}  {acc_str}"
                )
            else:
                print(
                    f"  [{label}|T{task_idx}] Ep[{epoch+1:2d}/{epochs}]  "
                    f"CE={ce_avg:.4f}  GradH={ge_avg:.4f}  ActH={ae_avg:.4f}  {acc_str}"
                )

    # Always save the terminal checkpoint too
    if ckpt_dir is not None:
        torch.save({
            "epoch":         epochs - 1,
            "task_idx":      task_idx,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "sparsity_mode": model.sparsity_mode,
            "hidden_size":   model.hidden_size,
            "k":             model.k,
            "k_aux":         model.k_aux,
            "factor_h":      getattr(model, "factor_h", None),
            "factor_w":      getattr(model, "factor_w", None),
        }, ckpt_dir / f"last_task{task_idx}_{label}.pt")

    return history


# -- Main experiment -----------------------------------------------------------

def train(cfg: dict) -> None:
    print("=" * 70)
    print("  SPLIT-FASHIONMNIST CONTINUAL LEARNING EXPERIMENT")
    print("=" * 70)
    print(json.dumps({k: v for k, v in cfg.items() if k != "layers"}, indent=2))
    print("  layers:")
    for lname, lvals in cfg["layers"].items():
        print(f"    {lname}: {json.dumps(lvals)}")
    print()
    print("  Tasks (class-incremental, single 10-way head):")
    for i, t in enumerate(TASKS):
        c0, c1 = t["classes"]
        print(f"    Task {i}: {FMNIST_CLASSES[c0]} ({c0}) vs {FMNIST_CLASSES[c1]} ({c1})")
    print()

    # Device selection
    if torch.cuda.is_available() and str(cfg["device"]).startswith("cuda"):
        device = cfg["device"]
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")

    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    res_dir  = out_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # FashionMNIST — build per-task subsets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(cfg["data_dir"], train=True,  download=True, transform=transform)
    full_val   = datasets.FashionMNIST(cfg["data_dir"], train=False, download=True, transform=transform)
    _pin = str(device).startswith("cuda")

    task_train_loaders: list[DataLoader] = []
    task_val_loaders:   list[DataLoader] = []
    for t in TASKS:
        tr_sub = _task_subset(full_train, t["classes"])
        va_sub = _task_subset(full_val,   t["classes"])
        task_train_loaders.append(DataLoader(
            tr_sub, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2, pin_memory=_pin
        ))
        task_val_loaders.append(DataLoader(
            va_sub, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=_pin
        ))
        idx = TASKS.index(t)
        print(f"  Task {idx}: {t['name']}  — {len(tr_sub):,} train | {len(va_sub):,} val")
    print()

    layer_cfg = cfg["layers"]
    criterion = nn.CrossEntropyLoss().to(device)

    all_results: dict = {}

    for variant in VARIANT_LABELS:
        print("\n" + "=" * 70)
        print(f"  VARIANT: {DISPLAY_NAMES[variant]}")
        print("=" * 70)

        # Fresh model for each variant
        hs = cfg.get("hidden_size", 256)
        if variant == "topo_auxk_pooled":
            lc = layer_cfg["fc1"]
            model = SimpleNN(
                hidden_size=hs,
                sparsity_mode="topk_pooled",
                factor_h=lc.get("factor_h", 4.0),
                factor_w=lc.get("factor_w", 4.0),
                k=cfg.get("auxk_k_pooled", 4),
                k_aux=cfg.get("auxk_k_aux_pooled", 8),
            ).to(device)
        elif variant == "topo_auxk":
            model = SimpleNN(
                hidden_size=hs,
                sparsity_mode="topk",
                k=cfg.get("auxk_k", 32),
                k_aux=cfg.get("auxk_k_aux", 64),
            ).to(device)
        else:
            model = SimpleNN(hidden_size=hs).to(device)

        # Single optimizer shared across all tasks (standard CL protocol)
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

        # Region locker for topo_regionlock / regionlock_notopo variants
        locker = None
        if variant in ("topo_regionlock", "regionlock_notopo"):
            threshold = float(cfg.get("regionlock_threshold", 0.1))
            locker = CorticalRegionLock(hidden_size=hs, threshold=threshold)
        elif variant == "topo_regionlock_pooled":
            threshold = float(cfg.get("regionlock_threshold", 0.1))
            lc = layer_cfg["fc1"]
            locker = CorticalRegionLock(
                hidden_size=hs, threshold=threshold,
                factor_h=lc.get("factor_h", 4.0),
                factor_w=lc.get("factor_w", 4.0),
            )

        # EWC object for the ewc variant (consolidation happens after each task)
        ewc_obj = None
        if variant == "ewc":
            ewc_obj = EWC(model, lambda_ewc=float(cfg.get("ewc_lambda", 400.0)))

        # Replay buffer for the replay variant (filled incrementally after each task)
        replay_buf = None
        if variant == "replay":
            replay_buf = ReplayBuffer(
                capacity=int(cfg.get("replay_capacity", 500)),
                replay_batch_size=int(cfg.get("replay_batch_size", 32)),
            )

        # acc_matrix[j][i] = accuracy on task i after training through task j  (i ≤ j)
        acc_matrix: list[list[float]] = []
        per_task_histories: list[dict] = []

        for task_idx in range(N_TASKS):
            task_info = TASKS[task_idx]
            print(f"\n  Task {task_idx}: {task_info['name']}  ({cfg['task_epochs']} epochs)")
            if locker is not None:
                print(f"    RegionLock: {locker.fraction_locked:.1%} of units locked")
            if replay_buf is not None:
                print(f"    Replay buffer: {len(replay_buf)} samples")

            # Rebuild topo_loss each task so it stays attached to the current model
            topo_loss, eff_layer_cfg = _variant_config(model, variant, layer_cfg)

            history = run_task_phase(
                label=variant,
                task_idx=task_idx,
                model=model,
                train_loader=task_train_loaders[task_idx],
                task_val_loaders=task_val_loaders[:task_idx + 1],
                criterion=criterion,
                optimizer=optimizer,
                epochs=cfg["task_epochs"],
                device=device,
                print_freq=cfg["print_freq"],
                topo_loss=topo_loss,
                layer_cfg=eff_layer_cfg,
                ckpt_dir=ckpt_dir,
                region_locker=locker,
                ewc=ewc_obj,
                replay_buffer=replay_buf,
            )

            per_task_histories.append(history)

            # Post-task updates ────────────────────────────────────────────
            # Lock regions activated by this task (before moving to next task)
            if locker is not None:
                locker.update_masks(model, task_train_loaders[task_idx], device)
                print(f"    RegionLock: {locker.fraction_locked:.1%} of units now locked")
            # EWC: consolidate Fisher on current task's data
            if ewc_obj is not None:
                ewc_obj.consolidate(
                    model, task_train_loaders[task_idx], criterion, device,
                    n_samples=int(cfg.get("ewc_n_samples", 2048)),
                )
                print(f"    EWC: Fisher consolidated after task {task_idx}")
            # Replay: add current task data to the buffer
            if replay_buf is not None:
                replay_buf.add_from_loader(task_train_loaders[task_idx])
                print(f"    Replay buffer: {len(replay_buf)} samples stored")

            # Final-epoch accuracy on every task seen so far
            final_row = history["val_accs"][-1]   # list of length task_idx+1
            acc_matrix.append(final_row)

            acc_str = "  ".join(f"T{i}={a:.1f}%" for i, a in enumerate(final_row))
            print(f"\n  → After task {task_idx}: {acc_str}")

        # Backward Transfer: BWT = mean(acc_matrix[T-1][i] - acc_matrix[i][i])  for i < T-1
        T   = N_TASKS
        bwt = float(np.mean([
            acc_matrix[T - 1][i] - acc_matrix[i][i]
            for i in range(T - 1)
        ]))
        # Intransigence: how well does the model learn each task right after training it
        learning_acc = float(np.mean([acc_matrix[i][i] for i in range(T)]))
        final_avg_acc = float(np.mean(acc_matrix[T - 1]))

        print(f"\n  BWT = {bwt:+.2f}pp   Learning acc = {learning_acc:.1f}%   Final avg acc = {final_avg_acc:.1f}%")

        all_results[variant] = {
            "tasks":              [{"classes": t["classes"], "name": t["name"]} for t in TASKS],
            # acc_matrix[j][i] — outer list over j (which task was just trained),
            # inner list over i (which task is evaluated), length = j+1
            "acc_matrix":         acc_matrix,
            "bwt":                bwt,
            "learning_acc":       learning_acc,
            "final_avg_acc":      final_avg_acc,
            # per_task_histories[j] contains training history for task j:
            #   val_accs[epoch][sub_idx] — sub_idx in 0..j
            "per_task_histories": per_task_histories,
            "config":             {k: v for k, v in cfg.items() if k != "layers"},
        }

    # -- Save results ----------------------------------------------------------
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path   = res_dir / f"split_fmnist_results_{ts}.json"
    last_path = res_dir / "split_fmnist_results_latest.json"

    for path in (ts_path, last_path):
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {last_path}")
    print("  Run analyze_split_fmnist.py to generate figures.")

    print("\nContinual learning summary:")
    print(f"  {'Variant':<22}  {'BWT':>8}   {'Learn Acc':>10}   {'Final Acc':>10}")
    print("  " + "-" * 58)
    for v, r in all_results.items():
        print(
            f"  {DISPLAY_NAMES[v]:<22}  {r['bwt']:>+7.2f}pp"
            f"   {r['learning_acc']:>9.1f}%   {r['final_avg_acc']:>9.1f}%"
        )


# -- Config & entry point ------------------------------------------------------

DEFAULT_CFG = {
    "data_dir":           None,
    "output_dir":         None,
    "hidden_size":        256,
    "task_epochs":        10,
    "batch_size":         128,
    "lr":                 5e-4,
    "device":             "cuda:0",
    "print_freq":         2,
    "auxk_k":             16,
    "auxk_k_aux":         64,
    "auxk_k_pooled":      1,
    "auxk_k_aux_pooled":  4,
    "regionlock_threshold": 0.1,
    "ewc_lambda":         400.0,
    "ewc_n_samples":      2048,
    "replay_capacity":    500,
    "replay_batch_size":  32,
    "layers": {
        "fc1": {
            "topo_scale":    10.0,
            "factor_h":       4.0,
            "factor_w":       4.0,
            "lambda_kl":      0.35,
            "lambda_entropy": 2.0,
            "lambda_sim":     0.5,
            "temperature":    1.0,
            "auxk_alpha":     0.5,
            "dead_threshold": 8,
        }
    },
}


def _load_json(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def get_config() -> dict:
    p = argparse.ArgumentParser(
        description="Split-FashionMNIST continual learning experiment"
    )
    p.add_argument("--config",              default=None)
    p.add_argument("--data-dir",            default=None)
    p.add_argument("--output-dir",          default=None)
    p.add_argument("--task-epochs",         type=int,   default=None)
    p.add_argument("--batch-size",          type=int,   default=None)
    p.add_argument("--lr",                  type=float, default=None)
    p.add_argument("--device",              default=None)
    p.add_argument("--print-freq",          type=int,   default=None)
    p.add_argument("--auxk-k",              type=int,   default=None)
    p.add_argument("--auxk-k-aux",          type=int,   default=None)
    p.add_argument("--auxk-k-pooled",       type=int,   default=None)
    p.add_argument("--auxk-k-aux-pooled",   type=int,   default=None)
    cli = p.parse_args()

    cfg = copy.deepcopy(DEFAULT_CFG)

    # JSON config override
    default_json = str(BASE_DIR / "configs" / "split_fmnist.json")
    config_path  = cli.config or (default_json if Path(default_json).exists() else None)
    if config_path:
        cfg.update(_load_json(config_path))

    # CLI overrides (highest priority)
    for cli_attr, cfg_key in [
        ("data_dir",           "data_dir"),
        ("output_dir",         "output_dir"),
        ("task_epochs",        "task_epochs"),
        ("batch_size",         "batch_size"),
        ("lr",                 "lr"),
        ("device",             "device"),
        ("print_freq",         "print_freq"),
        ("auxk_k",             "auxk_k"),
        ("auxk_k_aux",         "auxk_k_aux"),
        ("auxk_k_pooled",      "auxk_k_pooled"),
        ("auxk_k_aux_pooled",  "auxk_k_aux_pooled"),
    ]:
        val = getattr(cli, cli_attr, None)
        if val is not None:
            cfg[cfg_key] = val

    # Resolve default paths
    if not cfg.get("data_dir"):
        cfg["data_dir"]   = str(BASE_DIR / "data")
    if not cfg.get("output_dir"):
        cfg["output_dir"] = str(OUTPUT_DIR)

    return cfg


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
