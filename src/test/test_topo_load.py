import matplotlib
matplotlib.use("Agg")  # non-interactive backend for SLURM nodes

from huggingface_hub import snapshot_download
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import toponets
from topoloss.cortical_sheet.output import get_cortical_sheet_linear, get_cortical_sheet_conv

# Derive base experiment directory two levels up from this file
BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = BASE_DIR / "outputs" / "test_topo_load"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

# Per-family layer spec:
#   mode "single"  → find the first matching exact name, show 9 input-feature slices
#   mode "suffix"  → find ALL layers whose name ends with the suffix (one per block),
#                    show the first input-feature slice of each block in a grid
_LAYER_SPECS = {
    "resnet18": {"mode": "single",  "candidates": ["fc", "classifier"]},
    "resnet50": {"mode": "single",  "candidates": ["fc", "classifier"]},
    "vit_b_32": {"mode": "suffix",  "suffix": "mlp.3"},   # last MLP module in each transformer block
    "nanogpt":  {"mode": "suffix",  "suffix": "c_fc"},    # feed-forward c_fc in each transformer block
}


def _find_single_layer(model: nn.Module, candidates: list):
    """Return (name, layer) for the first exact-name match that is nn.Linear."""
    named = dict(model.named_modules())
    for candidate in candidates:
        layer = named.get(candidate)
        if isinstance(layer, nn.Linear):
            return candidate, layer
    # fallback: last nn.Linear
    found_name, found_layer = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            found_name, found_layer = name, m
    return found_name, found_layer


def _find_layers_by_suffix(model: nn.Module, suffix: str):
    """Return list of (name, layer) for every nn.Linear whose name ends with suffix."""
    return [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and (name == suffix or name.endswith("." + suffix))
    ]


def _get_sheet(layer: nn.Module, strict: bool = True):
    return get_cortical_sheet_linear(layer=layer, strict_layer_type=strict).cpu().detach()


def _get_conv_sheet(layer: nn.Conv2d):
    # Returns (H, W, in_channels * kH * kW) — use get_cortical_sheet_conv, not the linear variant
    return get_cortical_sheet_conv(layer=layer, strict_layer_type=True).cpu().detach()


def _find_residual_conv_layers(model: nn.Module):
    """
    Return [(name, layer)] for every Conv2d that lives inside a residual block
    (i.e. name matches layer{n}.{block}.conv{k}) — excludes the stem conv1
    and any downsample projections.
    """
    import re
    pattern = re.compile(r'^layer\d+\.\d+\.conv\d+$')
    return [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d) and pattern.match(name)
    ]


def save_conv_cortical_sheets(model: nn.Module, model_family: str, tau: float):
    """
    Visualize cortical sheets of every Conv2d in all residual blocks.
    Saves one PNG per residual stage (layer1 … layer4).
    """
    model = model.eval().cpu()
    safe_tau = str(tau).replace(".", "_")
    all_convs = _find_residual_conv_layers(model)

    if not all_convs:
        print(f"  [skip] No residual-block Conv2d layers found for {model_family} τ={tau}")
        return

    # Group by stage: layer1, layer2, …
    from collections import defaultdict
    import re
    stages = defaultdict(list)
    for name, layer in all_convs:
        stage = re.match(r'^(layer\d+)', name).group(1)
        stages[stage].append((name, layer))

    for stage, convs in sorted(stages.items()):
        n = len(convs)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        print(f"  Visualizing {n} Conv2d layers in {stage} — {model_family} τ={tau}")

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for idx, (name, layer) in enumerate(convs):
            ax = axes_flat[idx]
            try:
                sheet = _get_conv_sheet(layer)  # (H, W, in_channels * kH * kW)
                data = sheet[:, :, 0].numpy().astype(float)
                lo, hi = data.min(), data.max()
                if hi > lo:
                    data = (data - lo) / (hi - lo)
                ax.imshow(data, cmap="RdBu", vmin=0, vmax=1)
            except Exception as exc:
                ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=8,
                        transform=ax.transAxes, wrap=True, color="red")
            ax.set_title(name.replace(stage + ".", "") + f"\n{tuple(layer.weight.shape)}", fontsize=9)
            ax.axis("off")

        for idx in range(n, len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(f"{model_family}  τ={tau}  [{stage} conv layers]", fontsize=14)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{model_family}_tau{safe_tau}_{stage}_convs.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_path}")


def save_cortical_sheet(model: nn.Module, model_family: str, tau: float):
    """Visualize the paper-specified layers for a model and save a PNG."""
    model = model.eval().cpu()
    safe_tau = str(tau).replace(".", "_")
    spec = _LAYER_SPECS.get(model_family, {"mode": "single", "candidates": []})

    if spec["mode"] == "single":
        # --- ResNet-18 / ResNet-50: one layer, grid of input-feature slices ---
        layer_name, layer = _find_single_layer(model, spec["candidates"])
        if layer is None:
            print(f"  [skip] No linear layer found for {model_family} τ={tau}")
            return

        print(f"  Visualizing layer '{layer_name}' ({layer.weight.shape}) — {model_family} τ={tau}")
        try:
            sheet = _get_sheet(layer)   # (H, W, in_features)
        except Exception as exc:
            print(f"  [skip] get_cortical_sheet_linear failed: {exc}")
            return

        n_show = min(9, sheet.shape[2])
        ncols, nrows = 3, (n_show + 2) // 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        for idx, ax in enumerate(axes.flat):
            if idx < n_show:
                ax.imshow(sheet[:, :, idx].numpy(), cmap="RdBu")
                ax.set_title(f"Input feature {idx}", fontsize=11)
            ax.axis("off")
        fig.suptitle(f"{model_family}  τ={tau}  [{layer_name}]", fontsize=14)

    else:
        # --- ViT-b-32 / NanoGPT: one subplot per transformer block -----------
        layers = _find_layers_by_suffix(model, spec["suffix"])
        if not layers:
            print(f"  [skip] No layers with suffix '{spec['suffix']}' found for {model_family} τ={tau}")
            return

        n_blocks = len(layers)
        ncols = 4
        nrows = (n_blocks + ncols - 1) // ncols
        print(f"  Visualizing {n_blocks} × '{spec['suffix']}' layers — {model_family} τ={tau}")

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for idx, (name, layer) in enumerate(layers):
            ax = axes_flat[idx]
            try:
                sheet = _get_sheet(layer)   # (H, W, in_features) — show first input feature
                ax.imshow(sheet[:, :, 0].numpy(), cmap="RdBu")
            except Exception as exc:
                ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=7, wrap=True)
            block_label = name.split(".")[-2] if "." in name else name
            ax.set_title(f"block {block_label}\n{layer.weight.shape}", fontsize=9)
            ax.axis("off")

        # hide unused axes
        for idx in range(n_blocks, len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(f"{model_family}  τ={tau}  [{spec['suffix']} per block]", fontsize=14)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{model_family}_tau{safe_tau}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Download models
# ---------------------------------------------------------------------------

snapshot_download(
    repo_id="murtylab/topo-resnet50-imagenet",
    local_dir=str(BASE_DIR / "topomodels" / "topo-resnet50-imagenet")
)

snapshot_download(
    repo_id="murtylab/topo-vit-b-32-imagenet",
    local_dir=str(BASE_DIR / "topomodels" / "topo-vit-b-32-imagenet")
)

snapshot_download(
    repo_id="murtylab/topo-resnet18-imagenet",
    local_dir=str(BASE_DIR / "topomodels" / "topo-resnet18-imagenet")
)

snapshot_download(
    repo_id="murtylab/topo-nanogpt-fineweb10B",
    local_dir=str(BASE_DIR / "topomodels" / "topo-nanogpt-fineweb10B")
)

# ---------------------------------------------------------------------------
# Load models and visualize
# ---------------------------------------------------------------------------

for tau in [0.5, 1.0, 5.0, 10.0, 20.0, 50.0]:
    model = toponets.resnet18(tau=tau, checkpoint_path=str(BASE_DIR / "topomodels" / "topo-resnet18-imagenet" / f"all_topo_tau_{tau}.pt"))
    save_cortical_sheet(model, "resnet18", tau)
    save_conv_cortical_sheets(model, "resnet18", tau)

for tau in [30.0]:
    model = toponets.resnet50(tau=tau, checkpoint_path=str(BASE_DIR / "topomodels" / "topo-resnet50-imagenet" / f"all_topo_tau_{tau}.pt"))
    save_cortical_sheet(model, "resnet50", tau)
    save_conv_cortical_sheets(model, "resnet50", tau)

for tau in [10.0]:
    model = toponets.vit_b_32(tau=tau, checkpoint_path=str(BASE_DIR / "topomodels" / "topo-vit-b-32-imagenet" / f"tau_{tau}.pt"))
    save_cortical_sheet(model, "vit_b_32", tau)

for tau in [0.5, 1.0, 3.0, 50.0]:
    model = toponets.nanogpt(tau=tau, checkpoint_path=str(BASE_DIR / "topomodels" / "topo-nanogpt-fineweb10B" / f"tau_{tau}.pt"))
    save_cortical_sheet(model, "nanogpt", tau)
