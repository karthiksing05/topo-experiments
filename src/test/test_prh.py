"""
I would like to see if we can plot manifolds along each cortical sheet with a corpus of training data
that allows us to map unseen data to points on the cortical sheet that are synonymous across representations.

The idea is that given a corpus of training data, we create a manifold / surface for both modalities,
and we see if the Euclidean distances of unseen data with respect to a given waypoint are 

TODO This involves topography in the representation so will hold on this for now!!

Maybe print and test representations!!
============
Platonic Representation Hypothesis probe.

Loads a handful of COCO (Karpathy test) examples, runs them through each
loaded `toponets` model, and saves cortical-sheet visualisations of the
*activations* of the paper-specified final layer for each model / τ pair.

Vision models  → `fc` (ResNets) / `heads.head` (ViT-b-32)  – fed images
Language model → `lm_head` (NanoGPT)                        – fed captions

Outputs: outputs/test_prh/
  <model>_tau<tau>_activations.png   one column per sample
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import toponets
from topoloss.core import find_cortical_sheet_size
from huggingface_hub import snapshot_download

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "test_prh"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE   = BASE_DIR / "huggingfacehub_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE_DIR / "topomodels"

N_SAMPLES = 5   # Flickr30k examples to visualise per model

# ── Paper-specified final layer per model family ───────────────────────────────

_FINAL_LAYER = {
    "resnet18": "fc",
    "resnet50": "fc",
    "vit_b_32": "heads.head",
    "nanogpt":  "lm_head",
}
_INPUT_MODALITY = {
    "resnet18": "image",
    "resnet50": "image",
    "vit_b_32": "image",
    "nanogpt":  "text",
}

# ── Dataset ───────────────────────────────────────────────────────────────────

def load_flickr30k_samples(n: int = N_SAMPLES):
    """Return n (PIL image, captions-list) pairs from Flickr30k test split."""
    from datasets import load_dataset

    print(f"\nLoading {n} Flickr30k samples (test split) ...")
    ds = load_dataset(
        "nlphuji/flickr30k",
        split="test",
        cache_dir=str(HF_CACHE / "datasets"),
        trust_remote_code=True,
    )
    images, captions = [], []
    for row in ds:
        im = row.get("image")
        if im is None:
            continue
        if not isinstance(im, Image.Image):
            try:
                im = Image.open(im).convert("RGB")
            except Exception:
                continue
        else:
            im = im.convert("RGB")
        caps = row.get("caption") or row.get("captions") or row.get("sentences") or []
        if isinstance(caps, str):
            caps = [caps]
        if not caps:
            continue
        images.append(im)
        captions.append([str(c) for c in caps][:5])
        if len(images) >= n:
            break
    print(f"  Loaded {len(images)} samples.")
    return images, captions

# ── Transforms ────────────────────────────────────────────────────────────────

def _vision_transform(image_size: int = 224):
    from torchvision.transforms import (
        CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor,
    )
    return Compose([
        Resize(256, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _tokenize_captions(captions: list, max_len: int = 64) -> torch.Tensor:
    """Tokenise with tiktoken (gpt-2 BPE) to a fixed-length (n, max_len) tensor."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    rows = []
    for cap in captions:
        ids = enc.encode(cap)[:max_len]
        ids += [0] * (max_len - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)

# ── Activation hook ───────────────────────────────────────────────────────────

class _Hook:
    def __init__(self, module: nn.Module):
        self._h = module.register_forward_hook(self._fn)
        self.output: torch.Tensor | None = None

    def _fn(self, _mod, _inp, output):
        self.output = output.detach().cpu()

    def remove(self):
        self._h.remove()


def _find_layer(model: nn.Module, name: str) -> nn.Module | None:
    for n, m in model.named_modules():
        if n == name:
            return m
    return None


def _act_to_sheet(act: torch.Tensor) -> torch.Tensor:
    """Collapse to 1-D, find cortical-sheet H×W, normalise to [0,1], return (H, W) tensor."""
    act = act.float()
    if act.dim() > 1:
        act = act.mean(dim=0).flatten()
    else:
        act = act.flatten()
    # Normalise so the colormap always has meaningful contrast
    lo, hi = act.min(), act.max()
    if hi > lo:
        act = (act - lo) / (hi - lo)
    size = find_cortical_sheet_size(act.numel())
    return act.reshape(size.height, size.width)

# ── Main visualiser ───────────────────────────────────────────────────────────

def visualize_activations(
    model: nn.Module,
    model_family: str,
    tau: float,
    images: list,
    captions: list,
):
    """
    For each COCO sample run the model, capture the final-layer activations,
    reshape to the cortical-sheet grid, and save a figure.

    Layout (one column per sample):
      Row 0 – reference  (thumbnail thumbnail for images, wrapped caption for text)
      Row 1 – activation cortical sheet (RdBu heatmap)
    """
    layer_name = _FINAL_LAYER[model_family]
    modality   = _INPUT_MODALITY[model_family]

    layer = _find_layer(model, layer_name)
    if layer is None:
        print(f"  [skip] layer '{layer_name}' not found in {model_family}")
        return

    hook = _Hook(layer)
    model.eval()
    transform = _vision_transform()
    safe_tau  = str(tau).replace(".", "_")

    sheets = []
    refs   = []   # PIL thumbnails or caption strings

    with torch.no_grad():
        if modality == "image":
            for img, caps in zip(images, captions):
                x = transform(img).unsqueeze(0)
                try:
                    model(x)
                except Exception as e:
                    print(f"  [warn] forward error ({model_family} τ={tau}): {e}")
                if hook.output is None:
                    print(f"  [warn] hook did not fire for {model_family} τ={tau} — layer '{layer_name}'")
                    continue
                sheets.append(_act_to_sheet(hook.output[0]))
                thumb = img.copy()
                thumb.thumbnail((112, 112))
                refs.append(thumb)
                hook.output = None  # reset

        else:  # text
            first_caps = [c[0] if c else "" for c in captions]
            tokens     = _tokenize_captions(first_caps)
            for i, (tok, caps) in enumerate(zip(tokens, captions)):
                try:
                    model(tok.unsqueeze(0))
                except Exception as e:
                    print(f"  [warn] forward error ({model_family} τ={tau}): {e}")
                if hook.output is None:
                    print(f"  [warn] hook did not fire for {model_family} τ={tau} — layer '{layer_name}'")
                    continue
                sheets.append(_act_to_sheet(hook.output[0]))
                refs.append(caps[0] if caps else f"sample {i}")
                hook.output = None

    hook.remove()

    if not sheets:
        print(f"  [skip] no activations captured for {model_family} τ={tau}")
        return

    n = len(sheets)
    fig = plt.figure(figsize=(3.5 * n, 7))
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.05, wspace=0.05)

    for col, (ref, sheet) in enumerate(zip(refs, sheets)):
        # Row 0 – reference
        ax_ref = fig.add_subplot(gs[0, col])
        if isinstance(ref, Image.Image):
            ax_ref.imshow(np.array(ref))
        else:
            ax_ref.text(
                0.5, 0.5, ref[:120],
                ha="center", va="center", fontsize=6,
                wrap=True, transform=ax_ref.transAxes,
            )
            ax_ref.set_facecolor("#f5f5f5")
        ax_ref.axis("off")

        # Row 1 – cortical-sheet activation
        ax_act = fig.add_subplot(gs[1, col])
        ax_act.imshow(sheet.numpy(), cmap="RdBu")
        ax_act.axis("off")

    fig.suptitle(
        f"{model_family}  τ={tau}  |  {layer_name}  activations  (COCO samples)",
        fontsize=12,
    )
    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu")
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, shrink=0.4, pad=0.01)

    plt.tight_layout(rect=[0, 0, 0.96, 0.95])
    out_path = OUTPUT_DIR / f"{model_family}_tau{safe_tau}_activations.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")

# ── Download models (idempotent) ──────────────────────────────────────────────

for repo_id, subdir in [
    ("murtylab/topo-resnet18-imagenet",    "topo-resnet18-imagenet"),
    ("murtylab/topo-resnet50-imagenet",    "topo-resnet50-imagenet"),
    ("murtylab/topo-vit-b-32-imagenet",    "topo-vit-b-32-imagenet"),
    ("murtylab/topo-nanogpt-fineweb10B",   "topo-nanogpt-fineweb10B"),
]:
    snapshot_download(repo_id=repo_id, local_dir=str(MODELS_DIR / subdir))

# ── Load Flickr30k samples once ──────────────────────────────────────────────

coco_images, coco_captions = load_flickr30k_samples(N_SAMPLES)

# ── Load models and visualise ─────────────────────────────────────────────────

print("\n=== ResNet-18 ===")
for tau in [0.5, 1.0, 5.0, 10.0, 20.0, 50.0]:
    model = toponets.resnet18(
        tau=tau,
        checkpoint_path=str(MODELS_DIR / "topo-resnet18-imagenet" / f"all_topo_tau_{tau}.pt"),
    )
    visualize_activations(model, "resnet18", tau, coco_images, coco_captions)

print("\n=== ResNet-50 ===")
for tau in [30.0]:
    model = toponets.resnet50(
        tau=tau,
        checkpoint_path=str(MODELS_DIR / "topo-resnet50-imagenet" / f"all_topo_tau_{tau}.pt"),
    )
    visualize_activations(model, "resnet50", tau, coco_images, coco_captions)

print("\n=== ViT-b-32 ===")
for tau in [10.0]:
    model = toponets.vit_b_32(
        tau=tau,
        checkpoint_path=str(MODELS_DIR / "topo-vit-b-32-imagenet" / f"tau_{tau}.pt"),
    )
    visualize_activations(model, "vit_b_32", tau, coco_images, coco_captions)

print("\n=== NanoGPT ===")
for tau in [0.5, 1.0, 3.0, 50.0]:
    model = toponets.nanogpt(
        tau=tau,
        checkpoint_path=str(MODELS_DIR / "topo-nanogpt-fineweb10B" / f"tau_{tau}.pt"),
    )
    visualize_activations(model, "nanogpt", tau, coco_images, coco_captions)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")