"""
examples/resnet_cifar10.py

ResNet-18 with TopoSeedLayer conv replacements on CIFAR-10.

Run:
    python examples/resnet_cifar10.py [--epochs 50] [--batch-size 128]
                                       [--lr 0.1] [--lambda-reg 0.005]
                                       [--viz-every 10] [--viz-dir outputs/viz_resnet]
                                       [--no-topo] [--device auto]
"""

from __future__ import annotations

import argparse
import sys
import os
import time

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as T

from toposeed import TopoSeedLayer, get_device
from toposeed import plot_all, plot_multi_layer_selectivity

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird",  "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def replace_conv(module: nn.Module, depth: int = 0) -> None:
    """
    Recursively walk `module` and replace every nn.Conv2d with TopoSeedLayer.
    Skips the very first conv layer (stem conv) to keep input processing fast.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            # Skip depthwise / groups != 1 (ResNet-18 has none, but guard anyway)
            if child.groups != 1:
                continue
            grid_size = max(3, child.out_channels // 16)
            topo_layer = TopoSeedLayer(
                layer_type="conv",
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size[0],
                stride=child.stride[0],
                padding=child.padding[0],
                bias=(child.bias is not None),
                grid_size=grid_size,
                warmup_steps=500 + depth * 200,   # deeper = longer warmup
                expansion_threshold=0.15,
                death_threshold=0.02,
                death_sustained_steps=400,
                beta=0.7,
                lambda_intra=0.01,
            )
            setattr(module, name, topo_layer)
        else:
            replace_conv(child, depth + 1)


def make_topo_resnet18() -> nn.Module:
    model = models.resnet18(weights=None, num_classes=10)
    # Adapt for CIFAR-10: smaller spatial input (32×32)
    # Replace 7×7 stem with 3×3 conv and remove early maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # Now replace conv layers with TopoSeedLayer
    replace_conv(model)
    return model


def get_topo_layers(model: nn.Module) -> list[TopoSeedLayer]:
    return [m for m in model.modules() if isinstance(m, TopoSeedLayer)]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def summarise_topo_layers(topo_layers: list[TopoSeedLayer]) -> None:
    print(f"\n{'Layer':>6} | {'n_out':>6} | {'active':>8} | {'dormant':>8} | "
          f"{'patches':>8} | {'expand':>8} | {'deaths':>8}")
    print("-" * 72)
    for idx, layer in enumerate(topo_layers):
        s = layer.get_stats()
        print(f"  {idx:>4} | {layer.n_out:>6} | {s['active_neuron_count']:>8} | "
              f"{s['dormant_neuron_count']:>8} | {s['num_patches']:>8} | "
              f"{s['expansions_this_epoch']:>8} | {s['deaths_this_epoch']:>8}")
    print()


def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print(f"Device: {device}")

    # ---- Data ---------------------------------------------------------------
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        norm,
    ])
    test_tf = T.Compose([T.ToTensor(), norm])

    train_ds = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_tf
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_tf
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=pin)

    # ---- Model --------------------------------------------------------------
    if args.no_topo:
        model: nn.Module = models.resnet18(weights=None, num_classes=10)
        # CIFAR adaptation
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model = model.to(device)
        print("Running BASELINE ResNet-18 (no TopoSeed)")
        topo_layers: list[TopoSeedLayer] = []
    else:
        model = make_topo_resnet18().to(device)
        topo_layers = get_topo_layers(model)
        print(f"Running TOPO ResNet-18 ({len(topo_layers)} TopoSeedLayer conv replacements)")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    total_steps = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for layer in topo_layers:
            layer.reset_epoch_stats()

        running_loss = 0.0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            task_loss = criterion(logits, labels)

            if topo_layers:
                reg_loss = sum(l.get_reg_loss() for l in topo_layers) * args.lambda_reg
                total_loss = task_loss + reg_loss
            else:
                total_loss = task_loss

            total_loss.backward()
            optimizer.step()

            # Update TopoSeed buffers after backward
            for layer in topo_layers:
                layer.update_buffers()

            running_loss += task_loss.item()
            total_steps += 1

        scheduler.step()

        elapsed = time.time() - t0
        train_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, test_loader, device)
        best_val_acc = max(best_val_acc, val_acc)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
            f"best={best_val_acc:.4f} | lr={lr_now:.5f} | "
            f"time={elapsed:.1f}s"
        )

        if topo_layers and epoch % 5 == 0:
            summarise_topo_layers(topo_layers)

        # Periodic visualisations (a representative subset of layers)
        if topo_layers and args.viz_every > 0 and epoch % args.viz_every == 0:
            import pathlib
            viz_dir = pathlib.Path(args.viz_dir)
            # Pick first, middle, last topo layers for periodic snapshots
            n = len(topo_layers)
            snap_layers = {0: topo_layers[0],
                           n // 2: topo_layers[n // 2],
                           n - 1: topo_layers[-1]}
            for idx, layer in snap_layers.items():
                plot_all(
                    model=model,
                    layer=layer,
                    layer_name=f"conv{idx}",
                    dataloader=test_loader,
                    class_names=CIFAR10_CLASSES,
                    device=device,
                    out_dir=viz_dir,
                    epoch=epoch,
                )

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")

    # Final summary table
    if topo_layers:
        print("\n=== Final Active Neuron Summary ===")
        summarise_topo_layers(topo_layers)

        # ---- Final visualisations -------------------------------------------
        import pathlib
        viz_dir = pathlib.Path(args.viz_dir)
        print(f"\nSaving final visualisations to {viz_dir} ...")
        n = len(topo_layers)
        # Visualise every layer individually
        for idx, layer in enumerate(topo_layers):
            plot_all(
                model=model,
                layer=layer,
                layer_name=f"conv{idx}",
                dataloader=test_loader,
                class_names=CIFAR10_CLASSES,
                device=device,
                out_dir=viz_dir,
                epoch=None,
                prefix="final_",
            )
        # Multi-layer selectivity: first, middle, last
        sel_indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
        plot_multi_layer_selectivity(
            model=model,
            topo_layers=[topo_layers[i] for i in sel_indices],
            layer_names=[f"conv{i}" for i in sel_indices],
            out_path=viz_dir / "final_multi_layer_selectivity.png",
            dataloader=test_loader,
            class_names=CIFAR10_CLASSES,
            device=device,
            title="CIFAR-10 — Multi-Layer Selectivity",
        )
        print("  Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TopoSeed ResNet-18 on CIFAR-10")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lambda-reg", type=float, default=0.005,
                   help="Scale factor for summed TopoSeed regularisation loss")
    p.add_argument("--viz-every", type=int, default=10,
                   help="Save visualisations every N epochs (0 = only at end)")
    p.add_argument("--viz-dir", type=str, default="outputs/toposeed/viz_resnet",
                   help="Output directory for visualisation PNGs")
    p.add_argument("--no-topo", action="store_true",
                   help="Run baseline ResNet-18 without TopoSeed")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: cuda | mps | cpu | auto (default: auto)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
