#!/usr/bin/env python3
"""run_4q.py — Training of 1 seed of the 4-qubit QCNN model (Wave-K cross-arch).

Architecture:
  Classical backbone (LeNet-style) -> QuantumConvLayer 4q (K=4 parallel blocks)
  -> FC head -> softmax. Identical to the 9q noiseless multi-seed run EXCEPT for
  num_qubits=4 and kernel_size=2 (and stride=3 for consistency with Emerald v4.3.0).

Backend selectable via CLI:
  --backend sim_noisy     → AerSimulator density_matrix + ibm_fez restricted (qubit 0-3)
  --backend sim_noiseless → AerSimulator statevector (no noise)

In both cases the quantum block uses the same sampler-based QuantumEnginePSSim
with shots=500 (hardware-faithful, mirroring Emerald). Only the backend
behind the sampler differs.

Output: JSON with history, predictions, final metrics, in a format compatible
with wilcoxon_cross_arch.py.

CLI usage:
  python run_4q.py \
      --seed 42 \
      --backend sim_noisy \
      --noise-snapshot /path/to/ibm_fez_20260516.pkl \
      --train-dir /path/to/dataset/training \
      --val-dir   /path/to/dataset/validation \
      --output-dir ./results_4q_noisy

The same script is imported from the local notebook for the noiseless run.
"""

from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# ENV — MUST precede import numpy/torch (see parallel_quanv_sim.py)
# ────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from qiskit_aer import AerSimulator

# parallel_quanv_sim.py must be accessible (same directory or PYTHONPATH)
from parallel_quanv_sim import (
    ParallelQuanvCircuit,
    QuantumEnginePSSim,
    BackendManagerSimNoisy,
    AerSamplerWrapperFixed,
)


# ============================================================================
# Config (allineato al multiseed 9q noiseless per pairing Wilcoxon)
# ============================================================================

@dataclass
class Config4q:
    # Pairing
    seed: int = 42

    # Quantum architecture (4q, diverso dal 9q-noiseless)
    num_qubits: int = 4
    kernel_size: int = 2              # 2×2 quantum kernel (era 3×3 per 9q)
    stride: int = 3                   # 16→5 quanv output, coerente con Emerald v4.3.0
    quanv_padding: int = 0
    num_parallel_blocks: int = 4      # K=4 parallel blocks (independent sub-PUBs)
    measure_qubit: int = 0
    shots: int = 500                  # hardware-faithful, mirroring Emerald
    max_bindings_per_job: int = 500
    optimization_level: int = 1

    # Classical architecture (IDENTICO al multiseed 9q)
    num_conv_channels: int = 6
    conv_kernel_size: int = 5
    conv_padding: int = 2
    dropout_rate: float = 0.0

    # Dataset (IDENTICO al multiseed 9q)
    img_size: int = 64
    in_channels: int = 3
    num_classes: int = 2
    selected_classes: list = field(
        default_factory=lambda: ['AnnualCrop', 'Forest'])
    max_samples_per_class: int = 100

    # Training (IDENTICO al multiseed 9q)
    batch_size: int = 16
    max_epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 1e-4
    num_workers: int = 0   # 0 per evitare overhead spawn su small dataset

    # Backend
    backend_type: str = "sim_noisy"   # "sim_noisy" | "sim_noiseless"
    noise_snapshot_path: Optional[str] = None
    noise_qubits: list = field(default_factory=lambda: [0, 1, 2, 3])
    aer_max_parallel_experiments: int = 8

    # I/O
    train_dir: str = "./dataset/training"
    val_dir: str = "./dataset/validation"
    output_dir: str = "./results_4q"

    @property
    def num_weights(self) -> int:
        return self.num_qubits

    @property
    def total_qubits(self) -> int:
        return self.num_qubits * self.num_parallel_blocks

    @property
    def feature_map_size(self) -> int:
        return self.img_size // 4

    @property
    def quanv_output_size(self) -> int:
        return ((self.feature_map_size + 2 * self.quanv_padding - self.kernel_size)
                // self.stride + 1)

    @property
    def patches_per_channel(self) -> int:
        return self.quanv_output_size ** 2

    @property
    def flatten_size(self) -> int:
        return self.num_conv_channels * self.patches_per_channel


# ============================================================================
# Backend manager minimo per modalità noiseless (riusa interfaccia parallel_quanv_sim)
# ============================================================================

class BackendManagerSimNoiseless:
    """Drop-in compatible con BackendManagerSimNoisy ma senza NoiseModel.

    Uses AerSimulator(method='statevector'): identical in API to the
    sampler-based noisy but substantially faster (for the local Mac run).
    L'engine QuantumEnginePSSim non vede differenze: ottiene un sampler
    che produce counts via shots.
    """
    def __init__(self, config, aer_max_parallel_experiments: int = 8):
        self.config = config
        self.aer_parallel = aer_max_parallel_experiments
        self.backend = None
        self.sampler = None
        self.backend_name = "AerSimulator(statevector, noiseless)"
        self.noise_model = None
        self.noise_model_full = None
        self.num_backend_qubits = 0
        self.rng = np.random.default_rng(config.seed)

    def initialize(self):
        self.backend = AerSimulator(
            method='statevector',
            max_parallel_experiments=self.aer_parallel,
            max_parallel_threads=self.aer_parallel,
        )
        self.num_backend_qubits = max(64, self.config.total_qubits)
        self.sampler = AerSamplerWrapperFixed(self.backend, self.config.shots)
        print(f"  {self.backend_name} | K={self.config.num_parallel_blocks}")

    def transpile_circuit(self, circuit):
        # Niente NoiseModel → niente vincoli di basis. Lascio passare tale e quale.
        # Aer accetta ry, h, cx nativamente in statevector method.
        return circuit

    def close(self):
        pass


# ============================================================================
# Quantum conv layer (autograd Function + nn.Module wrapper)
# ============================================================================

class QuantumConvFunctionPSSim(torch.autograd.Function):
    """Adapter tra QuantumEnginePSSim (numpy) e PyTorch autograd.

    Pattern identical to QuantumConvFunction of the 9q multi-seed run, but uses
    engine.forward_only() + engine.step() instead of engine.forward() +
    engine.forward_and_gradient().
    """

    @staticmethod
    def forward(ctx, input_patches: torch.Tensor, weights: torch.Tensor,
                engine: QuantumEnginePSSim):
        in_np = input_patches.detach().cpu().numpy()
        w_np = weights.detach().cpu().numpy()
        out = engine.forward_only(in_np, w_np)  # shape (N, 1)
        ctx.save_for_backward(input_patches, weights)
        ctx.engine = engine
        return torch.tensor(out, dtype=torch.float32,
                            device=input_patches.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        in_t, w_t = ctx.saved_tensors
        engine: QuantumEnginePSSim = ctx.engine
        device = grad_output.device

        in_np = in_t.detach().cpu().numpy()
        w_np = w_t.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()  # (N, 1)

        # Skip input grads se l'upstream ha applicato .detach() (multiseed pattern)
        skip_input = not in_t.requires_grad

        _, gw_jac, gx_jac = engine.step(in_np, w_np, compute_gx=not skip_input)
        # gw_jac: (nw, N, 1), gx_jac: (n, N, 1) o None

        # Chain rule weights: sum su N e su observable
        grad_w = np.einsum('jnq,nq->j', gw_jac, grad_out_np)

        if skip_input or gx_jac is None:
            grad_in = None
        else:
            grad_in_np = np.einsum('inq,nq->ni', gx_jac, grad_out_np)
            grad_in = torch.tensor(grad_in_np, dtype=torch.float32, device=device)

        return (grad_in,
                torch.tensor(grad_w, dtype=torch.float32, device=device),
                None)


class QuantumConvLayer(nn.Module):
    """4q quantum conv: estrae patch ks×ks per canale, applica QuantumEngine.

    Channel batching: all channels concatenated into a single engine call
    (B*C*P bindings in total). Exact replica of the logic of the 9q multiseed run
    (cell 16) module n=4 instead of n=9.
    """

    def __init__(self, config: Config4q, engine: QuantumEnginePSSim,
                 rng: np.random.Generator):
        super().__init__()
        self.config = config
        self.n = config.num_qubits
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.padding = config.quanv_padding
        self.engine = engine

        # Quantum weights (4 RZ, shared across all channels and all blocks)
        init_w = (rng.random(config.num_weights) * 2 - 1) * 0.3
        self.quantum_weights = nn.Parameter(
            torch.tensor(init_w, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H_out, W_out)."""
        B, C, H, W = x.shape

        if self.padding > 0:
            x_pad = F.pad(x, [self.padding] * 4, mode='constant', value=0.0)
        else:
            x_pad = x
        _, _, Hp, Wp = x_pad.shape

        H_out = (Hp - self.kernel_size) // self.stride + 1
        W_out = (Wp - self.kernel_size) // self.stride + 1
        P = H_out * W_out

        all_patches = []
        for c in range(C):
            x_c = x_pad[:, c:c+1, :, :]
            patches = F.unfold(x_c, kernel_size=self.kernel_size,
                               stride=self.stride)
            # Normalizzazione → [0, π] per encoding RY ottimale
            p_min = patches.min(dim=2, keepdim=True).values
            p_max = patches.max(dim=2, keepdim=True).values
            p_range = (p_max - p_min).clamp(min=1e-8)
            patches_scaled = (patches - p_min) / p_range * np.pi
            all_patches.append(patches_scaled.permute(0, 2, 1).contiguous())

        all_patches = torch.cat(all_patches, dim=1)   # (B, C*P, n)
        all_flat = all_patches.reshape(-1, self.n)    # (B*C*P, n)

        # Detach: skip input gradients (ottimizzazione Wave K E del multiseed)
        all_flat = all_flat.detach()

        q_out = QuantumConvFunctionPSSim.apply(
            all_flat, self.quantum_weights, self.engine)
        # (B*C*P, 1)

        q_out = q_out.reshape(B, C, P).reshape(B, C, H_out, W_out)
        return q_out


# ============================================================================
# Classical layers + hybrid model
# ============================================================================

class Backbone(nn.Module):
    """LeNet-style backbone, IDENTICO al multiseed 9q."""
    def __init__(self, c: Config4q):
        super().__init__()
        ch, ks, p = c.num_conv_channels, c.conv_kernel_size, c.conv_padding
        self.c1 = nn.Sequential(
            nn.Conv2d(c.in_channels, ch, ks, padding=p),
            nn.BatchNorm2d(ch), nn.ReLU(), nn.MaxPool2d(2))
        self.c2 = nn.Sequential(
            nn.Conv2d(ch, ch, ks, padding=p),
            nn.BatchNorm2d(ch), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        return self.c2(self.c1(x))


class FC(nn.Module):
    """FC head, IDENTICO al multiseed 9q (modulo dimensione input)."""
    def __init__(self, c: Config4q):
        super().__init__()
        f = c.flatten_size
        f1 = max(f // 2, 32)
        layers = [nn.Linear(f, f1), nn.ReLU()]
        if c.dropout_rate > 0:
            layers.append(nn.Dropout(c.dropout_rate))
        layers.append(nn.Linear(f1, c.num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class HybridQCNN(nn.Module):
    """Backbone → Quanv → FC."""
    def __init__(self, config: Config4q, engine: QuantumEnginePSSim,
                 rng: np.random.Generator):
        super().__init__()
        self.backbone = Backbone(config)
        self.quanv = QuantumConvLayer(config, engine, rng)
        self.fc = FC(config)

    def forward(self, x):
        x = self.backbone(x)        # (B, C, 16, 16)
        x = self.quanv(x)           # (B, C, Hq, Wq)
        x = x.flatten(start_dim=1)  # (B, C*Hq*Wq)
        return self.fc(x)


# ============================================================================
# EuroSAT dataset (compatibile con struttura del multiseed)
# ============================================================================

class EuroSATDataset(Dataset):
    """Loads EuroSAT images from an ImageFolder structure.

    Si aspetta `root_dir/<classname>/*.jpg`. Filtra `selected_classes`
    e limita `max_per_class` samples per classe.
    """

    def __init__(self, root_dir: str, classes=None, max_per_class=None,
                 transform=None, img_size: int = 64):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset dir not found: {root_dir}")
        if classes is None:
            classes = sorted([d.name for d in self.root_dir.iterdir()
                              if d.is_dir()])
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        self.transform = transform

        self.samples = []
        for c in classes:
            cls_dir = self.root_dir / c
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Class dir not found: {cls_dir}")
            imgs = sorted([f for f in cls_dir.iterdir()
                           if f.suffix.lower() in {'.jpg', '.jpeg', '.png',
                                                    '.tif', '.tiff'}])
            if max_per_class is not None:
                imgs = imgs[:max_per_class]
            for img in imgs:
                self.samples.append((str(img), self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {root_dir} "
                               f"(classes={classes})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label


def build_dataloaders(config: Config4q):
    train_ds = EuroSATDataset(
        config.train_dir, classes=config.selected_classes,
        max_per_class=config.max_samples_per_class, img_size=config.img_size)
    val_ds = EuroSATDataset(
        config.val_dir, classes=config.selected_classes,
        max_per_class=config.max_samples_per_class, img_size=config.img_size)
    g = torch.Generator(); g.manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, generator=g)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers)
    return train_loader, val_loader, train_ds, val_ds


# ============================================================================
# Training di 1 seed
# ============================================================================

def train_one_seed(config: Config4q, verbose: bool = True) -> dict:
    """Runs training for 1 seed and returns metrics + predictions.

    Ritorna un dict con chiavi:
      seed, config, history (per-epoch), best_val_acc, final_val_acc,
      predictions (y_true, y_pred), wall_time_s
    """
    t_start = time.time()

    # Seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    # Backend manager (sceglie noisy/noiseless)
    if config.backend_type == "sim_noisy":
        if not config.noise_snapshot_path:
            raise ValueError("sim_noisy requires --noise-snapshot")
        bm = BackendManagerSimNoisy(
            config, config.noise_snapshot_path,
            aer_max_parallel_experiments=config.aer_max_parallel_experiments,
            noise_qubits=config.noise_qubits)
    elif config.backend_type == "sim_noiseless":
        bm = BackendManagerSimNoiseless(
            config,
            aer_max_parallel_experiments=config.aer_max_parallel_experiments)
    else:
        raise ValueError(f"Unknown backend_type: {config.backend_type}")
    bm.initialize()

    # Engine (transpila il sub-circuit una volta sola)
    engine = QuantumEnginePSSim(config, bm)

    # Modello + ottimizzatore
    model = HybridQCNN(config, engine, rng)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Data
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(config)
    print(f"  Train: {len(train_ds)} img | Val: {len(val_ds)} img | "
          f"classes={config.selected_classes}", flush=True)

    history = {
        'epoch_train_loss': [], 'epoch_train_acc': [],
        'epoch_val_loss': [], 'epoch_val_acc': [],
        'epoch_wall_s': [],
    }

    best_val_acc = 0.0
    final_y_true: list = []
    final_y_pred: list = []

    n_train_batches = len(train_loader)

    for epoch in range(config.max_epochs):
        t_ep = time.time()

        # ── Train ──
        model.train()
        tr_losses, tr_corrs, tr_tot = [], 0, 0
        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            t_b = time.time()
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            preds = out.argmax(dim=1)
            tr_corrs += (preds == labels).sum().item()
            tr_tot += labels.size(0)
            batch_acc = (preds == labels).float().mean().item()
            print(f"  Epoch {epoch+1:2d}/{config.max_epochs}  "
                  f"train batch {batch_idx:>2d}/{n_train_batches}  "
                  f"loss={loss.item():.3f} acc={batch_acc:.3f}  "
                  f"({time.time()-t_b:.1f}s)", flush=True)
        tr_loss = float(np.mean(tr_losses))
        tr_acc = tr_corrs / tr_tot

        # ── Val ──
        model.eval()
        v_losses, v_corrs, v_tot = [], 0, 0
        epoch_y_true, epoch_y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                out = model(imgs)
                loss = criterion(out, labels)
                v_losses.append(loss.item())
                preds = out.argmax(dim=1)
                v_corrs += (preds == labels).sum().item()
                v_tot += labels.size(0)
                epoch_y_true.extend(labels.cpu().numpy().tolist())
                epoch_y_pred.extend(preds.cpu().numpy().tolist())
        val_loss = float(np.mean(v_losses))
        val_acc = v_corrs / v_tot

        ep_wall = time.time() - t_ep
        history['epoch_train_loss'].append(tr_loss)
        history['epoch_train_acc'].append(tr_acc)
        history['epoch_val_loss'].append(val_loss)
        history['epoch_val_acc'].append(val_acc)
        history['epoch_wall_s'].append(ep_wall)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        # Salvo SEMPRE le predictions dell'ultima epoca (final)
        final_y_true = epoch_y_true
        final_y_pred = epoch_y_pred

        if verbose:
            print(f"  ── Epoch {epoch+1:2d}/{config.max_epochs} DONE | "
                  f"tr_loss={tr_loss:.3f} tr_acc={tr_acc:.3f} | "
                  f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} | "
                  f"{ep_wall:.1f}s ──", flush=True)

    wall_total = time.time() - t_start
    print(f"  Seed {config.seed}: best_val_acc={best_val_acc:.4f} "
          f"final_val_acc={history['epoch_val_acc'][-1]:.4f} "
          f"in {wall_total:.0f}s", flush=True)

    bm.close()

    return {
        'seed': config.seed,
        'backend_type': config.backend_type,
        'noise_snapshot': config.noise_snapshot_path,
        'config': asdict(config),
        'history': history,
        'best_val_acc': best_val_acc,
        'final_val_acc': history['epoch_val_acc'][-1],
        'predictions': {'y_true': final_y_true, 'y_pred': final_y_pred},
        'wall_time_s': wall_total,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Train 1 seed del modello QCNN 4q (noisy o noiseless).")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--backend', choices=['sim_noisy', 'sim_noiseless'],
                    default='sim_noisy')
    ap.add_argument('--noise-snapshot', default=None,
                    help='Path al pickle del NoiseModel (per sim_noisy)')
    ap.add_argument('--noise-qubits', type=int, nargs='+',
                    default=[0, 1, 2, 3],
                    help='Qubit fisici su cui restringere il NoiseModel')
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--aer-parallel', type=int, default=8)
    ap.add_argument('--max-samples', type=int, default=100,
                    help='max_samples_per_class')
    ap.add_argument('--max-epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--shots', type=int, default=500)
    args = ap.parse_args()

    cfg = Config4q(
        seed=args.seed,
        backend_type=args.backend,
        noise_snapshot_path=args.noise_snapshot,
        noise_qubits=list(args.noise_qubits),
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        aer_max_parallel_experiments=args.aer_parallel,
        max_samples_per_class=args.max_samples,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        shots=args.shots,
    )

    print("=" * 70)
    print(f"  4q QCNN training — seed={cfg.seed}, backend={cfg.backend_type}")
    print("=" * 70)
    print(f"  Quanv output: {cfg.quanv_output_size}×{cfg.quanv_output_size}, "
          f"flatten size: {cfg.flatten_size}")
    print(f"  Quantum: K={cfg.num_parallel_blocks} blocchi × {cfg.num_qubits}q, "
          f"shots={cfg.shots}")
    print(f"  Training: {cfg.max_epochs} ep, batch={cfg.batch_size}, "
          f"lr={cfg.lr}, max_samples/class={cfg.max_samples_per_class}")

    result = train_one_seed(cfg)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.output_dir) / f"results_seed_{cfg.seed:03d}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Saved: {out_path}")


if __name__ == '__main__':
    main()
