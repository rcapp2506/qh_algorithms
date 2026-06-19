"""
run_ccnn_small_matched_multiseed.py - CCNN-small matched to the QCNN run (Wave-K)

Mirror architecture of HybridConvNet (run_qcnn_multiseed.py), with the
QuantumConvLayer replaced by a classical Conv2d(6->6, kernel=3, padding=0).
Everything else (Conv1, Conv2, BN, head) is identical to the QCNN run.

Expected param count: ~463,908 (vs 463,574 for the QCNN). Difference: 334 params,
0.07% of the total -> like-for-like comparison.

USAGE
=====
On the i9, with the same venv as run_qcnn_multiseed.py:
    cd ~/code/Q-CONV
    python run_ccnn_small_matched_multiseed.py \
        --train-dir ./dataset/training \
        --val-dir ./dataset/validation \
        --output-dir ./Output_CCNN_small_matched_v1 \
        --max-epochs 10 \
        --max-samples 100 \
        --parallel-seeds 3   # 3 seed paralleli (come QCNN)

Tempi attesi: ~5-10 min per seed (no quantum, no parameter shift),
~30-60 min totali R=10 con 3 in parallelo.

OUTPUT
======
- results_run_NN_seed_SSSSS.json per ciascun seed (formato identico a QCNN)
- val_correct_final per-item per ciascun seed → McNemar paired vs QCNN
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# Force classical single-threading (no oversubscription needed)
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Lightning import compatible with both forms of the package:
#   pip install lightning           → import lightning as L
#   pip install pytorch-lightning   → import pytorch_lightning as L
try:
    import lightning as L
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import CSVLogger
except ModuleNotFoundError:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import Accuracy
from PIL import Image


# ───────────────────────────────────────────────────────────────────────────
# 1.  Config - cloned from QCNNConfig with only the relevant shapes
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class CCNNSmallConfig:
    # Dataset
    train_dir: str = "./dataset/training"
    val_dir: str = "./dataset/validation"
    img_size: int = 64
    in_channels: int = 3
    num_classes: int = 2
    selected_classes: Optional[list[str]] = None
    max_samples_per_class: Optional[int] = 100

    # Architettura speculare al QCNN (HybridConvNet defaults)
    num_conv_channels: int = 6
    conv_kernel_size: int = 5
    conv_padding: int = 2
    # "replacement" for the quanv layer: 6->6, ks=3, pad=0 (same shape as QuantumConvLayer)
    replacement_kernel_size: int = 3
    replacement_padding: int = 0
    dropout_rate: float = 0.0      # come QCNN (zero), non come CCNN-grande (0.05)
    use_batch_norm: bool = True    # come QCNN

    # Training (identico a QCNN)
    batch_size: int = 16
    max_epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 1e-4
    num_workers: int = 4
    early_stop_patience: int = 50  # disabilitato

    # Loop statistico (stesse seed del QCNN R=10)
    num_stat_runs: int = 10
    base_seed: int = 42

    # I/O
    run_name: str = "ccnn_small_matched_v1"
    output_dir: str = "Output_CCNN_small_matched_v1"
    seed: int = 42

    @property
    def feature_map_size(self) -> int:
        return self.img_size // 4

    @property
    def replacement_output_size(self) -> int:
        fm = self.feature_map_size
        return (fm + 2 * self.replacement_padding - self.replacement_kernel_size) + 1

    @property
    def flatten_size(self) -> int:
        return self.num_conv_channels * self.replacement_output_size ** 2


# ───────────────────────────────────────────────────────────────────────────
# 2.  Dataset (replicato da run_qcnn_multiseed.py)
# ───────────────────────────────────────────────────────────────────────────
class EuroSATDataset(Dataset):
    def __init__(self, root: str, img_size: int, selected_classes=None,
                 max_samples_per_class=None, seed=42):
        import random
        root = Path(root)
        if selected_classes is None:
            selected_classes = sorted([d.name for d in root.iterdir() if d.is_dir()])[:2]
        self.classes = selected_classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for c in self.classes:
            files = sorted((root / c).glob("*.jpg")) + sorted((root / c).glob("*.png"))
            files = list(files)
            random.Random(seed).shuffle(files)
            if max_samples_per_class is not None:
                files = files[:max_samples_per_class]
            for f in files:
                self.samples.append((str(f), self.class_to_idx[c]))

        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.float32) / 255.0   # HWC
        arr = arr.transpose(2, 0, 1)                    # CHW
        return torch.from_numpy(arr), label


class EuroSATDataModule(L.LightningDataModule):
    def __init__(self, config: CCNNSmallConfig):
        super().__init__()
        self.config = config
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        self.train_ds = EuroSATDataset(
            self.config.train_dir, self.config.img_size,
            self.config.selected_classes, self.config.max_samples_per_class,
            seed=self.config.seed,
        )
        self.val_ds = EuroSATDataset(
            self.config.val_dir, self.config.img_size,
            self.config.selected_classes, self.config.max_samples_per_class,
            seed=self.config.seed,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size,
                          shuffle=True, num_workers=self.config.num_workers,
                          persistent_workers=self.config.num_workers > 0,
                          pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.batch_size,
                          shuffle=False, num_workers=self.config.num_workers,
                          persistent_workers=self.config.num_workers > 0,
                          pin_memory=False)


# ───────────────────────────────────────────────────────────────────────────
# 3.  Metrics callback (per_epoch + val_correct_final per-item)
# ───────────────────────────────────────────────────────────────────────────
class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.train_accs   = []
        self.val_losses   = []
        self.val_accs     = []
        self.val_correct_final = []   # 0/1 per item at the last epoch
        self.val_labels_final  = []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "val_loss" in m: self.val_losses.append(float(m["val_loss"]))
        if "val_accuracy" in m: self.val_accs.append(float(m["val_accuracy"]))

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "train_loss" in m: self.train_losses.append(float(m["train_loss"]))
        if "train_accuracy" in m: self.train_accs.append(float(m["train_accuracy"]))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                 batch_idx, dataloader_idx=0):
        # Only in the last validation epoch: record per-item correctness
        if trainer.current_epoch == trainer.max_epochs - 1:
            x, y = batch
            with torch.no_grad():
                logits = pl_module(x)
                preds = logits.argmax(1)
                correct = (preds == y).int().cpu().tolist()
                labels  = y.cpu().tolist()
            self.val_correct_final.extend(correct)
            self.val_labels_final.extend(labels)


# ───────────────────────────────────────────────────────────────────────────
# 4.  Modello — sostituzione speculare di HybridConvNet
# ───────────────────────────────────────────────────────────────────────────
class MatchedClassicalConvNet(nn.Module):
    """Identical to HybridConvNet (run_qcnn_multiseed.py) except for the
    QuantumConvLayer, here replaced by Conv2d(6->6, ks=3, pad=0)."""

    def __init__(self, config: CCNNSmallConfig):
        super().__init__()
        ch = config.num_conv_channels
        ks = config.conv_kernel_size
        pad = config.conv_padding
        drop = config.dropout_rate

        bn = nn.BatchNorm2d if config.use_batch_norm else nn.Identity

        self.conv1 = nn.Sequential(
            nn.Conv2d(config.in_channels, ch, ks, padding=pad),
            bn(ch), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, ks, padding=pad),
            bn(ch), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Sostituzione del QuantumConvLayer: Conv2d 6→6 ks=3 pad=0
        # (same output shape as the quanv: 14x14x6)
        self.replacement = nn.Sequential(
            nn.Conv2d(ch, ch, config.replacement_kernel_size,
                      padding=config.replacement_padding),
            bn(ch), nn.ReLU(),
        )

        self.drop1 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop_q = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        flat = config.flatten_size
        fc1_out = max(flat // 3, 64)
        self.classifier = nn.Sequential(
            nn.Linear(flat, fc1_out),
            nn.ReLU(),
            nn.Linear(fc1_out, config.num_classes),
        )

    def forward(self, x):
        x = self.drop1(self.conv1(x))
        x = self.drop2(self.conv2(x))
        x = self.drop_q(self.replacement(x))
        return self.classifier(x.flatten(1))


class MatchedClassifier(L.LightningModule):
    def __init__(self, model: MatchedClassicalConvNet, config: CCNNSmallConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=config.num_classes)

    def forward(self, x): return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(1)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_acc(preds, y), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(1)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_acc(preds, y), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr,
                               weight_decay=self.config.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config.max_epochs)
        return [opt], [sched]


# ───────────────────────────────────────────────────────────────────────────
# 5.  Loop multi-seed
# ───────────────────────────────────────────────────────────────────────────
SEED_LIST = [42, 153, 264, 375, 486, 597, 708, 819, 930, 1041]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_single_training(config: CCNNSmallConfig, seed: int, run_idx: int,
                         output_dir: Path):
    L.seed_everything(seed, workers=True)
    cfg = CCNNSmallConfig(**{**asdict(config), "seed": seed})

    dm = EuroSATDataModule(cfg)
    dm.setup()

    model = MatchedClassicalConvNet(cfg)
    n_params = count_params(model)
    classifier = MatchedClassifier(model, cfg)

    cb = MetricsCallback()
    logger = CSVLogger(save_dir=str(output_dir), name=f"run_{run_idx:02d}_s{seed}")
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[cb],
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
    )
    t0 = time.time()
    trainer.fit(classifier, datamodule=dm)
    elapsed = time.time() - t0

    # Save JSON
    result = {
        "seed": seed,
        "run_idx": run_idx,
        "elapsed": elapsed,
        "actual_epochs": cfg.max_epochs + 1,   # PyTorch Lightning starts at 0; align con QCNN JSON
        "train_losses": cb.train_losses,
        "val_losses": cb.val_losses,
        "train_accuracies": cb.train_accs,
        "val_accuracies": cb.val_accs,
        "best_val_acc": float(max(cb.val_accs)) if cb.val_accs else 0.0,
        "best_val_loss": float(min(cb.val_losses)) if cb.val_losses else 0.0,
        "final_train_acc": float(cb.train_accs[-1]) if cb.train_accs else 0.0,
        "final_val_acc":   float(cb.val_accs[-1]) if cb.val_accs else 0.0,
        "val_correct_final": cb.val_correct_final,
        "val_labels_final":  cb.val_labels_final,
        "n_val": len(cb.val_correct_final),
    }

    out = {
        "architecture": "ccnn_small_matched_v1",
        "config": {k: getattr(cfg, k) for k in [
            "num_conv_channels", "conv_kernel_size", "conv_padding",
            "replacement_kernel_size", "replacement_padding",
            "dropout_rate", "use_batch_norm",
            "max_epochs", "batch_size", "lr", "max_samples_per_class",
            "num_stat_runs", "base_seed",
        ]},
        "n_trainable_params": n_params,
        "run_idx": run_idx,
        "seed": seed,
        "result": result,
        "wallclock_seconds": elapsed,
    }
    out_path = output_dir / f"results_run_{run_idx:02d}_seed_{seed:05d}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ⏱  {elapsed:.1f}s — final_val_acc={result['final_val_acc']:.4f}, "
          f"best={result['best_val_acc']:.4f}, params={n_params:,}")
    print(f"  → {out_path}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default="./dataset/training")
    p.add_argument("--val-dir",   default="./dataset/validation")
    p.add_argument("--output-dir", default="./Output_CCNN_small_matched_v1")
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--parallel-seeds", type=int, default=3,
                   help="Number of seeds in parallel via multiprocessing")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = CCNNSmallConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        max_samples_per_class=args.max_samples,
        num_workers=args.num_workers,
    )

    # Sanity check: param count su un modello dummy
    dummy = MatchedClassicalConvNet(base_config)
    print(f"\n*** MatchedClassicalConvNet param count: {count_params(dummy):,} ***")
    print(f"    (atteso: ~463,908; QCNN runned: 463,574; Δ = {count_params(dummy) - 463574:,})")
    print(f"    flatten_size: {base_config.flatten_size}")
    print(f"    replacement_output_size: {base_config.replacement_output_size}\n")

    if args.parallel_seeds <= 1:
        # Serial
        for run_idx, seed in enumerate(SEED_LIST):
            print(f"\n=== RUN {run_idx+1}/{len(SEED_LIST)} — seed={seed} ===")
            run_single_training(base_config, seed, run_idx, output_dir)
    else:
        # Parallel via multiprocessing
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.parallel_seeds) as pool:
            tasks = [(base_config, seed, run_idx, output_dir)
                     for run_idx, seed in enumerate(SEED_LIST)]
            pool.starmap(run_single_training, tasks)


if __name__ == "__main__":
    main()
