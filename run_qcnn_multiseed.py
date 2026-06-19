#!/usr/bin/env python3
"""run_qcnn_multiseed.py - Standalone batch QCNN R=10 multi-seed (Wave K v3)

Launched from the terminal, not from Jupyter. Fixes the spawn+notebook
AttributeError bug by moving _worker_run_one_seed to module top-level (so the
child process can import it via `import __main__`).

Usage:
    cd ~/code/Q-CONV
    conda activate qiskitq
    nohup python run_qcnn_multiseed.py > qcnn_run.log 2>&1 &
    tail -f qcnn_run.log

or in screen/tmux:
    screen -S qcnn
    python run_qcnn_multiseed.py
    Ctrl-A D  # detach
    # ... reconnect later:
    screen -r qcnn

Output:
    Output_QCNN_v1_multiseed/results.json          (final, R completed seeds)
    Output_QCNN_v1_multiseed/partial_state.json    (checkpoint after each seed)
    Output_QCNN_v1_multiseed/predictions/*.csv     (for Wilcoxon cross-arch)
    Output_QCNN_v1_multiseed/stat_runs/run_NN_sXXX/ (Lightning logs per seed)

S3 tuning configuration (post-bench v3):
    OMP_NUM_THREADS=2 (BEFORE the imports)
    n_parallel_seeds=3, aer_max_parallel_experiments=3, n_parallel_chunks=4
    Speedup: 2.36x vs serial S1. R=10 estimated ~16-20h on i9-12900H.

Robustness:
    - Checkpointing per-seed atomic (tmp+replace)
    - A single failed seed does NOT interrupt the loop; the others continue
    - KeyboardInterrupt (Ctrl-C) saves the current state before shutdown
    - For partial recovery: cp partial_state.json results.json

CLI arguments:
    --num-runs R           (default 10)
    --output-dir DIR       (default Output_QCNN_v1_multiseed)
    --max-epochs E         (default 10)
    --max-samples N        (default 100 per class)
    --n-parallel-seeds W   (default 3, S3 tuning)
    --serial               (force serial loop, ignore ProcessPool)

Author: Wave K post-bench v3 (Cap.3 Cappuccio thesis)
"""

# ═══════════════════════════════════════════════════════════════════════════
#  ENV VAR SETUP (must precede every numpy/torch/qiskit import)
# ═══════════════════════════════════════════════════════════════════════════
import os as _os_threads
_os_threads.environ.setdefault('OMP_NUM_THREADS', '2')
_os_threads.environ.setdefault('MKL_NUM_THREADS', '2')
_os_threads.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
_os_threads.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

# ═══════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import argparse
import copy as _copy
import csv
import gc
import io
import json as _json
import multiprocessing as _mp
import os
import pickle as _pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics import Accuracy
from PIL import Image

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    HAS_AER = True
except ImportError:
    HAS_AER = False

QISKIT_VERSION = tuple(int(x) for x in qiskit.__version__.split('.')[:2])
assert QISKIT_VERSION >= (2, 0), f"Qiskit >= 2.0 required, found {qiskit.__version__}"


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG (cell 4 of the notebook)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QCNNConfig:
    """Config - Filippi reproduction (Wave K v3 batch)."""

    # ── Backend ──
    backend_type: Literal["statevector", "aer", "aer_noise", "ibm"] = "aer"
    ibm_token: Optional[str] = None
    ibm_backend_name: Optional[str] = None
    ibm_instance: Optional[str] = None
    ibm_channel: str = "ibm_quantum"
    ibm_min_qubits: int = 127
    optimization_level: int = 1
    noise_backend_name: str = "ibm_brisbane"

    # ── Quantum circuit ──
    num_qubits: int = 9
    kernel_size: int = 3
    stride: int = 1
    quanv_padding: int = 0
    measure_qubit: int = 0
    shots: int = 0

    # ── Rete classica ──
    num_conv_channels: int = 6
    conv_kernel_size: int = 5
    conv_padding: int = 2
    dropout_rate: float = 0.0

    # ── Dataset EuroSAT ──
    train_dir: str = "./dataset/training"
    val_dir: str = "./dataset/validation"
    img_size: int = 64
    in_channels: int = 3
    num_classes: int = 2
    selected_classes: Optional[List[str]] = None
    max_samples_per_class: Optional[int] = 100

    # ── Training ──
    batch_size: int = 16
    max_epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 1e-4
    num_workers: int = 4
    early_stop_patience: int = 12

    # ── Loop statistico ──
    num_stat_runs: int = 10
    base_seed: int = 42

    # ── Parallelismo (S3 tuning) ──
    aer_max_parallel_experiments: int = 3
    n_parallel_chunks: int = 4
    parallel_seeds: bool = True
    n_parallel_seeds: int = 3

    # ── I/O ──
    run_name: str = "filippi_v2"
    output_dir: str = "Output_QCNN_v1_multiseed"
    seed: int = 42

    @property
    def num_weights(self) -> int:
        return self.num_qubits

    @property
    def feature_map_size(self) -> int:
        return self.img_size // 4

    @property
    def quanv_output_size(self) -> int:
        fm = self.feature_map_size
        return (fm + 2 * self.quanv_padding - self.kernel_size) // self.stride + 1

    @property
    def flatten_size(self) -> int:
        return self.num_conv_channels * self.quanv_output_size ** 2

    @property
    def patches_per_channel(self) -> int:
        return self.quanv_output_size ** 2


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET (cell 6)
# ═══════════════════════════════════════════════════════════════════════════

class EuroSATDataset(Dataset):
    """EuroSAT con supporto selezione classi + limite campioni."""

    def __init__(self, root_dir, transform=None, num_classes=4,
                 selected_classes=None, max_samples_per_class=None, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.rng = random.Random(seed)

        available = sorted([d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))])

        if selected_classes:
            self.classes = [c for c in selected_classes if c in available]
        else:
            self.classes = available[:num_classes]

        self.data = []
        for cls_idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name)
            imgs = sorted([f for f in os.listdir(cls_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
            if max_samples_per_class and len(imgs) > max_samples_per_class:
                self.rng.shuffle(imgs)
                imgs = imgs[:max_samples_per_class]
            for img_file in imgs:
                self.data.append((os.path.join(cls_path, img_file), cls_idx))

        self.rng.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class EuroSATDataModule(L.LightningDataModule):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, config: QCNNConfig):
        super().__init__()
        self.config = config
        self.class_names = None

    def setup(self, stage=None):
        c = self.config
        train_tf = transforms.Compose([
            transforms.Resize((c.img_size, c.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((c.img_size, c.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        if c.train_dir and os.path.exists(c.train_dir):
            self.train_dataset = EuroSATDataset(
                c.train_dir, train_tf, c.num_classes,
                c.selected_classes, c.max_samples_per_class, c.seed)
            self.class_names = self.train_dataset.classes
            print(f"  Training: {len(self.train_dataset)} img "
                  f"({len(self.class_names)} classi: {self.class_names})")
        else:
            print(f"  ⚠️  Train dir non trovata → dataset sintetico")
            self.train_dataset = self._synth(600)
            self.class_names = [f'C{i}' for i in range(c.num_classes)]

        if c.val_dir and os.path.exists(c.val_dir):
            self.val_dataset = EuroSATDataset(
                c.val_dir, val_tf, c.num_classes,
                c.selected_classes, c.max_samples_per_class, c.seed + 1)
            print(f"  Validation: {len(self.val_dataset)} img")
        else:
            print(f"  ⚠️  Val dir non trovata → dataset sintetico")
            self.val_dataset = self._synth(200)

    def _synth(self, n):
        class S(Dataset):
            def __init__(s, n, nc, sz, ch):
                s.data = [(torch.randn(ch, sz, sz), random.randint(0, nc-1)) for _ in range(n)]
            def __len__(s): return len(s.data)
            def __getitem__(s, i): return s.data[i]
        c = self.config
        return S(n, c.num_classes, c.img_size, c.in_channels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                         shuffle=True, num_workers=self.config.num_workers,
                         pin_memory=True, persistent_workers=self.config.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size,
                         shuffle=False, num_workers=self.config.num_workers,
                         pin_memory=True, persistent_workers=self.config.num_workers > 0)


# ═══════════════════════════════════════════════════════════════════════════
#  METRICS LOGGER (cell 8)
# ═══════════════════════════════════════════════════════════════════════════

class MetricsLogger(Callback):
    """Records metrics for each epoch - logs to file + memory."""

    def __init__(self, log_dir=None):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.log_dir = log_dir
        self._csv_file = None
        self._csv_writer = None

    def on_fit_start(self, trainer, pl_module):
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self._csv_file = open(os.path.join(self.log_dir, 'metrics.csv'), 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        tl = metrics.get('train_loss_epoch', metrics.get('train_loss', torch.tensor(0))).item()
        vl = metrics.get('val_loss', torch.tensor(0)).item()
        ta = metrics.get('train_accuracy_epoch', metrics.get('train_accuracy', torch.tensor(0))).item()
        va = metrics.get('val_accuracy', torch.tensor(0)).item()

        self.train_losses.append(tl)
        self.val_losses.append(vl)
        self.train_accuracies.append(ta)
        self.val_accuracies.append(va)

        if self._csv_writer:
            self._csv_writer.writerow([trainer.current_epoch, f'{tl:.6f}', f'{vl:.6f}',
                                       f'{ta:.6f}', f'{va:.6f}'])
            self._csv_file.flush()

    def on_fit_end(self, trainer, pl_module):
        if self._csv_file:
            self._csv_file.close()


# ═══════════════════════════════════════════════════════════════════════════
#  BACKEND MANAGER (cell 10)
# ═══════════════════════════════════════════════════════════════════════════

class BackendManager:
    """Gestisce Estimator per Qiskit 2.x."""

    def __init__(self, config: QCNNConfig):
        self.config = config
        self.estimator = None
        self.backend = None
        self.session = None
        self.pass_manager = None
        self.rng = np.random.default_rng(config.seed)
        self.backend_name = "unknown"
        self.backend_type = config.backend_type
        self.available_qubits = config.num_qubits

    def initialize(self):
        print(f"Backend: {self.config.backend_type.upper()}")
        if self.config.backend_type == "statevector":
            self._setup_statevector()
        elif self.config.backend_type == "aer":
            self._setup_aer()
        elif self.config.backend_type == "aer_noise":
            self._setup_aer_noise()
        elif self.config.backend_type == "ibm":
            self._setup_ibm()
        print(f"  ✓ {self.backend_name}")

    def _setup_statevector(self):
        self.estimator = StatevectorEstimator()
        self.available_qubits = self.config.num_qubits
        self.backend_name = "StatevectorEstimator"

    def _init_aer_estimator(self, aer_backend):
        try:
            self.estimator = AerEstimator(aer_backend)
        except TypeError:
            self.estimator = AerEstimator()

    def _setup_aer(self):
        if not HAS_AER:
            print("  ⚠️ qiskit-aer not available -> StatevectorEstimator fallback")
            self._setup_statevector()
            return
        n_parallel = getattr(self.config, 'aer_max_parallel_experiments', 0)
        aer_backend = AerSimulator(method='statevector')
        if n_parallel > 0:
            aer_backend.set_options(max_parallel_experiments=n_parallel,
                                    max_parallel_threads=n_parallel)
        self.backend = aer_backend
        self.available_qubits = aer_backend.num_qubits
        self._init_aer_estimator(aer_backend)
        if n_parallel > 0:
            self.backend_name = f"AerSimulator(statevector, parallel={n_parallel})"
        else:
            self.backend_name = f"AerSimulator(statevector, parallel=auto)"

    def _setup_aer_noise(self):
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            svc_kwargs = {}
            if self.config.ibm_token:
                svc_kwargs = dict(channel=self.config.ibm_channel, token=self.config.ibm_token)
                if self.config.ibm_instance:
                    svc_kwargs['instance'] = self.config.ibm_instance
            service = QiskitRuntimeService(**svc_kwargs)
            real_backend = service.backend(self.config.noise_backend_name)
            from qiskit_aer.noise import NoiseModel
            noise_model = NoiseModel.from_backend(real_backend)
            aer_backend = AerSimulator(noise_model=noise_model, method='density_matrix')
            self.backend = aer_backend
            self.available_qubits = aer_backend.num_qubits
            self._init_aer_estimator(aer_backend)
            self.pass_manager = generate_preset_pass_manager(
                optimization_level=self.config.optimization_level, backend=aer_backend)
            self.backend_name = f"AerSimulator(noise={self.config.noise_backend_name})"
        except Exception as e:
            print(f"  ⚠️ Noise model fallback: {e}")
            self._setup_aer()

    def _setup_ibm(self):
        from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
        svc_kwargs = {}
        if self.config.ibm_token:
            svc_kwargs = dict(channel=self.config.ibm_channel, token=self.config.ibm_token)
            if self.config.ibm_instance:
                svc_kwargs['instance'] = self.config.ibm_instance
        service = QiskitRuntimeService(**svc_kwargs)
        backend = service.least_busy(min_num_qubits=self.config.ibm_min_qubits, operational=True)
        self.backend = backend
        self.available_qubits = backend.num_qubits
        self.estimator = EstimatorV2(backend=backend)
        self.pass_manager = generate_preset_pass_manager(
            optimization_level=self.config.optimization_level, backend=backend)
        self.backend_name = backend.name

    def transpile(self, circuit):
        if self.pass_manager:
            return self.pass_manager.run(circuit)
        return circuit

    def close(self):
        if self.session:
            self.session.close()


# ═══════════════════════════════════════════════════════════════════════════
#  CIRCUIT BUILDER (cell 12)
# ═══════════════════════════════════════════════════════════════════════════

class FilippiCircuitBuilder:
    """Filippi quantum circuit: 9 qubits, 9 trainable RZ, Z measurement on qubit 0."""

    def __init__(self, config: QCNNConfig):
        self.n = config.num_qubits
        self.num_weights = config.num_weights
        self.measure_qubit = config.measure_qubit
        self.total_qubits = self.n

        self.input_params = ParameterVector('x', self.n)
        self.weight_params = ParameterVector('w', self.num_weights)

        self.circuit = self._build()

        self.observables = [
            SparsePauliOp.from_sparse_list(
                [('Z', [self.measure_qubit], 1.0)], num_qubits=self.n
            )
        ]
        self.num_observables = 1

        self.param_list = list(self.circuit.parameters)
        self.num_params = len(self.param_list)
        param_to_idx = {p: i for i, p in enumerate(self.param_list)}
        self.input_indices = [param_to_idx[self.input_params[i]] for i in range(self.n)]
        self.weight_indices = [param_to_idx[self.weight_params[i]] for i in range(self.num_weights)]

    def _build(self):
        qc = QuantumCircuit(self.n)
        for i in range(self.n):
            qc.ry(self.input_params[i], i)
        for i in range(self.n):
            qc.h(i)
        for i in range(self.n - 1):
            qc.cx(i, i + 1)
        for i in range(self.n):
            qc.rz(self.weight_params[i], i)
        for i in range(self.n - 2, -1, -1):
            qc.cx(i, i + 1)
        return qc

    def build_param_array(self, inputs_2d, weights_1d):
        N = inputs_2d.shape[0]
        params = np.zeros((N, self.num_params))
        for j, idx in enumerate(self.weight_indices):
            params[:, idx] = weights_1d[j]
        for j, idx in enumerate(self.input_indices):
            params[:, idx] = inputs_2d[:, j]
        return params, N

    def parse_output(self, evs, N):
        return evs[:N]


# ═══════════════════════════════════════════════════════════════════════════
#  QUANTUM ENGINE con PUB splitting (cell 14, Wave K v3)
# ═══════════════════════════════════════════════════════════════════════════

class QuantumEngine:
    """Motore esecuzione con PUB batching + PUB splitting (K chunks).

    Speedup 2.36x vs K=1 AerP=4 (bench v2 FBG): K-PUB ≡ 1-PUB a max_diff=0.
    """

    SHIFT = np.pi / 2

    def __init__(self, config: QCNNConfig, backend_manager: BackendManager):
        self.config = config
        self.backend_manager = backend_manager
        self.n = config.num_qubits
        self.K = max(1, getattr(config, 'n_parallel_chunks', 1))

        self.builder = FilippiCircuitBuilder(config)
        self.circuit = self.builder.circuit
        self.observables = self.builder.observables
        self.num_weights = self.builder.num_weights
        self.num_observables = self.builder.num_observables

        if backend_manager.pass_manager:
            self.circuit_exec = backend_manager.transpile(self.circuit)
        else:
            self.circuit_exec = self.circuit

        self.total_estimator_calls = 0
        self.total_pub_count = 0

    def _run_pubs(self, pubs):
        job = self.backend_manager.estimator.run(pubs)
        results = job.result()
        self.total_estimator_calls += 1
        self.total_pub_count += len(pubs)
        return results

    def _effective_K(self, N):
        return self.K if N >= self.K else 1

    def _make_chunked_pubs(self, pv, K_eff):
        if K_eff == 1:
            return [(self.circuit_exec, self.observables, pv[:, np.newaxis, :])]
        chunks = np.array_split(pv, K_eff, axis=0)
        return [(self.circuit_exec, self.observables, c[:, np.newaxis, :])
                for c in chunks]

    def _gather_chunks(self, results, start_idx, K_eff, N):
        if K_eff == 1:
            return np.array(results[start_idx].data.evs)[:N]
        parts = [np.array(results[start_idx + k].data.evs) for k in range(K_eff)]
        return np.concatenate(parts, axis=0)[:N]

    def forward(self, inputs, weights):
        pv, N = self.builder.build_param_array(inputs, weights)
        K_eff = self._effective_K(N)
        pubs = self._make_chunked_pubs(pv, K_eff)
        results = self._run_pubs(pubs)
        evs = self._gather_chunks(results, 0, K_eff, N)
        return self.builder.parse_output(evs, N)

    def forward_and_gradient(self, inputs, weights, skip_input_grad=False):
        N = inputs.shape[0]
        nw = self.num_weights
        n = self.n
        n_obs = self.num_observables
        shift = self.SHIFT
        K_eff = self._effective_K(N)

        pubs = []
        pv_fwd, _ = self.builder.build_param_array(inputs, weights)
        pubs.extend(self._make_chunked_pubs(pv_fwd, K_eff))

        for j in range(nw):
            w_plus = weights.copy(); w_plus[j] += shift
            w_minus = weights.copy(); w_minus[j] -= shift
            pv_p, _ = self.builder.build_param_array(inputs, w_plus)
            pv_m, _ = self.builder.build_param_array(inputs, w_minus)
            pubs.extend(self._make_chunked_pubs(pv_p, K_eff))
            pubs.extend(self._make_chunked_pubs(pv_m, K_eff))

        if not skip_input_grad:
            for i in range(n):
                x_plus = inputs.copy(); x_plus[:, i] += shift
                x_minus = inputs.copy(); x_minus[:, i] -= shift
                pv_p, _ = self.builder.build_param_array(x_plus, weights)
                pv_m, _ = self.builder.build_param_array(x_minus, weights)
                pubs.extend(self._make_chunked_pubs(pv_p, K_eff))
                pubs.extend(self._make_chunked_pubs(pv_m, K_eff))

        results = self._run_pubs(pubs)

        fwd = self._gather_chunks(results, 0, K_eff, N)

        grad_w = np.zeros((nw, N, n_obs))
        for j in range(nw):
            ev_p = self._gather_chunks(results, K_eff * (1 + 2*j), K_eff, N)
            ev_m = self._gather_chunks(results, K_eff * (2 + 2*j), K_eff, N)
            grad_w[j] = (ev_p - ev_m) / 2.0

        if skip_input_grad:
            grad_x = None
        else:
            grad_x = np.zeros((n, N, n_obs))
            base = 1 + 2 * nw
            for i in range(n):
                ev_p = self._gather_chunks(results, K_eff * (base + 2*i), K_eff, N)
                ev_m = self._gather_chunks(results, K_eff * (base + 2*i + 1), K_eff, N)
                grad_x[i] = (ev_p - ev_m) / 2.0

        return fwd, grad_w, grad_x


# ═══════════════════════════════════════════════════════════════════════════
#  QUANTUM CONV FUNCTION + LAYER (cell 16)
# ═══════════════════════════════════════════════════════════════════════════

class QuantumConvFunction(torch.autograd.Function):
    """Autograd con parameter shift batched — output scalare per patch."""

    @staticmethod
    def forward(ctx, input_patches, weights, engine):
        input_np = input_patches.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        outputs = engine.forward(input_np, weights_np)
        ctx.save_for_backward(input_patches, weights)
        ctx.engine = engine
        return torch.tensor(outputs, dtype=torch.float32, device=input_patches.device)

    @staticmethod
    def backward(ctx, grad_output):
        input_patches, weights = ctx.saved_tensors
        engine = ctx.engine
        device = grad_output.device

        input_np = input_patches.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        # Ottimizzazione E: salta input shifts se .detach() applicato a monte
        skip_input = not input_patches.requires_grad

        _, grad_w_jac, grad_x_jac = engine.forward_and_gradient(
            input_np, weights_np, skip_input_grad=skip_input)

        grad_weights = np.einsum('jnq,nq->j', grad_w_jac, grad_out_np)

        if skip_input or grad_x_jac is None:
            grad_inputs = None
        else:
            grad_inputs = np.einsum('inq,nq->ni', grad_x_jac, grad_out_np)
            grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32, device=device)

        return (grad_inputs,
                torch.tensor(grad_weights, dtype=torch.float32, device=device),
                None)


class QuantumConvLayer(nn.Module):
    """Quantum convolutional layer - Filippi + channel batching.

    Quantum weights shared across all channels (depthwise, shared weights).
    All channels in a SINGLE estimator call (channel batching).
    """

    def __init__(self, config: QCNNConfig, backend_manager: BackendManager):
        super().__init__()
        self.config = config
        self.n = config.num_qubits
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.padding = config.quanv_padding

        self.input_scale = nn.Parameter(torch.ones(self.n) * 1.0)
        self.engine = QuantumEngine(config, backend_manager)

        init_w = (backend_manager.rng.random(config.num_weights) * 2 - 1) * 0.3
        self.quantum_weights = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))

    def forward(self, x):
        B, C, H, W = x.shape

        if self.padding > 0:
            x_pad = F.pad(x, [self.padding]*4, mode='constant', value=0.0)
        else:
            x_pad = x
        _, _, Hp, Wp = x_pad.shape

        H_out = (Hp - self.kernel_size) // self.stride + 1
        W_out = (Wp - self.kernel_size) // self.stride + 1
        P = H_out * W_out

        all_patches = []
        for c in range(C):
            x_c = x_pad[:, c:c+1, :, :]
            patches = F.unfold(x_c, kernel_size=self.kernel_size, stride=self.stride)
            p_min = patches.min(dim=2, keepdim=True).values
            p_max = patches.max(dim=2, keepdim=True).values
            p_range = (p_max - p_min).clamp(min=1e-8)
            patches_scaled = (patches - p_min) / p_range * np.pi
            all_patches.append(patches_scaled.permute(0, 2, 1).contiguous())

        all_patches = torch.cat(all_patches, dim=1)
        all_flat = all_patches.reshape(-1, self.n)

        # Ottimizzazione E: detach input grad (dimezza i PUB del backward, 37→19)
        all_flat = all_flat.detach()

        q_out = QuantumConvFunction.apply(all_flat, self.quantum_weights, self.engine)
        q_out = q_out.reshape(B, C, P)
        return q_out.reshape(B, C, H_out, W_out)


# ═══════════════════════════════════════════════════════════════════════════
#  HYBRID MODEL (cell 18 + 20)
# ═══════════════════════════════════════════════════════════════════════════

class HybridConvNet(nn.Module):
    """Conv1(3→6)+BN+Pool → Conv2(6→6)+BN+Pool → Quanv(6→6, 9q) → FC."""

    def __init__(self, config: QCNNConfig, quantum_layer: QuantumConvLayer):
        super().__init__()
        ch = config.num_conv_channels
        ks = config.conv_kernel_size
        pad = config.conv_padding
        drop = config.dropout_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(config.in_channels, ch, ks, padding=pad),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, ks, padding=pad),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.drop1 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop_q = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.quantum_conv = quantum_layer

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
        x = self.drop_q(self.quantum_conv(x))
        return self.classifier(x.flatten(1))


class HybridQCNNClassifier(L.LightningModule):

    def __init__(self, model: HybridConvNet, config: QCNNConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(1)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_acc(preds, y), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(1)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.val_acc(preds, y), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr,
                               weight_decay=self.config.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config.max_epochs)
        return [opt], [sched]


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS (cell 30)
# ═══════════════════════════════════════════════════════════════════════════

def create_fresh_model(config, backend_manager, device, seed, verbose=True):
    """Nuovo modello con seed diverso."""
    L.seed_everything(seed, workers=True)
    backend_manager.rng = np.random.default_rng(seed)

    if not verbose:
        f = io.StringIO()
        with redirect_stdout(f):
            ql = QuantumConvLayer(config, backend_manager)
            mdl = HybridConvNet(config, ql)
    else:
        ql = QuantumConvLayer(config, backend_manager)
        mdl = HybridConvNet(config, ql)

    return mdl.to(device), ql


def collect_val_predictions(classifier, data_module, device):
    """Pass deterministica sul val set post-trainer.fit per vettore per-item di correttezza."""
    classifier.eval()
    classifier = classifier.to(device)
    correct, labels = [], []
    with torch.no_grad():
        for x, y in data_module.val_dataloader():
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            preds = logits.argmax(dim=1)
            corr = (preds == y).to(torch.int64).cpu().tolist()
            correct.extend(int(c) for c in corr)
            labels.extend(int(v) for v in y.cpu().tolist())
    return correct, labels


def run_single_training(config, backend_manager, data_module, device, seed, run_idx, verbose=True):
    """Una singola run di training."""
    print(f"\n{'='*60}")
    print(f"  RUN {run_idx+1}/{config.num_stat_runs} — seed={seed}")
    print(f"{'='*60}")

    mdl, ql = create_fresh_model(config, backend_manager, device, seed, verbose)
    classifier = HybridQCNNClassifier(mdl, config)

    log_dir = os.path.join(config.output_dir, 'stat_runs', f'run_{run_idx:02d}_s{seed}')
    metrics_logger = MetricsLogger(log_dir=log_dir)

    callbacks = [
        metrics_logger,
        EarlyStopping(monitor='val_loss', patience=config.early_stop_patience,
                      mode='min', verbose=verbose),
    ]

    if run_idx == 0:
        best_ckpt = ModelCheckpoint(
            dirpath=log_dir, filename='best-{epoch}-{val_loss:.4f}',
            monitor='val_loss', mode='min', save_top_k=1)
        callbacks.append(best_ckpt)

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=TensorBoardLogger(config.output_dir, name='stat_logs',
                                version=f'run_{run_idx:02d}'),
        accelerator='auto', devices=1,
        log_every_n_steps=1,
        enable_progress_bar=verbose,
        enable_checkpointing=(run_idx == 0),
    )

    t0 = time.time()
    trainer.fit(classifier, data_module)
    elapsed = time.time() - t0
    actual_epochs = trainer.current_epoch + 1

    val_correct, val_labels = collect_val_predictions(classifier, data_module, device)

    result = {
        'seed': seed, 'run_idx': run_idx, 'elapsed': elapsed,
        'actual_epochs': actual_epochs,
        'train_losses': list(metrics_logger.train_losses),
        'val_losses': list(metrics_logger.val_losses),
        'train_accuracies': list(metrics_logger.train_accuracies),
        'val_accuracies': list(metrics_logger.val_accuracies),
        'best_val_acc': max(metrics_logger.val_accuracies) if metrics_logger.val_accuracies else 0,
        'best_val_loss': min(metrics_logger.val_losses) if metrics_logger.val_losses else float('inf'),
        'final_train_acc': metrics_logger.train_accuracies[-1] if metrics_logger.train_accuracies else 0,
        'final_val_acc': metrics_logger.val_accuracies[-1] if metrics_logger.val_accuracies else 0,
        'val_correct_final': val_correct,
        'val_labels_final': val_labels,
        'n_val': len(val_correct),
        'estimator_calls': ql.engine.total_estimator_calls,
        'pub_count': ql.engine.total_pub_count,
    }

    print(f"  ⏱  {elapsed:.0f}s ({actual_epochs} epochs)")
    print(f"  Best val_acc: {result['best_val_acc']:.4f}")
    print(f"  Final val_acc: {result['final_val_acc']:.4f} ({sum(val_correct)}/{len(val_correct)})")

    del classifier, trainer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER FUNCTION (top-level → spawn-safe)
# ═══════════════════════════════════════════════════════════════════════════

def _worker_run_one_seed(cfg, run_idx, seed):
    """Runs a single training run in a sub-process.

    MUST be at module top-level because spawn must be able to import it
    via `import __main__`. Do NOT define it inside main() or another function.
    """
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bm = BackendManager(cfg)
    _bm.initialize()
    _dm = EuroSATDataModule(cfg)
    _dm.setup()
    _result = run_single_training(cfg, _bm, _dm, _device, seed, run_idx, verbose=False)
    try:
        _bm.close()
    except Exception:
        pass
    return _result


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINTING (cell 31 v3)
# ═══════════════════════════════════════════════════════════════════════════

def _config_subdict(config):
    """Sub-dict config per il partial_state.json (schema = cell 50 notebook)."""
    return {
        'num_qubits': config.num_qubits, 'kernel_size': config.kernel_size,
        'num_conv_channels': config.num_conv_channels,
        'num_classes': config.num_classes, 'max_epochs': config.max_epochs,
        'batch_size': config.batch_size, 'lr': config.lr,
        'max_samples_per_class': config.max_samples_per_class,
        'backend_type': config.backend_type, 'quanv_padding': config.quanv_padding,
        'measure_qubit': config.measure_qubit,
        'num_stat_runs': config.num_stat_runs, 'base_seed': config.base_seed,
    }


def _save_partial(partial_path, config, results, completed_indices, failed_indices):
    """Atomic save (tmp + replace) of the partial state.
    Schema compatible with the final results.json: cp partial_state.json results.json
    works for replay.
    """
    partial = {
        'architecture': 'hybrid_qcnn_v1_partial',
        'config': _config_subdict(config),
        'results': [results[i] for i in sorted(completed_indices)],
        'completed_indices': sorted(completed_indices),
        'failed_indices': sorted(failed_indices),
        'n_completed': len(completed_indices),
        'n_total': config.num_stat_runs,
        'timestamp': time.time(),
    }
    tmp = partial_path + '.tmp'
    with open(tmp, 'w') as f:
        _json.dump(partial, f, indent=2, default=str)
    os.replace(tmp, partial_path)


def _save_final_results(config, results):
    """Saves the final results.json and predictions/*.csv (cell-50 notebook schema)."""
    save_path = os.path.join(config.output_dir, 'results.json')
    with open(save_path, 'w') as f:
        _json.dump({
            'architecture': 'hybrid_qcnn_v1',
            'config': _config_subdict(config),
            'results': results,
            'stats_summary': {},   # computed in the notebook §18
            'wilcoxon_results': {},  # computed in the notebook §18.7
        }, f, indent=2)
    print(f"\n✓ Results saved: {save_path}")

    pred_dir = os.path.join(config.output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    for r in results:
        csv_path = os.path.join(pred_dir, f"predictions_run{r['run_idx']:02d}_s{r['seed']}.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['item_idx', 'label', 'correct'])
            for i, (lab, corr) in enumerate(zip(r['val_labels_final'], r['val_correct_final'])):
                w.writerow([i, lab, corr])
    print(f"✓ Predictions per-run: {pred_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--num-runs', type=int, default=10, help='R (default 10)')
    p.add_argument('--output-dir', default='Output_QCNN_v1_multiseed')
    p.add_argument('--max-epochs', type=int, default=10)
    p.add_argument('--max-samples', type=int, default=100,
                   help='max samples per classe (default 100)')
    p.add_argument('--n-parallel-seeds', type=int, default=3,
                   help='worker process (S3 default 3)')
    p.add_argument('--aer-parallel', type=int, default=3,
                   help='aer_max_parallel_experiments (S3 default 3)')
    p.add_argument('--n-parallel-chunks', type=int, default=4,
                   help='PUB splitting K (default 4)')
    p.add_argument('--serial', action='store_true',
                   help='force serial loop (ignore ProcessPool)')
    p.add_argument('--smoke-test', action='store_true',
                   help='smoke test rapido: R=1, max_epochs=1, dataset sintetico')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"=" * 72)
    print(f"  QCNN R={args.num_runs} multi-seed batch (Wave K post-bench v3)")
    print(f"  Avvio: {datetime.now().isoformat()}")
    print(f"=" * 72)
    print(f"Qiskit {qiskit.__version__} | PyTorch {torch.__version__} | "
          f"Lightning {L.__version__} | Aer: {'✓' if HAS_AER else '✗'}")
    print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', '-')}")
    print()

    config = QCNNConfig(
        num_stat_runs=args.num_runs,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs if not args.smoke_test else 1,
        max_samples_per_class=args.max_samples if not args.smoke_test else 16,
        n_parallel_seeds=args.n_parallel_seeds,
        aer_max_parallel_experiments=args.aer_parallel,
        n_parallel_chunks=args.n_parallel_chunks,
        parallel_seeds=(not args.serial),
    )
    if args.smoke_test:
        config.num_stat_runs = 1
        config.train_dir = "_nonexistent_force_synthetic_"
        config.val_dir = "_nonexistent_force_synthetic_"

    print(f"Config:")
    print(f"  R = {config.num_stat_runs}, max_epochs = {config.max_epochs}, "
          f"max_samples/class = {config.max_samples_per_class}")
    print(f"  parallel_seeds = {config.parallel_seeds}, "
          f"n_parallel_seeds = {config.n_parallel_seeds}")
    print(f"  aer_max_parallel_experiments = {config.aer_max_parallel_experiments}, "
          f"n_parallel_chunks = {config.n_parallel_chunks}")
    print(f"  output_dir = {config.output_dir}")
    print()

    L.seed_everything(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    seeds = [config.base_seed + run_idx * 111 for run_idx in range(config.num_stat_runs)]
    partial_path = os.path.join(config.output_dir, 'partial_state.json')

    if not config.parallel_seeds:
        # ── Serial mode ──
        print(f"Mode: SERIAL ({config.num_stat_runs} seeds in sequence)\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bm = BackendManager(config)
        bm.initialize()
        dm = EuroSATDataModule(config)
        dm.setup()

        results = []
        completed = set()
        failed = set()
        t_start = time.time()
        for run_idx, seed in enumerate(seeds):
            try:
                r = run_single_training(config, bm, dm, device, seed, run_idx, verbose=True)
                results.append(r)
                completed.add(run_idx)
                # Incremental save: results is a list of successes only
                _save_partial(partial_path, config,
                              {i: results[i] for i in range(len(results))},
                              completed, failed)
                elapsed = (time.time() - t_start) / 60
                print(f"  [done {len(completed):2d}/{config.num_stat_runs}, "
                      f"{elapsed:.1f} min wallclock]")
            except Exception as e:
                failed.add(run_idx)
                print(f"  ⚠ Run {run_idx+1} FAILED: {type(e).__name__}: {e}")
        bm.close()
    else:
        # ── Parallel mode (S3 default) ──
        cfg_pickleable = _copy.copy(config)
        try:
            _pickle.dumps(cfg_pickleable)
        except Exception as e:
            raise RuntimeError(f"Config non pickleable: {e}.")

        ctx = _mp.get_context('spawn')

        n_blas = int(os.environ.get('OMP_NUM_THREADS', '1'))
        n_aer = config.aer_max_parallel_experiments
        print(f"Modalita: PARALLELA ({config.n_parallel_seeds} worker su "
              f"{config.num_stat_runs} seed)")
        print(f"  Thread BLAS/worker = {n_blas}; Aer parallel/worker = {n_aer}")
        print(f"  Tot thread attivi ~ {config.n_parallel_seeds * (n_blas + n_aer)}")
        print(f"  PUB splitting K = {config.n_parallel_chunks}")
        print(f"  Checkpoint: {partial_path}")
        print()

        results = [None] * config.num_stat_runs
        completed = set()
        failed = set()
        t_start = time.time()

        with ProcessPoolExecutor(max_workers=config.n_parallel_seeds,
                                 mp_context=ctx) as pool:
            futures = {
                pool.submit(_worker_run_one_seed, cfg_pickleable, run_idx, seed): run_idx
                for run_idx, seed in enumerate(seeds)
            }
            try:
                for f in as_completed(futures):
                    run_idx = futures[f]
                    try:
                        results[run_idx] = f.result()
                        completed.add(run_idx)
                        _save_partial(partial_path, config, results, completed, failed)
                        elapsed = (time.time() - t_start) / 60
                        print(f"  Run {run_idx+1:2d}/{config.num_stat_runs} OK  "
                              f"seed={seeds[run_idx]}  "
                              f"final_val_acc={results[run_idx]['final_val_acc']:.4f}  "
                              f"[done {len(completed):2d}/{config.num_stat_runs}, "
                              f"{elapsed:.1f} min wallclock]")
                    except Exception as e:
                        failed.add(run_idx)
                        _save_partial(partial_path, config, results, completed, failed)
                        print(f"  ⚠ Run {run_idx+1:2d}/{config.num_stat_runs} FAILED  "
                              f"seed={seeds[run_idx]}  "
                              f"{type(e).__name__}: {e}")
                        # NON rilanciare: gli altri future devono completare.
            except KeyboardInterrupt:
                print(f"\n  ⚠ Interruzione utente (Ctrl-C). "
                      f"Stato: {len(completed)} OK, {len(failed)} fail.")
                _save_partial(partial_path, config, results, completed, failed)
                raise

        # Compatta results rimuovendo i None
        results = [r for r in results if r is not None]

    # ── Final report ──
    total_min = (time.time() - t_start) / 60
    print(f"\n{'=' * 72}")
    print(f"  COMPLETED: {len(results)}/{config.num_stat_runs} runs "
          f"in {total_min:.1f} min ({total_min/60:.2f} h)")
    print(f"{'=' * 72}")

    if failed:
        print(f"\n⚠ WARNING: {len(failed)} runs FAILED -> indices {sorted(failed)}")
        print(f"  partial_state.json contains the {len(completed)} successful runs.")
        print(f"  To use the partials: cp {partial_path} "
              f"{os.path.join(config.output_dir, 'results.json')}")
        # Save results.json with the partials anyway - analysis still possible
        _save_final_results(config, results)
    else:
        _save_final_results(config, results)
        print(f"\n✓ Tutti i {config.num_stat_runs} run completati con successo.")
        print(f"  partial_state.json can be removed (results.json is the definitive data).")

    print(f"\nFine: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
