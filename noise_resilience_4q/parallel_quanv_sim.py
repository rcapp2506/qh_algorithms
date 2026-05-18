"""parallel_quanv_sim.py — Porting of ParallelQuanv from Emerald to Aer noisy

Drop-in replacement for the quantum block of the
`hybrid_qcnn_v4_3_0_emerald.ipynb` notebook when running on simulator with
a NoiseModel deserialised from a snapshot (e.g. ibm_fez on a cluster node).

Mathematical equivalence
------------------------
The original `ParallelQuanvCircuit` (Emerald) builds a single monolithic
circuit of `K*n` qubits with K disjoint blocks. On 54-qubit IQM hardware
this is efficient (one job submission per K patches).

On a density_matrix simulator the size grows as 2^(2*K*n):
  - K=4, n=4 -> 2^32 entries = 70 GiB of monolithic density matrix
  - K=2, n=4 -> 2^16 entries = 1 MiB (manageable but minimal)

However the K blocks are *product-state* (no two-qubit cross-block gate),
so rho_global = rho_0 (x) rho_1 (x) ... (x) rho_{K-1} exactly.
For any block-local observable, <Z>_k depends only on rho_k.

Operational consequence: simulating 1 monolithic K*n-qubit circuit or
K independent n-qubit sub-circuits produces identical expectation values
(modulo independent shot noise). The porting consists in replacing
G monolithic PUBs with G*K independent sub-PUBs, distributed on Aer
worker `max_parallel_experiments`. Cost:
  - 1 monolithic PUB density_matrix K*n-qubit: O(4^(K*n)) ops/gate
  - K sub PUBs density_matrix n-qubit: K * O(4^n) ops/gate
  - Saving factor: 4^(K*n) / (K * 4^n) = 4^((K-1)*n) / K

For K=4, n=4: saving = 4^12 / 4 = ~4.2 million. From infeasible to instant.

Single-block architecture (identical to the monolithic Emerald version)
-----------------------------------------------------------------------
For each block k (offset = 0 in the sub-circuit):
    qc.ry(x_k[i], i)        # i=0..n-1   angle encoding
    qc.h(i)                 # i=0..n-1   superposition
    qc.cx(i, i+1)           # i=0..n-2   forward entangling chain
    qc.rz(w[i], i)          # i=0..n-1   variational layer (shared w)
    qc.cx(i, i+1)           # i=n-2..0   reverse entangling chain
    qc.measure(mo, 0)                    # 1 Z measurement

Weights `w` (size n) are SHARED across the K blocks (weight-sharing,
convolutional property). Inputs `x_k` are distinct per block (patch k).

API compatibility
-----------------
Exposes the same interfaces as `ParallelFilippiCircuit` + `QuantumEnginePS`
of the v4.3.0 notebook:
    cb = ParallelQuanvCircuit(config)
    engine = QuantumEnginePSSim(config, backend_manager)
    ev = engine.forward_only(patches, w)         # patches: (N, n), w: (n,)
    fwd, gw, gx = engine.step(patches, w, compute_gx=True)

`BackendManagerSimNoisy` has the same shape as `BackendManager` but accepts
`noise_snapshot_path` and configures `AerSimulator(method='density_matrix',
noise_model=...)`. It replaces the original `_setup_sim()` which used
noiseless statevector.

Threading
---------
IMPORTANT: the sbatch script MUST export:
    OMP_NUM_THREADS=8
    MKL_NUM_THREADS=1        # not 8! documented oversubscription
    OPENBLAS_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS=1
before invoking Python. See smoke_parallel_quanv.sbatch.
"""

from __future__ import annotations
import math
import pickle
import time
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


# =============================================================================
# Circuit
# =============================================================================

class ParallelQuanvCircuit:
    """K sub-circuiti 4-qubit indipendenti, weight-sharing, sampler-based.

    Exact mathematical equivalent of the monolithic `ParallelFilippiCircuit`
    of the v4.3.0 notebook for any block-local observable
    (see module docstring).
    """

    def __init__(self, config):
        self.n = config.num_qubits
        self.K = config.num_parallel_blocks
        self.num_weights = config.num_weights
        self.mo = config.measure_qubit

        # Un solo template di sub-circuito, riusato K volte con bind diversi.
        # The input parameters are ONE single ParameterVector of n (not K different
        # as in the monolithic case), because each sub-circuit is bound separately
        # con un patch diverso.
        self.x = ParameterVector('x', self.n)
        self.w = ParameterVector('w', self.num_weights)
        self.sub_circuit = self._build_sub_circuit()

        # Monolithic reference, NOT used in forward/step (would be too
        # large for density_matrix). Exposed for debug/equivalence verification.
        # Costruzione lazy: si crea solo se richiesto esplicitamente.
        self._monolithic = None

        print(f"  ParallelQuanvCircuit: K={self.K} sub-circuits di {self.n}q "
              f"(equivalente logico {self.K * self.n}q), "
              f"depth_sub={self.sub_circuit.depth()}, "
              f"ops_sub={dict(self.sub_circuit.count_ops())}")

    def _build_sub_circuit(self) -> QuantumCircuit:
        """Costruisce un singolo blocco n-qubit con 1 classical register 1-bit.

        Identical to the k-th block of the ParallelFilippiCircuit (v4.3.0 notebook)
        ma senza offset di registro.
        """
        qr = QuantumRegister(self.n, 'q')
        cr = ClassicalRegister(1, 'c0')
        qc = QuantumCircuit(qr, cr)
        for i in range(self.n):
            qc.ry(self.x[i], i)
        for i in range(self.n):
            qc.h(i)
        for i in range(self.n - 1):
            qc.cx(i, i + 1)
        for i in range(self.n):
            qc.rz(self.w[i], i)
        for i in range(self.n - 2, -1, -1):
            qc.cx(i, i + 1)
        qc.measure(self.mo, 0)
        return qc

    def bind(self, x_vals: np.ndarray, w_vals: np.ndarray) -> QuantumCircuit:
        """Bind un sub-circuit con specifici input x e weights w."""
        params = {p: float(x_vals[i]) for i, p in enumerate(self.x)}
        params.update({p: float(w_vals[i]) for i, p in enumerate(self.w)})
        return self.sub_circuit.assign_parameters(params)

    # --- Monolithic helper (debug only, not used in training) ----------
    def build_monolithic_for_debug(self) -> QuantumCircuit:
        """Costruisce la versione monolitica K*n-qubit (solo per benchmark di
        equivalence, NOT for density_matrix simulation at K=4). Usable only
        a K piccoli (K<=2) o per statevector noiseless."""
        if self._monolithic is not None:
            return self._monolithic
        x_per_k = [ParameterVector(f'x{k}', self.n) for k in range(self.K)]
        qr = QuantumRegister(self.K * self.n, 'q')
        cr = ClassicalRegister(self.K, 'c')
        qc = QuantumCircuit(qr, cr)
        for k in range(self.K):
            off = k * self.n
            for i in range(self.n):
                qc.ry(x_per_k[k][i], off + i)
            for i in range(self.n):
                qc.h(off + i)
            for i in range(self.n - 1):
                qc.cx(off + i, off + i + 1)
            for i in range(self.n):
                qc.rz(self.w[i], off + i)
            for i in range(self.n - 2, -1, -1):
                qc.cx(off + i, off + i + 1)
            qc.measure(off + self.mo, k)
        self._monolithic = qc
        self._monolithic_x_per_k = x_per_k
        return qc


# =============================================================================
# Backend
# =============================================================================

class BackendManagerSimNoisy:
    """Versione di BackendManager per Aer density_matrix con NoiseModel.

    Replaces `_setup_sim()` of the v4.3.0 notebook (which used
    statevector noiseless). Configurazione mirata per un nodo HPC con
    OMP=8 and BLAS=1 (see sbatch).

    Parametri rilevanti:
      - noise_snapshot_path: path to the pickle of the NoiseModel (e.g. ibm_fez)
      - aer_max_parallel_experiments: numero worker Aer paralleli (=8 su Levante)
      - noise_qubits: list of chip qubit IDs to which the NoiseModel is restricted
        (default = range(num_qubits), i.e. qubits 0..n-1 of the chip).

    **NoiseModel restriction** (performance-critical)
    --------------------------------------------------------
    Lo snapshot ibm_fez_20260516.pkl contiene la calibrazione di 156 qubit
    fisici. Se passato così com'è a AerSimulator, ogni `run()` paga un
    overhead di ~4 secondi indipendente dal batch size, perché Aer ricostruisce
    the error structures for all 156 qubits. Furthermore the per-circuit cost
    is ~30x larger than the minimal synthetic NoiseModel.

    Soluzione: restringere il NoiseModel ai soli qubit effettivamente usati
    by the sub-circuit (default: qubits 0..n-1). The qubits of our sub-circuit
    are Aer virtual qubits (they have no direct physical correspondence in
    simulation), so we map virtual 0..n-1 -> physical 0..n-1 of the chip.

    Misurato sul nostro sandbox: 61 ms/circ (full) → 2.9 ms/circ (ridotto), 22× speedup.

    For a more "realistic" mapping of the hardware pattern (qubits with
    fidelity migliore), passare `noise_qubits=[3, 7, 12, 22]` o simile.
    """

    def __init__(self, config, noise_snapshot_path: str,
                 aer_max_parallel_experiments: int = 8,
                 noise_qubits: Optional[list] = None):
        self.config = config
        self.noise_snapshot_path = noise_snapshot_path
        self.aer_parallel = aer_max_parallel_experiments
        # Default: the first `num_qubits` of the chip
        self.noise_qubits = (list(noise_qubits) if noise_qubits is not None
                             else list(range(config.num_qubits)))
        if len(self.noise_qubits) != config.num_qubits:
            raise ValueError(
                f"noise_qubits ha {len(self.noise_qubits)} elementi ma "
                f"il sub-circuit ha {config.num_qubits} qubit. Devono coincidere.")
        self.backend = None
        self.sampler = None
        self.backend_name = "?"
        self.num_backend_qubits = 0
        self.noise_model: Optional[NoiseModel] = None
        self.noise_model_full: Optional[NoiseModel] = None  # tenuto per debug
        self.rng = np.random.default_rng(config.seed)

    def initialize(self):
        # Load NoiseModel ONCE (cached on the instance)
        print(f"  Carico NoiseModel da {self.noise_snapshot_path} ...")
        t0 = time.time()
        with open(self.noise_snapshot_path, 'rb') as f:
            snapshot = pickle.load(f)
        if isinstance(snapshot, NoiseModel):
            self.noise_model_full = snapshot
        elif isinstance(snapshot, dict):
            if 'noise_model' in snapshot and isinstance(
                    snapshot['noise_model'], NoiseModel):
                self.noise_model_full = snapshot['noise_model']
            elif 'noise_model_dict' in snapshot:
                self.noise_model_full = NoiseModel.from_dict(
                    snapshot['noise_model_dict'])
            else:
                self.noise_model_full = NoiseModel.from_dict(snapshot)
        else:
            raise TypeError(
                f"Snapshot type {type(snapshot).__name__} not recognised. "
                f"Atteso NoiseModel o dict.")
        n_phys = self._count_qubits_in_noise(self.noise_model_full)
        print(f"  NoiseModel full caricato in {time.time()-t0:.2f}s "
              f"(≥{n_phys} qubit fisici)")

        # Restringi il NoiseModel ai soli qubit di interesse.
        # Measured speedup: ~22x over the full model on the 4-qubit sub-circuit.
        t0 = time.time()
        self.noise_model = self._restrict_noise_model(
            self.noise_model_full, self.noise_qubits)
        n_err = self._count_errors(self.noise_model)
        print(f"  NoiseModel ristretto a qubit {self.noise_qubits} "
              f"in {time.time()-t0:.2f}s ({n_err} quantum errors)")
        print(f"  basis_gates: {sorted(self.noise_model.basis_gates)}")

        # AerSimulator density_matrix con NoiseModel ristretto.
        # max_parallel_experiments=8 distribuisce i PUB sui worker.
        # max_parallel_threads=8 cappa il pool totale (con BLAS=1 nelle env
        # vars dello sbatch, significa 8 worker single-thread BLAS).
        self.backend = AerSimulator(
            method='density_matrix',
            noise_model=self.noise_model,
            max_parallel_experiments=self.aer_parallel,
            max_parallel_threads=self.aer_parallel,
        )
        # For the config.num_parallel_blocks check in the notebook __init__,
        # esponiamo un num_qubits "logico" largo (non c'è un chip reale).
        self.num_backend_qubits = max(64, self.config.total_qubits)
        self.backend_name = (
            f"AerSimulator(density_matrix, noise={self.noise_snapshot_path.split('/')[-1]}, "
            f"parallel={self.aer_parallel})")
        self.sampler = AerSamplerWrapperFixed(self.backend, self.config.shots)
        print(f"  {self.backend_name} | K={self.config.num_parallel_blocks}")

    @staticmethod
    def _count_qubits_in_noise(nm: NoiseModel) -> int:
        """Stima il numero di qubit per cui il NoiseModel ha calibrazioni."""
        qubits = set()
        for d in (getattr(nm, '_local_quantum_errors', {}),
                  getattr(nm, '_local_readout_errors', {})):
            for instr_or_qbts in d.values() if isinstance(d, dict) else []:
                if isinstance(instr_or_qbts, dict):
                    for qbts in instr_or_qbts.keys():
                        qubits.update(qbts)
                else:
                    # _local_readout_errors: keys are directly tuples
                    pass
        # Per _local_readout_errors la struttura è {tuple: ReadoutError}
        for qbts in getattr(nm, '_local_readout_errors', {}).keys():
            qubits.update(qbts)
        return max(qubits) + 1 if qubits else 0

    @staticmethod
    def _count_errors(nm: NoiseModel) -> int:
        """Total number of quantum errors registered in the NoiseModel."""
        n = 0
        for instr_dict in getattr(nm, '_local_quantum_errors', {}).values():
            n += len(instr_dict) if isinstance(instr_dict, dict) else 0
        n += len(getattr(nm, '_local_readout_errors', {}))
        return n

    @staticmethod
    def _restrict_noise_model(nm_full: NoiseModel,
                              qubit_ids: list) -> NoiseModel:
        """Extracts a sub-NoiseModel from the `qubit_ids` of the full NoiseModel only.

        Rimappa i qubit fisici qubit_ids[i] → virtual i (0, 1, ..., len-1)
        in the sub-NoiseModel. Therefore a sub-circuit using virtual qubits
        0..n-1 sees the noise of physical qubits qubit_ids[0..n-1].

        Cruciale per performance: il NoiseModel ibm_fez full ha 156 qubit
        of calibration; restricting it to the 4 qubits of the sub-circuit yields
        ~22× speedup per ms/circ e azzeramento dell'overhead per-run.
        """
        qubit_set = set(qubit_ids)
        phys_to_virt = {phys: virt for virt, phys in enumerate(qubit_ids)}
        nm_sub = NoiseModel(basis_gates=list(nm_full.basis_gates))

        # Quantum errors per gate
        local_q = getattr(nm_full, '_local_quantum_errors', {})
        for instr_name, qbts_to_err in local_q.items():
            if not isinstance(qbts_to_err, dict):
                continue
            for qbts, err in qbts_to_err.items():
                if all(q in qubit_set for q in qbts):
                    remapped = [phys_to_virt[q] for q in qbts]
                    nm_sub.add_quantum_error(err, [instr_name], remapped,
                                             warnings=False)

        # Readout errors
        local_r = getattr(nm_full, '_local_readout_errors', {})
        for qbts, err in local_r.items():
            if all(q in qubit_set for q in qbts):
                remapped = [phys_to_virt[q] for q in qbts]
                nm_sub.add_readout_error(err, remapped, warnings=False)

        return nm_sub

    def transpile_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile using the basis gates of the NoiseModel.

        Filters out non-gate instructions (if_else, delay, measure, reset) that the snapshot
        ibm_fez includes in basis_gates but which are not decomposable gates.
        The physical gates of Heron r2 are: cz, sx, rz, x, id.
        """
        PHYSICAL_GATES = {'cz', 'sx', 'rz', 'x', 'id', 'rzz', 'rzx'}
        all_basis = list(self.noise_model.basis_gates)
        basis = [g for g in all_basis if g in PHYSICAL_GATES]
        if not basis:
            raise RuntimeError(
                f"Nessun gate fisico in NoiseModel.basis_gates={all_basis}")
        return transpile(
            circuit,
            basis_gates=basis,
            optimization_level=self.config.optimization_level,
        )

    def close(self):
        print("Backend chiuso")


class AerSamplerWrapperFixed:
    """Sampler wrapper per AerSimulator(density_matrix) con noise_model.

    Identical in spirit to `AerSamplerWrapper` of the v4.3.0 notebook, but
    exposes `data.c0` with the single-1-bit-register convention used
    dai sub-circuiti di `ParallelQuanvCircuit`. Niente parsing di
    multi-register: ogni sub-circuit ha 1 solo registro classico da 1 bit.
    """

    def __init__(self, backend, default_shots: int = 500):
        self.backend = backend
        self.default_shots = default_shots

    def run(self, circuits, shots: Optional[int] = None):
        if shots is None:
            shots = self.default_shots
        if not isinstance(circuits, list):
            circuits = [circuits]
        job = self.backend.run(circuits, shots=shots)
        return _AerJobResult(job, circuits)


class _AerJobResult:
    def __init__(self, aer_job, circuits):
        self.aer_job = aer_job
        self.circuits = circuits

    def result(self):
        r = self.aer_job.result()
        return [_AerSubCircuitResult(r.get_counts(i))
                for i in range(len(self.circuits))]


class _AerSubCircuitResult:
    """Result of a single 4-qubit sub-circuit with 1 classical bit.

    `data.c0.get_counts()` restituisce {'0': n_zero, '1': n_one}.
    """
    def __init__(self, counts: dict):
        self.data = _AerSubCircuitData(counts)


class _AerSubCircuitData:
    def __init__(self, counts: dict):
        # counts may be on 1 bit or more (if measure_all was used).
        # Normalizziamo a {'0': n, '1': n} sul bit 0 (l'unico misurato).
        if not counts:
            self._counts = {'0': 0, '1': 0}
        else:
            first_key = next(iter(counts))
            if len(first_key.replace(' ', '')) == 1:
                # Già 1 bit
                self._counts = {'0': counts.get('0', 0),
                                '1': counts.get('1', 0)}
            else:
                # Marginalizza sul bit rightmost (bit 0 di Qiskit)
                acc = {'0': 0, '1': 0}
                for bs, cnt in counts.items():
                    bs_clean = bs.replace(' ', '')
                    bit = bs_clean[-1]
                    acc[bit] = acc.get(bit, 0) + cnt
                self._counts = acc
        self.c0 = _CountsAccessor(self._counts)


class _CountsAccessor:
    def __init__(self, counts):
        self._counts = counts

    def get_counts(self):
        return self._counts


# =============================================================================
# Engine (sampler-based, K-aware su sub-PUB indipendenti)
# =============================================================================

class QuantumEnginePSSim:
    """Parameter Shift engine per backend simulator con K sub-PUB indipendenti.

    API-compatible with `QuantumEnginePS` of the v4.3.0 notebook:
    espone `forward_only(patches, w)` e `step(patches, w, compute_gx=...)`.

    Differenza con il `QuantumEnginePS` Emerald:
      - submits G*K sub-circuits (G groups x K blocks per group) as a
        PUB list to the sampler, instead of G monolithic circuits of K blocks
      - Aer parallelizza nativamente sui worker
      - ev extraction: 1 value per sub-result (data.c0), instead of K
        values per monolithic result
    """

    SHIFT = np.pi / 2

    def __init__(self, config, bm):
        self.config = config
        self.K = config.num_parallel_blocks
        self.n = config.num_qubits
        self.nw = config.num_weights
        self.mb = config.max_bindings_per_job
        self.cb = ParallelQuanvCircuit(config)

        # Transpile UNA volta sul template di sub-circuit (cache).
        # The transpiled sub-circuit is then used with assign_parameters() to
        # ogni bind: Aer evita ri-transpile inutili.
        print("  Transpiling sub-circuit template...")
        t0 = time.time()
        # Salviamo il template parametrico transpilato; il bind successivo
        # sostituisce solo i valori numerici.
        self._sub_transpiled = bm.transpile_circuit(self.cb.sub_circuit)
        # Ricostruiamo l'oggetto cb con sub_circuit transpilato per i bind:
        # tecnicamente la cb.sub_circuit originale è in basis logico,
        # whereas what we hand to the sampler must be in the noise basis.
        self.cb.sub_circuit = self._sub_transpiled
        dt = time.time() - t0
        print(f"  Transpile done in {dt:.2f}s | "
              f"depth_t={self._sub_transpiled.depth()}, "
              f"ops_t={dict(self._sub_transpiled.count_ops())}")

        self.sampler = bm.sampler
        # Stats
        self.jobs = 0
        self.circs = 0
        self.qpu = 0.0

    # --- Internals -----------------------------------------------------------

    def _bind(self, patches: np.ndarray, w: np.ndarray):
        """Produce G*K sub-circuiti bindati.

        patches: array shape (N, n) — N patches da n features ciascuna
        w: array shape (nw,) — pesi quantistici condivisi

        Ritorna:
          bound : list di G*K QuantumCircuit bindati (ordine: g=0 k=0..K-1,
                  g=1 k=0..K-1, ...)
          N     : numero originale di patches
          G     : number of logical groups (= ceil(N/K))
        """
        N = patches.shape[0]
        G = math.ceil(N / self.K)
        pad = np.zeros((G * self.K, self.n), dtype=np.float64)
        pad[:N] = patches
        bound = []
        for g in range(G):
            for k in range(self.K):
                x_k = pad[g * self.K + k]
                bound.append(self.cb.bind(x_k, w))
        return bound, N, G

    def _exec(self, bcs):
        """Submits the sub-circuits to the sampler in chunks of `mb`."""
        res = []
        t0 = time.time()
        for s in range(0, len(bcs), self.mb):
            b = bcs[s:min(s + self.mb, len(bcs))]
            res.extend(self.sampler.run(b, shots=self.config.shots).result())
            self.jobs += 1
            self.circs += len(b)
        self.qpu += time.time() - t0
        return res

    def _evs(self, res, N: int) -> np.ndarray:
        """Extracts N expectation values from the G*K sub-results.

        Ogni sub-result ha 1 misura sul qubit `mo` → counts {'0': n0, '1': n1}.
        <Z> = (n0 - n1) / (n0 + n1).
        """
        ev = []
        for r in res:
            c = r.data.c0.get_counts()
            t = sum(c.values())
            ev.append((c.get('0', 0) - c.get('1', 0)) / t if t else 0.0)
        return np.array(ev[:N]).reshape(-1, 1)

    # --- Public API (compatible with QuantumEnginePS of the notebook) --------

    def forward_only(self, patches: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Solo forward, niente gradienti (per validation)."""
        bound, N, _ = self._bind(patches, w)
        return self._evs(self._exec(bound), N)

    def step(self, patches: np.ndarray, w: np.ndarray,
             compute_gx: bool = True):
        """A single mini-batch step: forward + gw + (optional) gx.

        Ritorna:
          fwd : array (N, 1)
          gw  : array (nw, N, 1)
          gx  : array (n, N, 1) oppure None se compute_gx=False

        If freeze_backbone is active (so patches have no gradient
        with respect to the backbone parameters), pass compute_gx=False
        risparmia 2n PUB-batch (-47% sui passi di gradiente).
        """
        S = self.SHIFT

        # Forward
        bound_fwd, N, _ = self._bind(patches, w)
        fwd = self._evs(self._exec(bound_fwd), N)

        # Gradiente rispetto ai pesi w (parameter-shift)
        gw = np.zeros((self.nw, N, 1))
        for j in range(self.nw):
            wp = w.copy(); wp[j] += S
            wm = w.copy(); wm[j] -= S
            gw_p = self._evs(self._exec(self._bind(patches, wp)[0]), N)
            gw_m = self._evs(self._exec(self._bind(patches, wm)[0]), N)
            gw[j] = (gw_p - gw_m) / 2

        # Gradiente rispetto agli input x (parameter-shift per qubit)
        if compute_gx:
            gx = np.zeros((self.n, N, 1))
            for i in range(self.n):
                xp = patches.copy(); xp[:, i] += S
                xm = patches.copy(); xm[:, i] -= S
                gx_p = self._evs(self._exec(self._bind(xp, w)[0]), N)
                gx_m = self._evs(self._exec(self._bind(xm, w)[0]), N)
                gx[i] = (gx_p - gx_m) / 2
        else:
            gx = None

        return fwd, gw, gx


# =============================================================================
# Smoke import-time check (no side effects)
# =============================================================================

if __name__ == "__main__":
    print("parallel_quanv_sim.py — modulo importabile, niente da eseguire.")
    print("Per il bench: python bench_aer_4q.py")
    print("To integrate in the notebook: see the trailing comment of this file.")
