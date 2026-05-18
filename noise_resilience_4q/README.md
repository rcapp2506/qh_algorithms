# Noise resilience: 4q paired multi-seed under a calibrated IBM Heron r2 noise model

Code supporting **Sec. 3.4.10** of *"New Perspectives on Quantum Technologies"*
(R. Cappuccio, PhD thesis, University of Siena, 2026), the cross-architecture
noise study that compares two paired branches of a 4-qubit reduction of the
ParallelQuanv ansatz under matched seeds and hyperparameters.

The two branches differ only in the simulator backend:

| Branch | Backend | Where it runs |
|---|---|---|
| Noiseless | `AerSimulator(method='statevector')` | Local workstation |
| Noisy | `AerSimulator(method='density_matrix')` + IBM Heron r2 NoiseModel restricted to 4 qubits | Cluster node (Magellano / equivalent) |

Both branches share the same architecture (one ParallelQuanv block of `K=4`
parallel sub-circuits, each on `n=4` qubits, kernel size `2x2`, stride `3`,
500 shots per measurement), the same classical backbone, the same dataset
(EuroSAT, binary task `AnnualCrop` vs `Forest`, 100 images per class), and
the same hyperparameters (Adam, `lr=1e-3`, weight decay `1e-4`, batch 16,
10 epochs). Pairing is exact: the same `R=10` random seeds (range `[42,51]`)
determine initial weights, data ordering and Aer sampler RNG in both
branches.

## Files

| File | Where it runs | What it does |
|---|---|---|
| `parallel_quanv_sim.py` | Cluster + workstation | ParallelQuanvCircuit + sampler-based QuantumEnginePSSim, density_matrix-noisy or statevector-noiseless. |
| `run_4q.py` | Cluster + workstation | CLI: trains 1 seed; selects backend via `--backend sim_noisy` or `--backend sim_noiseless`. |
| `submit_4q_noisy_array.sbatch` | Cluster (SLURM) | Array job 10x that runs `run_4q.py --backend sim_noisy` once per seed. |
| `run_4q_noiseless_local.ipynb` | Local workstation | Serial R=10 loop that calls `run_4q.py --backend sim_noiseless` once per seed. |
| `wilcoxon_cross_arch.py` | Local workstation | Paired Wilcoxon signed-rank tests, Wilson 95% CIs, optional violin + mean±CI plot. |

The key design choice is that **`run_4q.py` is a single script** invoked by
both the SLURM array and the notebook with different flags. Same code path,
same architecture, same hyperparameters — only the backend behind the
sampler differs. This guarantees a like-for-like comparison.

## Requirements

Python 3.14 environment with:

- `qiskit==2.2`, `qiskit-aer>=0.15`, `qiskit-ibm-runtime>=0.27`
- `torch`, `torchvision`, `pillow`
- `numpy`, `scipy`, `matplotlib`
- `jupyter` (for the local notebook)

A pickled NoiseModel snapshot from the target IBM device is required for the
noisy branch; the snapshot used in the thesis (`ibm_fez_20260516.pkl`) was
fetched via `qiskit-ibm-runtime` and is shipped separately because of its
size.

The EuroSAT dataset must be split into `dataset/training/` and
`dataset/validation/` with one subdirectory per class. The split used in
the thesis is the same as that of the 9-qubit noiseless multi-seed run
documented in `HQCNN.ipynb`.

## Workflow

### Step 1 — Stage the files

On the cluster:

```bash
scp run_4q.py submit_4q_noisy_array.sbatch parallel_quanv_sim.py \
    user@cluster:/path/to/Q-CONV/code/
mv /path/to/Q-CONV/code/submit_4q_noisy_array.sbatch \
   /path/to/Q-CONV/slurm/
```

On the local workstation (which will run the noiseless branch and the
post-hoc analysis):

```bash
mkdir noise_resilience_4q && cd noise_resilience_4q
cp /path/to/run_4q.py /path/to/parallel_quanv_sim.py \
   /path/to/run_4q_noiseless_local.ipynb \
   /path/to/wilcoxon_cross_arch.py \
   /path/to/ibm_fez_<DATE>.pkl .
```

### Step 2 — Run the noisy branch on the cluster (R=10 array job)

```bash
cd /path/to/Q-CONV/slurm
sbatch submit_4q_noisy_array.sbatch
```

Wall-time per seed: ~90 minutes on a single Magellano node with 8 Aer
worker processes. The array submits 10 tasks in parallel; total wall-time
is dominated by the slowest task.

Each task writes `results_seed_<NNN>.json` into the output directory.

### Step 3 — Run the noiseless branch locally

Open `run_4q_noiseless_local.ipynb` in Jupyter and run all cells. The
notebook iterates over the same 10 seeds in serial; total wall-time ~5
hours on an M-series or i9 laptop (~30 min per seed).

Output: `./results_4q_noiseless/results_seed_<NNN>.json`.

### Step 4 — Download the cluster results

```bash
scp -r user@cluster:/path/to/Q-CONV/output/results_4q_noisy ./
```

### Step 5 — Cross-architecture analysis

```bash
python wilcoxon_cross_arch.py \
    --runs 4q_noisy:./results_4q_noisy 4q_noiseless:./results_4q_noiseless \
    --out  results_wilcoxon.json \
    --plot results_wilcoxon.png
```

Output:

- A formatted table of Wilson 95% CIs per architecture
- A formatted table of paired Wilcoxon p-values for each architecture pair
- (Optional) JSON dump of all numerical results
- (Optional) PNG with a violin plot + mean±CI per architecture

## Reproducing the thesis numbers

The numbers reported in Sec. 3.4.10 of the thesis correspond to
`base_seed=42`, R=10 (seed range `[42,51]`), 10 epochs, 100 images per
class in both training and validation, NoiseModel snapshot
`ibm_fez_20260516.pkl`.

Expected output of `wilcoxon_cross_arch.py` on the thesis data:

```
4q_noiseless: mean = 0.9640 ± 0.0147, Wilson 95% CI [0.957, 0.971]
4q_noisy    : mean = 0.9620 ± 0.0136, Wilson 95% CI [0.955, 0.969]
paired Wilcoxon (4q_noiseless vs 4q_noisy):
    p = 0.0625, Cohen's d_z = +0.50, mean Δ = +0.0020
```

## Citation

If you use this code, please cite:

```
@phdthesis{Cappuccio2026Thesis,
    author = {Cappuccio, Roberto},
    title  = {New Perspectives on Quantum Technologies: Progress on Quantum Sensing and Quantum Computation},
    school = {Universit\`a di Siena},
    year   = {2026},
}
```

## License

MIT — see top-level `LICENSE` in the `qh_algorithms` repository.
