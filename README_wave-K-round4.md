# Wave K Round 4 — Pipeline Filippi-based per Cap.3 sec:results-stats

Revisione finale dell'infrastruttura multi-seed Wave K, basata sul codice
reale di Filippi (`hybrid_qcnn_v2_0_filippi_5.ipynb`) anziché sui notebook
standalone del Round 3.

## Setup effettivo (post-tuning Wave K)

Setup definitivo del Cap.3 sec:stats-significance, coerente con il run
CCNN R=10 già completato:

- `max_samples_per_class = 100` per classe (binary EuroSAT Forest vs AnnualCrop)
- Split train/val 80/20 → N_train=160, N_val=200
- `max_epochs = 10`
- `R = 10` run multi-seed con `seed = 42 + run_idx * 111`
- Ottimizzazioni quantum (solo QCNN): detach input grad (E), StatevectorEstimator (F)

## Notebook

### `hybrid_qcnn_multiseed_stats_v1.ipynb`
QCNN modello `C16-Q64` di Filippi (tesi magistrale, Pisa AA 2024/2025,
supervisori Morsch + Cappuccio).

Output: `Output_QCNN_v1_multiseed/results.json`.

### `classical_cnn_multiseed_stats_v1.ipynb`
Notebook gemello — CCNN matched-capacity ablation. Stessi 16/32/64 canali,
dropout 5%, profondità del QCNN; sostituisce `QuantumConvLayer(64,64)` con
`Conv2d(64,64,kernel=3,padding=1)` classico equivalente.

**Run effettivo CCNN (commit `1c27bc7`)**: R=10 seed, ~15 s/seed CPU, totale
~2.5 min. Risultati: `final_val_acc = 0.9835 ± 0.0047` con bootstrap 95% CI
sulla mean [0.9810, 0.9865] a N_val=200. Tutti e 10 i seed convergono entro
3-4 epoche al plateau ~98%.

Output: `Output_CCNN_v1_multiseed/results.json`.

## Schema seed (unico)
`seed = 42 + run_idx * 111`, R=10 → seeds {42, 153, 264, 375, 486, 597,
708, 819, 930, 1041}. Identico nei due notebook per il pairing del Wilcoxon
signed-rank test.

## Ottimizzazioni QCNN-only

- **E**: detach input grad nel `QuantumConvLayer.forward` → dimezza PUB
  backward (36 → 18 per W=9, n=9). Trade-off documentato in
  `sec:results-stats` del Cap.3 ("A note on the gradient flow through the
  quanvolutional layer").
- **F**: `backend_type` "aer" → "statevector" (StatevectorEstimator puro,
  più veloce di AerSimulator per 9 qubit ideali)

Speedup atteso QCNN: ~3-5× rispetto a Filippi v2.0 baseline.

## Test statistico
`scipy.stats.wilcoxon(method='exact', alternative='two-sided')` sui R=10
valori `final_val_acc` paired sul `run_idx`. Con R=10 il p-value minimo
raggiungibile (esatto) è 2/2^10 ≈ 0.002.

## Sebastianelli (2021)
Citato testualmente nel Cap.3 come riferimento bibliografico per il purely-
quantum reference model. Non incluso nel Wilcoxon paired perché non
disponiamo dei dati grezzi seed-per-seed.

## Storia delle revisioni
- Commit `816a1d4` (Round 4 v1): pipeline Filippi-based introdotta
- Commit `4903fff` (Round 4 v2): ottimizzazioni A+B+E+F (40 sample, 15 ep)
- Commit `1c27bc7` (Round 4 v3): fix plot_all_curves suptitle CCNN
- Commit attuale (Round 4 v4): config allineato al run effettivo 100/10

## Ottimizzazioni di parallelismo (commit successivo a 1681c1a)

Tre ottimizzazioni ortogonali per ridurre il tempo simulator del QCNN:

**Opzione 1 — Aer multi-thread**: `backend_type="aer"` con
`aer_max_parallel_experiments=4`. AerSimulator(method='statevector') con
`max_parallel_experiments=N` parallelizza i PUB bindings su N thread,
sostituendo lo StatevectorEstimator puro che era single-thread.

**Opzione 2 — BLAS multi-thread**: in cell 2, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` settati a
4 PRIMA degli import numpy/torch/qiskit. Velocizza le matrix-vector
operations sottostanti statevector simulation.

**Opzione 3 — ProcessPoolExecutor sui R seed**: cell 31 contiene branch
`if not config.parallel_seeds: ...` (default seriale) `else: ...` con
ProcessPoolExecutor + spawn context. Ogni worker ricrea
BackendManager + DataModule fresh (non sono pickleable).

### Tuning consigliato in base al numero di core fisici

Per un sistema con N_CPU core fisici:
- **Conservativo, primo run**: `parallel_seeds=False` (seriale).
  `OMP_NUM_THREADS=N_CPU`, `aer_max_parallel_experiments=N_CPU`.
- **Aggressivo, dopo aver validato il baseline**:
  `parallel_seeds=True`, `n_parallel_seeds=N_CPU//4`.
  `OMP_NUM_THREADS=N_CPU//n_parallel_seeds`,
  `aer_max_parallel_experiments=N_CPU//n_parallel_seeds`.
  Totale thread attivi: ~N_CPU.

Esempio su 8 core: `n_parallel_seeds=2, OMP_NUM_THREADS=4`.
Esempio su 16 core: `n_parallel_seeds=4, OMP_NUM_THREADS=4`.
