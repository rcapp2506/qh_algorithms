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
