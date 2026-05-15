# Wave K Round 4 — Pipeline Filippi-based per Cap.3 sec:results-stats

Revisione finale dell'infrastruttura multi-seed Wave K, basata sul codice
reale di Filippi (`hybrid_qcnn_v2_0_filippi_5.ipynb`) anziché sui notebook
standalone del Round 3.

## Notebook (Round 4 + ottimizzazioni Wave K)

### `hybrid_qcnn_multiseed_stats_v1.ipynb`
QCNN modello `C16-Q64` di Filippi (tesi magistrale, Pisa AA 2024/2025,
supervisori Morsch + Cappuccio). EuroSAT 2 classi (coerente con Filippi
§6.3.1), R=10 multi-seed.

**Ottimizzazioni Wave K applicate** (vs Filippi v2.0 originale):
- **A**: `max_epochs` 40 → 15 (plateau Filippi ~10 ep + 50% buffer)
- **B**: `max_samples_per_class` 100 → 40 (binario satura presto, coerente
  con il regime di `sec:hw_validation` del Cap.3)
- **E**: detach input grad nel `QuantumConvLayer.forward` → dimezza il
  numero di PUB per backward (36 → 18 per W=9, n=9); il trunk classico a
  monte riceve gradient solo via classification loss, non attraverso la
  Jacobiana parameter-shift del quanv. Trade-off documentato nel
  manoscritto Cap.3 sec:results-stats.
- **F**: `backend_type` "aer" → "statevector" (StatevectorEstimator puro,
  più veloce di AerSimulator per 9 qubit ideali)

**Speedup atteso** sui tempi originali di Filippi v2.0:
- A: 2.7× (40→15 epoche)
- B: 2.5× (100→40 sample)
- E: 1.5× (18/36 = 0.5× backward, forward inalterato → ~1.5× totale)
- F: 2-3× (StatevectorEstimator vs AerSimulator per 9 qubit)
- **Cumulato: ~15-20× speedup** rispetto a Filippi v2.0 baseline.

Stima tempo/seed: da ~10 h a ~30-40 min su CPU (verificare al primo run).

Sezioni statistiche:
- **§18.5**: Wilson 95% CI + bootstrap CI per ogni run (single-arch)
- **§18.7**: Wilcoxon signed-rank paired QCNN-vs-CCNN (cross-arch)
- **§19**: Salvataggio JSON formato esteso + predizioni CSV per-item
- **§19.5**: Replay post-hoc dei plot da JSON salvati

Output: `Output_QCNN_v1_multiseed/results.json`.

### `classical_cnn_multiseed_stats_v1.ipynb`
Notebook gemello — CCNN matched-capacity ablation. Stessi 16/32/64 canali,
dropout 5%, profondità del QCNN; sostituisce `QuantumConvLayer(64,64)` con
`Conv2d(64,64,kernel=3,padding=1)` classico equivalente.

**Ottimizzazioni Wave K (A, B)** applicate per coerenza statistica con
QCNN. Niente E ed F (non c'è quantum stack). Il CCNN è veloce in se' e non
richiede ottimizzazioni aggressive: stima tempo/seed ~5-10 min su CPU.

Output: `Output_CCNN_v1_multiseed/results.json`.

## Schema seed (unico)
`seed = 42 + run_idx * 111`, R=10 → seeds {42, 153, 264, 375, 486, 597,
708, 819, 930, 1041}. Lo schema deve essere **identico** nei due notebook
per il pairing del Wilcoxon signed-rank test sul `run_idx`.

## Differenza dal Round 3
Il Round 3 (commit 3f97ea0) implementava `multirun.py` + 3 notebook
standalone (HQCNN/qiskit_one_q/esa_modello). Questo Round 4 li sostituisce
funzionalmente con la pipeline Filippi-based ma li lascia in tree come
storia. La pipeline di riferimento per la tesi è Round 4.

## Test statistico
`scipy.stats.wilcoxon(method='exact', alternative='two-sided')` sui R=10
valori `final_val_acc` paired sul `run_idx`. Con R=10 il p-value minimo
raggiungibile (esatto) è 2/2^10 ≈ 0.002.

## Sebastianelli (2021)
Citato testualmente nel Cap.3 come riferimento bibliografico per il purely-
quantum reference model. Non incluso nel Wilcoxon paired perché non
disponiamo dei dati grezzi seed-per-seed.
