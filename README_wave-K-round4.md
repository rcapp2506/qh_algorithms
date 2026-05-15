# Wave K Round 4 — Pipeline Filippi-based per Cap.3 sec:results-stats

Questa è la **revisione finale** dell'infrastruttura multi-seed Wave K, basata
sul codice reale di Filippi (`hybrid_qcnn_v2_0_filippi_5.ipynb`) anziché sui
notebook autonomi del Round 3 (`HQCNN.ipynb`, `esa_modello.ipynb`,
`qiskit_one_q.ipynb`, `multirun.py`).

## Notebook (Round 4)

### `hybrid_qcnn_multiseed_stats_v1.ipynb`
QCNN modello `C16-Q64` di Filippi (tesi magistrale, Pisa AA 2024/2025,
supervisori Morsch + Cappuccio). EuroSAT 2 classi (coerente con Filippi
§6.3.1), 40 epoche per run, R=10 multi-seed.

Sezioni statistiche:
- **§18.5**: Wilson 95% CI + bootstrap CI per ogni run (single-arch)
- **§18.7**: Wilcoxon signed-rank paired QCNN-vs-CCNN (cross-arch)
- **§19**: Salvataggio JSON formato esteso + predizioni CSV per-item
- **§19.5**: Replay post-hoc dei plot da JSON salvati (cambia stile senza
  rifare il training)

Output: `Output_QCNN_v1_multiseed/results.json`.

### `classical_cnn_multiseed_stats_v1.ipynb`
Notebook gemello — CCNN matched-capacity ablation. Stessi 16/32/64 canali,
dropout 5%, profondità del QCNN; sostituisce `QuantumConvLayer(64,64)` con
`Conv2d(64,64,kernel=3,padding=1)` classico equivalente.

Distinto dal LeNet-5 baseline 6/16 originale di Filippi, che soffriva di
asimmetria di capacità rispetto al QCNN. Stesso schema seed del QCNN.

Output: `Output_CCNN_v1_multiseed/results.json`.

## Schema seed (unico)
`seed = 42 + run_idx * 111`, R=10 → seeds {42, 153, 264, 375, 486, 597, 708,
819, 930, 1041}. Lo schema deve essere **identico** nei due notebook per
il pairing del Wilcoxon signed-rank test sul `run_idx`.

## Differenza dal Round 3
Il Round 3 (commit precedente) implementava `multirun.py` + 3 notebook
standalone (HQCNN/qiskit_one_q/esa_modello). Questo Round 4 li **sostituisce
funzionalmente** ma li **lascia in tree** come storia: la pipeline di
riferimento per la tesi è quella del Round 4 (Filippi-based), che riproduce
fedelmente il modello `C16-Q64` di Filippi anziché 3 architetture
indipendenti.

## Test statistico
`scipy.stats.wilcoxon(method='exact', alternative='two-sided')` sui R=10
valori `final_val_acc` paired sul `run_idx`. Con R=10 il p-value minimo
raggiungibile (esatto) è 2/2^10 ≈ 0.002.

## Sebastianelli (2021)
Citato testualmente nel Cap.3 come riferimento bibliografico per il purely-
quantum reference model. Non incluso nel Wilcoxon paired perché non
disponiamo dei dati grezzi seed-per-seed.
