# Noise-model characterisation — `ibm_fez` (4-qubit study)

Single source of truth for the gate-error numbers quoted in the noise-resilience
section of Chapter 3.

## What the simulation actually applies

The noisy branch runs each ParallelQuanv sub-circuit on physical qubits
`[0, 1, 2, 3]` of `ibm_fez` (IBM Heron r2), set in
`noise_resilience_4q/run_4q.py` (`Config4q.noise_qubits = [0, 1, 2, 3]`) and not
overridden by `submit_4q_noisy_array.sbatch`. The noise these four qubits carry
in the calibration snapshot actually used (`ibm_fez_20260516`) is:

| quantity | value |
|---|---|
| `eps_sx` (avg over qubits 0–3) | ~0.08% |
| `eps_cz` (avg over couplings 0–1, 1–2, 2–3) | ~0.47% (0.93% / 0.27% / 0.22%) |
| `F_4q = (1-eps_sx)^17 (1-eps_cz)^6` | ~0.96 |
| per-pass infidelity | ~4% |

`17` `sx` and `6` `cz` gates per sub-circuit after transpilation in the device
basis (`sx`, `cz`, `rz`, `x`, `id`).

## Device-wide averages (do not describe the sub-circuit)

The device-wide averages over all 156 qubits / 176 couplings are `eps_sx ~ 0.90%`
and `eps_cz ~ 2.80%`, giving `F_4q ~ 0.73`. These are inflated by a small number
of poorly calibrated qubits and couplings (e.g. qubits 17, 72 with sx error
~66%; couplings 27–28, 32–33, 71–72, … with cz error ~80%) that the sub-circuit
never touches. An earlier draft reported these device-wide figures as if they
were the per-circuit noise; they have been replaced by the restricted values
above.

## Reproduce

```
python extract_ibm_fez_noise.py --snapshot /path/to/ibm_fez_20260516.pkl
```

The script is self-verifying (asserts both the restricted and the device-wide
values) and writes `ibm_fez_noise_4q.json`. The 15.7 MB snapshot `.pkl` is not
committed; it lives in `$WORK_DIR/noise_models/ibm_fez_20260516.pkl` on the
cluster (see the sbatch script).

Average gate error is computed as `1 - average_gate_fidelity(SuperOp(error))`,
the average gate infidelity of the calibration error channel vs the identity.
Cross-check: the device-wide readout-error average reproduced this way (0.0400)
matches the snapshot JSON (`readout_error_avg = 0.0374`).
