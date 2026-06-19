#!/usr/bin/env python3
"""
Single source of truth for the gate-error characterisation of the 4-qubit
noise-resilience study (Sec. "Noise resilience under a calibrated IBM Heron r2
noise model" in Chapter 3).

The noisy branch of that study runs each ParallelQuanv sub-circuit on physical
qubits [0, 1, 2, 3] of the IBM Heron r2 device `ibm_fez` (see
noise_resilience_4q/run_4q.py: Config4q.noise_qubits = [0, 1, 2, 3]). This
script extracts, from the calibration snapshot actually used, the average gate
errors ON THOSE FOUR QUBITS and the resulting cumulative circuit fidelity.

It also reports the DEVICE-WIDE averages, which are substantially larger because
they are inflated by a small number of poorly calibrated qubits/couplings
(several with gate errors above 50%). Those device-wide values (eps_sx ~ 0.90%,
eps_cz ~ 2.80%, F_4q ~ 0.73) were reported in an earlier draft of the
manuscript; they do NOT describe the noise the sub-circuit experiences and have
been replaced by the restricted values computed here.

Average gate error of a channel is 1 - average_gate_fidelity(SuperOp(error)),
i.e. the average gate infidelity of the calibration error channel relative to
the identity.

Usage:
    python extract_ibm_fez_noise.py [--snapshot /path/to/ibm_fez_20260516.pkl]
                                    [--out ibm_fez_noise_4q.json]
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SuperOp, average_gate_fidelity

SUBCIRCUIT_QUBITS = [0, 1, 2, 3]
SUBCIRCUIT_COUPLINGS = [(0, 1), (1, 2), (2, 3)]
N_SX = 17  # sx gates per sub-circuit after transpilation in the device basis
N_CZ = 6   # cz gates per sub-circuit (3 couplings traversed forward and back)

# Expected values (the assertions below lock the manuscript text to this code).
EXP_SX_4Q = 0.0008   # ~0.08%
EXP_CZ_4Q = 0.0047   # ~0.47%
EXP_F4Q = 0.96
EXP_SX_DEVICE = 0.0090  # ~0.90%  (device-wide, inflated by broken qubits)
EXP_CZ_DEVICE = 0.0280  # ~2.80%  (device-wide, inflated by broken couplings)


def gate_error(qerror) -> float:
    """Average gate infidelity of a calibration error channel vs the identity."""
    return 1.0 - average_gate_fidelity(SuperOp(qerror))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="ibm_fez_20260516.pkl")
    ap.add_argument("--out", default=str(Path(__file__).with_name("ibm_fez_noise_4q.json")))
    args = ap.parse_args()

    nm = pickle.load(open(args.snapshot, "rb"))
    lqe = nm._local_quantum_errors

    sx = {q[0]: gate_error(qe) for q, qe in lqe["sx"].items()}
    cz = {tuple(sorted(q)): gate_error(qe) for q, qe in lqe["cz"].items()}

    # restricted to the four qubits the sub-circuit runs on
    sx_4q = {q: sx[q] for q in SUBCIRCUIT_QUBITS}
    cz_4q = {c: cz[c] for c in SUBCIRCUIT_COUPLINGS}
    eps_sx = float(np.mean(list(sx_4q.values())))
    eps_cz = float(np.mean(list(cz_4q.values())))
    f4q = float((1 - eps_sx) ** N_SX * (1 - eps_cz) ** N_CZ)

    # device-wide (origin of the earlier 0.90% / 2.80% figures)
    eps_sx_dev = float(np.mean(list(sx.values())))
    eps_cz_dev = float(np.mean(list(cz.values())))
    f4q_dev = float((1 - eps_sx_dev) ** N_SX * (1 - eps_cz_dev) ** N_CZ)

    result = {
        "snapshot": "ibm_fez_20260516",
        "subcircuit_qubits": SUBCIRCUIT_QUBITS,
        "n_sx": N_SX,
        "n_cz": N_CZ,
        "restricted_to_subcircuit_qubits": {
            "eps_sx_avg": eps_sx,
            "eps_cz_avg": eps_cz,
            "eps_cz_per_coupling": {f"{a}-{b}": cz_4q[(a, b)] for (a, b) in SUBCIRCUIT_COUPLINGS},
            "F_4q": f4q,
            "infidelity": 1 - f4q,
        },
        "device_wide": {
            "eps_sx_avg": eps_sx_dev,
            "eps_cz_avg": eps_cz_dev,
            "F_4q": f4q_dev,
            "infidelity": 1 - f4q_dev,
            "note": "inflated by a few qubits/couplings with gate errors > 50%; not the noise seen by the sub-circuit",
        },
    }

    print(f"restricted (qubits {SUBCIRCUIT_QUBITS}):")
    print(f"  eps_sx = {eps_sx*100:.3f}%   eps_cz = {eps_cz*100:.3f}%   "
          f"(0-1={cz_4q[(0,1)]*100:.2f}%, 1-2={cz_4q[(1,2)]*100:.2f}%, 2-3={cz_4q[(2,3)]*100:.2f}%)")
    print(f"  F_4q = (1-eps_sx)^{N_SX} (1-eps_cz)^{N_CZ} = {f4q:.4f}  -> infidelity {(1-f4q)*100:.1f}%")
    print(f"device-wide:")
    print(f"  eps_sx = {eps_sx_dev*100:.3f}%   eps_cz = {eps_cz_dev*100:.3f}%   F_4q = {f4q_dev:.3f}")

    # lock manuscript numbers to this extraction
    assert abs(eps_sx - EXP_SX_4Q) < 0.0003, eps_sx
    assert abs(eps_cz - EXP_CZ_4Q) < 0.0010, eps_cz
    assert abs(f4q - EXP_F4Q) < 0.01, f4q
    assert abs(eps_sx_dev - EXP_SX_DEVICE) < 0.0010, eps_sx_dev
    assert abs(eps_cz_dev - EXP_CZ_DEVICE) < 0.0020, eps_cz_dev
    print("\nall assertions passed")

    json.dump(result, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
