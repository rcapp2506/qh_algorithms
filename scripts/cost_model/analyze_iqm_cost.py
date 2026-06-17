"""Cost-model single source of truth (Chapter 3, Sec. cost_model / Phase E).

Loads the real IQM Resonance job records of the Phase E fine-tuning campaign
(IQM Emerald, 2026-05-02), reproduces every cost-model number reported in the
manuscript, exports a tidy per-job CSV, and regenerates the two cost-model
figures *from the real data* (no fabricated scatter).

Source data
-----------
data/iqm_cost_model/iqm_jobs_raw_dump.json
    Real job metadata dumped from the IQM Resonance jobs API
    (GET /webapp/api/jobs/paginated). Each record carries server-side
    timestamps, billed ``runtime_seconds`` and dashboard ``credit_cost``.

Cost-model dataset = all completed Emerald jobs created on 2026-05-02:
    195 jobs = 183 production (180 forward/weight-shift at N=28
               + 3 end-of-epoch validation at N=273)
             + 12 small-N calibration/exploration probes (N in {1,5,20,50}).

Reproduced quantities (asserted)
--------------------------------
per-job fit   runtime_seconds(N) = A + B*N,  A=1.49 s, B=0.225 s/circ, R^2=0.992
billing rule  credit_cost = ceil(runtime_seconds) * 0.75   (holds 195/195)
billing window = execution + init/teardown overhead (see job-anatomy figure):
    N=28 : 7.41 s = 6.16 (execution) + 1.25 (init/teardown), credit 6.00
    N=273: 63.38 s = 56.73 (execution) + 6.65 (init/teardown), credit 48.00
production billed ~ 1284 credits over ~1581 s of QPU runtime.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "iqm_cost_model" / "iqm_jobs_raw_dump.json"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

NOMINAL_RATE = 0.75  # credits / s (PAYG)


def parse_iso(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def delta(ts, a, b):
    ta, tb = parse_iso(ts.get(a)), parse_iso(ts.get(b))
    return (tb - ta).total_seconds() if (ta and tb) else None


def load_cost_model_jobs():
    """Return the 195 completed Emerald jobs of the 2026-05-02 campaign."""
    records = json.loads(DATA.read_text())
    jobs = [
        r for r in records
        if r["qc_alias"] == "emerald"
        and (r["created"] or "").startswith("2026-05-02")
        and r["runtime_seconds"] is not None
    ]
    jobs.sort(key=lambda r: r["created"])
    return jobs


def per_job_phases(job):
    ts = job["timestamps"]
    execution_s = delta(ts, "execution_start", "execution_end")
    billing_s = float(job["runtime_seconds"])
    return {
        "id": job["id"],
        "N": job["circuit_count"],
        "shots": job["shots_count"],
        "created": job["created"],
        "validation_s": delta(ts, "validation_started", "validation_ended"),
        "compile_s": delta(ts, "compile_start", "compile_end"),
        "queue_s": delta(ts, "compile_end", "execution_start"),
        "execution_s": execution_s,
        "postproc_s": delta(ts, "post_processing_started", "post_processing_ended"),
        "billing_window_s": billing_s,
        "init_teardown_s": (billing_s - execution_s) if execution_s is not None else None,
        "credit_cost": float(job["credit_cost"]),
    }


def linfit(x, y):
    """Least-squares y = A + B x; return A, B, R2, seA, seB."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    B, A = np.polyfit(x, y, 1)
    pred = A + B * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    R2 = 1.0 - ss_res / ss_tot
    n = len(x)
    s_err = math.sqrt(ss_res / (n - 2))
    Sxx = np.sum((x - x.mean()) ** 2)
    seB = s_err / math.sqrt(Sxx)
    seA = s_err * math.sqrt(1.0 / n + x.mean() ** 2 / Sxx)
    return A, B, R2, seA, seB


def median_of(rows, key, N=None):
    vals = [r[key] for r in rows if (N is None or r["N"] == N) and r[key] is not None]
    return float(np.median(vals)) if vals else None


# ----------------------------------------------------------------------
def main():
    jobs = load_cost_model_jobs()
    rows = [per_job_phases(j) for j in jobs]
    n = len(rows)
    print(f"Cost-model dataset: {n} completed Emerald jobs (2026-05-02)")

    from collections import Counter
    dist = dict(sorted(Counter(r["N"] for r in rows).items()))
    print(f"  circuit-count distribution: {dist}")

    # --- per-job fit runtime_seconds = A + B N ------------------------
    N = [r["N"] for r in rows]
    T = [r["billing_window_s"] for r in rows]
    A, B, R2, seA, seB = linfit(N, T)
    print(f"\n[per-job fit, n={n}]")
    print(f"  runtime_seconds(N) = ({A:.2f} +/- {seA:.2f}) s "
          f"+ ({B:.3f} +/- {seB:.3f}) s/circ * N,  R^2 = {R2:.3f}")

    # --- billing rule credit = ceil(runtime)*0.75 ---------------------
    ok = sum(1 for r in rows
             if abs(r["credit_cost"] - math.ceil(r["billing_window_s"]) * NOMINAL_RATE) < 1e-6)
    print(f"\n[billing rule] credit = ceil(runtime)*{NOMINAL_RATE}: holds {ok}/{n} jobs")

    # --- phase decomposition / billing window -------------------------
    decomp = {}
    for Nv in (28, 273):
        decomp[Nv] = {
            "n": sum(1 for r in rows if r["N"] == Nv),
            "validation": median_of(rows, "validation_s", Nv),
            "compile": median_of(rows, "compile_s", Nv),
            "queue": median_of(rows, "queue_s", Nv),
            "execution": median_of(rows, "execution_s", Nv),
            "postproc": median_of(rows, "postproc_s", Nv),
            "billing_window": median_of(rows, "billing_window_s", Nv),
            "init_teardown": median_of(rows, "init_teardown_s", Nv),
            "credit": median_of(rows, "credit_cost", Nv),
        }
        d = decomp[Nv]
        print(f"\n[N={Nv}] ({d['n']} jobs, medians)")
        print(f"  validation {d['validation']:.2f} | compile {d['compile']:.2f} | "
              f"queue {d['queue']:.2f} | execution {d['execution']:.2f}")
        print(f"  billing window {d['billing_window']:.2f} s "
              f"= execution {d['execution']:.2f} + init/teardown {d['init_teardown']:.2f}")
        print(f"  credit = ceil({d['billing_window']:.2f})*{NOMINAL_RATE} = {d['credit']:.2f}")

    # --- totals: production vs exploratory ----------------------------
    prod = [r for r in rows if r["N"] in (28, 273)]
    expl = [r for r in rows if r["N"] not in (28, 273)]
    prod_credits = sum(r["credit_cost"] for r in prod)
    prod_runtime = sum(r["billing_window_s"] for r in prod)
    print(f"\n[production] {len(prod)} jobs: {prod_credits:.1f} credits, {prod_runtime:.1f} s")
    # exploratory pre-fine-tuning batch (estimate execution times): 14:58
    batch = [r for r in expl if r["created"][11:16] == "14:58"]
    print(f"[exploratory probes] {len(expl)} jobs, {sum(r['credit_cost'] for r in expl):.1f} credits "
          f"(of which the 14:58 timing-estimation batch: "
          f"{sum(r['credit_cost'] for r in batch):.1f} credits)")

    # --- assertions (manuscript values) -------------------------------
    assert n == 195, f"expected 195 jobs, got {n}"
    assert abs(A - 1.49) < 0.03, f"intercept A={A:.3f} != 1.49"
    assert abs(B - 0.225) < 0.002, f"slope B={B:.4f} != 0.225"
    assert R2 > 0.99, f"R^2={R2:.4f} < 0.99"
    assert ok == n, f"billing rule fails on {n - ok} jobs"
    assert abs(decomp[28]["billing_window"] - 7.41) < 0.1
    assert abs(decomp[273]["billing_window"] - 63.38) < 0.2
    print("\nAll assertions passed.")

    # --- export tidy per-job CSV --------------------------------------
    csv_path = DATA.parent / "iqm_cost_model_per_job.csv"
    fields = ["id", "N", "shots", "created", "validation_s", "compile_s",
              "queue_s", "execution_s", "postproc_s", "billing_window_s",
              "init_teardown_s", "credit_cost"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"Wrote {csv_path.relative_to(REPO)}")

    fit = dict(A=A, B=B, R2=R2, seA=seA, seB=seB, n=n)
    make_cost_validation_figure(rows, fit, decomp)
    make_job_anatomy_figure(decomp)
    return rows, fit, decomp


# ----------------------------------------------------------------------
def make_cost_validation_figure(rows, fit, decomp):
    import matplotlib.pyplot as plt

    Ns = np.array([r["N"] for r in rows], float)
    Ts = np.array([r["billing_window_s"] for r in rows], float)
    A, B, R2 = fit["A"], fit["B"], fit["R2"]
    resid = Ts - (A + B * Ns)
    sd = resid.std(ddof=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) real per-job scatter + fit
    ax = axes[0]
    ax.scatter(Ns, Ts, s=22, color="#4a90d9", alpha=0.55, edgecolors="none",
               label=f"IQM jobs (n={len(rows)})")
    xg = np.linspace(0, 280, 200)
    ax.plot(xg, A + B * xg, color="#d62728", lw=2.2,
            label=r"fit: $\tau_{\rm job}=A+B\,N$")
    ax.fill_between(xg, A + B * xg - 2 * sd, A + B * xg + 2 * sd,
                    color="#d62728", alpha=0.12, label=fr"$\pm2\sigma$ ({2*sd:.2f} s)")
    ax.text(0.04, 0.92,
            fr"$A={A:.2f}$ s, $B={B:.3f}$ s/circ, $R^2={R2:.3f}$",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#d62728", alpha=0.9))
    ax.set_xlabel("Number of bound circuits per job, $N$", fontsize=11)
    ax.set_ylabel("Billed QPU runtime  [s]", fontsize=11)
    ax.set_title("(a) Per-job billed runtime vs circuit count", fontsize=12, pad=10)
    ax.set_xlim(-10, 285)
    ax.set_ylim(0, 72)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9.5)

    # (b) phase decomposition with correct billing window
    ax = axes[1]
    ax.set_ylim(0, 95)
    order = ["validation", "compile", "queue", "execution"]
    colors = {"validation": "#a6cee3", "compile": "#7fbf7b",
              "queue": "#fdbf6f", "execution": "#ff7f0e"}
    labels = {"validation": "validate", "compile": "compile",
              "queue": "queue (wait)", "execution": "execution"}
    groups = [(0, 28, "$N=28$\n(180 jobs)"), (1, 273, "$N=273$\n(3 jobs)")]
    for x, Nv, _ in groups:
        d = decomp[Nv]
        bottom = 0.0
        for ph in order:
            h = d[ph]
            ax.bar(x, h, bottom=bottom, width=0.55, color=colors[ph],
                   edgecolor="white", linewidth=0.5,
                   label=labels[ph] if x == 0 else None)
            if h >= 2.0:
                ax.text(x, bottom + h / 2, f"{h:.1f} s", ha="center", va="center",
                        fontsize=10, color="white" if ph == "execution" else "black")
            bottom += h
    for x, Nv, _ in groups:
        d = decomp[Nv]
        wall = d["validation"] + d["compile"] + d["queue"] + d["execution"] + (d["postproc"] or 0)
        ann = (f"billed: {d['billing_window']:.1f} s\n"
               f"  (exec {d['execution']:.1f} + ovh {d['init_teardown']:.1f})\n"
               f"credit: {d['credit']:.0f}")
        ax.text(x, wall + 3, ann, ha="center", va="bottom", fontsize=9.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#555555", linewidth=1.0, alpha=0.97))
    ax.set_xticks([g[0] for g in groups])
    ax.set_xticklabels([g[2] for g in groups], fontsize=10.5)
    ax.set_ylabel("Time  [s]", fontsize=11)
    ax.set_title("(b) Phase decomposition (server-side timestamps)", fontsize=12, pad=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.tight_layout()
    out = FIG_DIR / "cost_model_validation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO)}")


def make_job_anatomy_figure(decomp):
    """Anatomy of a single IQM job: wall-time phases vs the billed window.
    Grounded on the N=28 median and the N=273 validation job 019de9c2."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    COL = {"validation": "#cfe2f3", "compile": "#3a78bf", "queue": "#f7c895",
           "execution": "#e8741f", "postproc": "#a7e0a7",
           "billing": "#f4c542", "ceil_extra": "#9b7d4e"}
    panels = []
    for Nv, title in [(28, "(a) Job with 28 circuits (median of 180 Phase-E jobs)"),
                      (273, "(b) Job with 273 circuits (end-of-epoch validation)")]:
        d = decomp[Nv]
        panels.append(dict(
            title=title,
            wall=[("validation", d["validation"]), ("compile", d["compile"]),
                  ("queue", d["queue"]), ("execution", d["execution"]),
                  ("postproc", d["postproc"] or 0.0)],
            billing_start=d["validation"] + d["compile"] + d["queue"],
            execution=d["execution"], billing_window=d["billing_window"],
            init_teardown=d["init_teardown"], credit=d["credit"],
        ))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6))
    for ax, P in zip(axes, panels):
        x = 0.0
        for ph, h in P["wall"]:
            ax.barh(1.0, h, left=x, height=0.5, color=COL[ph], edgecolor="white")
            if h >= 1.0:
                ax.text(x + h / 2, 1.0, f"{ph}\n{h:.2f}s", ha="center", va="center",
                        fontsize=8, color="white" if ph in ("compile", "execution") else "black")
            x += h
        wall_total = x
        # billing window (= execution + init/teardown), starts at execution start
        bs = P["billing_start"]
        ax.barh(0.3, P["execution"], left=bs, height=0.32, color=COL["billing"],
                edgecolor="black", hatch="//")
        ax.barh(0.3, P["init_teardown"], left=bs + P["execution"], height=0.32,
                color=COL["ceil_extra"], edgecolor="black", hatch="\\\\")
        ax.text(bs + P["billing_window"] / 2, 0.3,
                f"IQM billing window {P['billing_window']:.2f}s  ->  "
                f"ceil = {math.ceil(P['billing_window'])}s x 0.75 = {P['credit']:.2f} cr",
                ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.set_xlim(0, max(wall_total, bs + P["billing_window"]) * 1.02)
        ax.set_ylim(-0.1, 1.5)
        ax.set_yticks([0.3, 1.0])
        ax.set_yticklabels(["billed", "wall"], fontsize=9)
        ax.set_title(P["title"], fontsize=11)
        ax.set_xlabel("time since job received [s]", fontsize=9)
        ax.grid(alpha=0.2, axis="x")
    handles = [mpatches.Patch(color=COL[k], label=k) for k in
               ["validation", "compile", "queue", "execution", "postproc"]]
    handles += [mpatches.Patch(facecolor=COL["billing"], hatch="//", edgecolor="black",
                               label="billed = execution"),
                mpatches.Patch(facecolor=COL["ceil_extra"], hatch="\\\\", edgecolor="black",
                               label="billed = init/teardown")]
    axes[0].legend(handles=handles, loc="upper right", fontsize=7.5, ncol=2, frameon=True)
    fig.tight_layout()
    out = FIG_DIR / "iqm_job_anatomy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
