#!/usr/bin/env python3
"""Generate the paper's data figures from results/audit-v2.json.

Usage:  python harness/make_figs.py
Outputs paper/figs/fig-audit.pdf, fig-precision.pdf, fig-gaps.pdf.
Every figure is regenerable from the public audit file - no other input.
"""
import json
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(ROOT, "results", "audit-v2.json"), encoding="utf-8") as fh:
    AUDIT = json.load(fh)

GPU_COLOR = "#2b6cb0"
FB_COLOR = "#c53030"


def fig_audit():
    """Figure: device-truth audit - wall-time ratio per configuration."""
    devices = AUDIT["devices"]
    rows = []
    for key, d in devices.items():
        label, backend = key.split("|")
        rows.append((f"{label}  ({backend})", d["median_ratio"], d["device"]))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [GPU_COLOR if r[2] == "gpu" else FB_COLOR for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hlines(range(len(rows)), 0, vals, color="#dddddd", linewidth=0.8)
    ax.scatter(vals, range(len(rows)), c=colors, s=28, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Median wall-time ratio vs. same-host CPU build")
    ax.set_xlim(0, 1.1)
    ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    ax.scatter([], [], c=GPU_COLOR, s=28, label="GPU-classified")
    ax.scatter([], [], c=FB_COLOR, s=28, label="CPU-fallback")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-audit.pdf"))
    plt.close(fig)


def fig_precision():
    """Figure: byte-exact agreement by precision and population."""
    gpu = AUDIT["per_quant"]["POP-GPU"]
    bcpu = AUDIT["per_quant"]["POP-BUILDCPU"]
    quants = ["q4_k_m", "q8_0", "f16"]
    qlabels = ["Q4_K_M", "Q8_0", "F16/FP16"]
    gpu_v = [gpu[q]["exact_pct"] for q in quants]
    bcpu_v = [bcpu[q]["exact_pct"] for q in quants]
    # cross-class host-pair ranges (results/evidence 06; see paper table)
    cc_lo = [13.7, 35.3, 90.0]
    cc_hi = [20.1, 39.2, 94.1]

    x = range(len(quants))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.bar([i - w / 2 for i in x], gpu_v, w, color=GPU_COLOR,
           label="GPU vs same-host CPU")
    ax.bar([i + w / 2 for i in x], bcpu_v, w, color="#63b3ed",
           label="GPU builds' CPU impl. vs standalone CPU")
    ax.plot(x, cc_hi, "_", color="#444444", markersize=14, markeredgewidth=1.6)
    ax.plot(x, cc_lo, "_", color="#444444", markersize=14, markeredgewidth=1.6)
    for i in x:
        ax.vlines(i, cc_lo[i], cc_hi[i], color="#444444", linewidth=1.0)
    ax.plot(x, [100, 100, 100], linestyle="none",
            label="same-class CPU host pairs (100%)")
    ax.axhline(100, color="#888888", linewidth=0.8, linestyle="--")
    for i, v in enumerate(gpu_v):
        ax.text(i - w / 2, v + 2.5, f"{v:.1f}", ha="center", fontsize=7)
    for i, v in enumerate(bcpu_v):
        ax.text(i + w / 2, v + 2.5, f"{v:.1f}", ha="center", fontsize=7)
    for i in x:
        ax.text(i + 0.02, (cc_lo[i] + cc_hi[i]) / 2 + 2.5,
                f"{cc_lo[i]:.1f}\u2013{cc_hi[i]:.1f}", ha="left", fontsize=6.5,
                color="#444444")
    ax.set_xticks(list(x))
    ax.set_xticklabels(qlabels)
    ax.set_ylabel("Byte-exact agreement (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-precision.pdf"))
    plt.close(fig)


def fig_gaps():
    """Figure: near-tie gap quantiles at first-divergence positions."""
    ref = AUDIT["gap_ref_quantiles"]
    bk = AUDIT["gap_backend_quantiles"]
    stats = [("median", "Median"), ("p90", "p90"), ("p99", "p99"),
             ("max", "Max")]
    ref_v = [ref[s[0]] for s in stats]
    bk_v = [bk[s[0]] for s in stats]

    x = range(len(stats))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.bar([i - w / 2 for i in x], ref_v, w, color=GPU_COLOR,
           label="Reference's preference for its own choice")
    ax.bar([i + w / 2 for i in x], bk_v, w, color="#63b3ed",
           label="Backend's preference for its choice")
    for i, v in enumerate(ref_v):
        ax.text(i - w / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=7)
    for i, v in enumerate(bk_v):
        ax.text(i + w / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s[1] for s in stats])
    ax.set_ylabel("Logprob gap (nats)")
    ax.set_ylim(0, max(ref_v + bk_v) * 1.18)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-gaps.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_audit()
    fig_precision()
    fig_gaps()
    print("wrote fig-audit.pdf, fig-precision.pdf, fig-gaps.pdf to", OUT)
