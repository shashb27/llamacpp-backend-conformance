#!/usr/bin/env python3
"""Generate paper figures from the v2 dataset + analyses.

Outputs vector PDFs to paper/figs/:
  fig-gaps.pdf     - near-tie logprob-gap distribution at first divergence
  fig-quant.pdf    - exact-match rate per backend x precision (core)
  fig-length.pdf   - agreement survival by generation position (n512 arm)
  fig-controls.pdf - control-pair agreement bars
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import load_records, token_logprob  # noqa: E402

FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

BACKENDS = ["cuda", "cuda13", "rocm", "vulkan"]
COLORS = {"cuda": "#76b900", "cuda13": "#4a7c59", "rocm": "#c7412b", "vulkan": "#8f6fdb"}


def fig_gaps():
    records = load_records("results/raw")
    groups = {}
    for (label, model, backend), rec in records.items():
        groups.setdefault((model, label), {})[backend] = rec
    gaps = []
    for (model, label), bks in groups.items():
        ref = bks.get("cpu")
        if not ref or ref.get("status") != "OK":
            continue
        rtoks = [p.get("tokens") or [] for p in ref.get("prompts", [])]
        for backend, rec in bks.items():
            if backend == "cpu" or rec.get("status") != "OK":
                continue
            btoks = [p.get("tokens") or [] for p in rec.get("prompts", [])]
            for i in range(min(len(btoks), len(rtoks))):
                a, b = btoks[i], rtoks[i]
                pos = None
                for k in range(min(len(a), len(b))):
                    if a[k].get("id") != b[k].get("id"):
                        pos = k
                        break
                if pos is None:
                    continue
                lp, in_top = token_logprob(b, pos, a[pos].get("id"))
                if lp is not None and b[pos].get("logprob") is not None:
                    gaps.append(b[pos]["logprob"] - lp)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(gaps, bins=80, range=(0, 1.0), color="#555577", edgecolor="white", linewidth=0.3)
    ax.axvline(sorted(gaps)[len(gaps) // 2], color="#c7412b", lw=1.2,
               label=f"median = {sorted(gaps)[len(gaps) // 2]:.3f} nats")
    ax.set_xlabel("logprob gap at first divergence (nats)")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig-gaps.pdf"))
    plt.close(fig)
    print(f"[figs] gaps: n={len(gaps)}")


def fig_quant():
    with open("results/analysis.json", encoding="utf-8") as fh:
        data = json.load(fh)
    quants = ["q4_k_m", "q6_k", "q8_0", "f16"]
    qdisp = {"q4_k_m": "Q4_K_M", "q6_k": "Q6_K", "q8_0": "Q8_0", "f16": "F16"}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.19
    for bi, backend in enumerate(BACKENDS):
        vals, xs = [], []
        for qi, q in enumerate(quants):
            key = f"{backend}/{q}" if q != "f16" else None
            if q == "f16":
                key = f"{backend}/f16" if f"{backend}/f16" in data["per_backend_quant"] else f"{backend}/fp16"
            v = data["per_backend_quant"].get(key, {}).get("gen_exact_pct")
            if v is not None:
                vals.append(v)
                xs.append(qi + (bi - 1.5) * width)
        ax.bar(xs, vals, width=width, color=COLORS[backend], label=backend)
    ax.set_xticks(range(len(quants)))
    ax.set_xticklabels([qdisp[q] for q in quants])
    ax.set_ylabel("exact match vs CPU (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncols=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig-quant.pdf"))
    plt.close(fig)
    print("[figs] quant done")


def fig_length():
    with open("results/analysis-2b-arms.json", encoding="utf-8") as fh:
        data = json.load(fh)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for backend in BACKENDS:
        surv = data["length_drift"]["survival"].get(backend)
        if not surv:
            continue
        xs = sorted(int(k) for k in surv.keys())
        ys = [surv[str(x)]["pct"] for x in xs]
        ax.plot(xs, ys, color=COLORS[backend], lw=1.6, label=backend)
    ax.set_xlabel("generation position (tokens)")
    ax.set_ylabel("prefix agreement with CPU (%)")
    ax.set_ylim(0, 105)
    ax.axvline(128, color="#999999", lw=0.8, ls=":")
    ax.text(132, 96, "core protocol cutoff", fontsize=7.5, color="#777777")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig-length.pdf"))
    plt.close(fig)
    print("[figs] length done")


def fig_controls():
    with open("results/analysis.json", encoding="utf-8") as fh:
        data = json.load(fh)
    rows = [
        ("Vulkan\\n7900 XTX pair", data["control_pairs"]["radeon-rx-7900-xtx-linux|radeon-rx-7900-xtx-2-linux|vulkan"]["cross_box_exact_pct"], "#8f6fdb"),
        ("CUDA\nPRO 6000 pair", data["control_pairs"]["rtx-pro-6000-blackwell-linux|rtx-pro-6000-blackwell-2-linux|cuda"]["cross_box_exact_pct"], "#76b900"),
        ("Vulkan\nPRO 6000 pair", data["control_pairs"]["rtx-pro-6000-blackwell-linux|rtx-pro-6000-blackwell-2-linux|vulkan"]["cross_box_exact_pct"], "#8f6fdb"),
        ("ROCm\n7900 XTX pair", data["control_pairs"]["radeon-rx-7900-xtx-linux|radeon-rx-7900-xtx-2-linux|rocm"]["cross_box_exact_pct"], "#c7412b"),
        ("CUDA vs CUDA13\nsame box", data["control_pairs"]["rtx-pro-6000-blackwell-linux|rtx-pro-6000-blackwell-linux|cuda-vs-cuda13"]["cross_box_exact_pct"], "#4a7c59"),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs = range(len(rows))
    vals = [r[1] for r in rows]
    ax.bar(xs, vals, color=[r[2] for r in rows], width=0.6)
    for x, v in zip(xs, vals):
        ax.text(x, v + 2, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("generation agreement (%)")
    ax.set_ylim(0, 115)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig-controls.pdf"))
    plt.close(fig)
    print("[figs] controls done")


def fig_matrix():
    """Heatmap: exact-match vs CPU per model file x backend, pooled over boxes."""
    import numpy as np
    records = load_records("results/raw")
    groups = {}
    for (label, model, backend), rec in records.items():
        groups.setdefault((model, label), {})[backend] = rec
    models = sorted({m for (m, l) in groups.keys()})
    backends = ["cpu", "cuda", "cuda13", "rocm", "vulkan"]
    agg = {}  # (model, backend) -> [match, n]
    for (model, label), bks in groups.items():
        ref = bks.get("cpu")
        if not ref or ref.get("status") != "OK":
            continue
        rg = [p.get("gen") or "" for p in ref.get("prompts", [])]
        for backend in backends[1:]:
            rec = bks.get(backend)
            if not rec or rec.get("status") != "OK":
                continue
            g = [p.get("gen") or "" for p in rec.get("prompts", [])]
            n = min(len(g), len(rg))
            m = sum(1 for i in range(n) if g[i] == rg[i])
            a = agg.setdefault((model, backend), [0, 0])
            a[0] += m
            a[1] += n
    grid = np.full((len(models), 4), np.nan)
    for (model, backend), (m, n) in agg.items():
        grid[models.index(model), backends.index(backend) - 1] = 100 * m / n
    fig, ax = plt.subplots(figsize=(5.8, 6.2))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(backends[1:], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    for i in range(len(models)):
        for j in range(4):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="black" if 15 < grid[i, j] < 85 else "white")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("exact match vs CPU (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig-matrix.pdf"))
    plt.close(fig)
    print("[figs] matrix done")


if __name__ == "__main__":
    fig_gaps()
    fig_quant()
    fig_length()
    fig_controls()
    fig_matrix()
