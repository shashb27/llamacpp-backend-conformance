#!/usr/bin/env python3
"""Cross-backend conformance analysis over the v2 dataset.

Computes, per backend family and per model/precision:
  - generation exact-match rates vs the CPU reference (per label x model)
  - near-tie mechanism: at each first-divergence token, the logprob gap
    between the CPU reference's argmax token and the backend's chosen token
    (from the CPU record's top-5 logprob set)
  - perplexity deltas vs CPU (the ppl-blindness axis)
  - control-pair cross-box agreement (identical GPUs, identical drivers)
  - per-category divergence rates
  - within-backend repeat stability (where repeats > 1)

Usage:
  python harness/analyze.py --raw results/raw --out results/analysis.json
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

BACKEND_ORDER = ["cpu", "cuda", "cuda13", "rocm", "sycl", "vulkan"]
QUANT_OF = {}


def quant_of(model):
    for q in ("fp16", "f16", "q8_0", "q6_k", "q4_k_m"):
        if model.endswith(q):
            return q
    return "?"


def load_records(raw_dir):
    records = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        label, backend, model = parts
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        records[(label, model, backend)] = rec
    return records


def token_logprob(cpu_tokens, pos, token_id):
    """Logprob the CPU reference assigns to token_id at generation position pos.

    Returns (logprob, in_top5). Falls back to the 5th entry's logprob as a
    lower bound when the token is outside the recorded top-5.
    """
    if pos >= len(cpu_tokens):
        return None, False
    tok = cpu_tokens[pos]
    if tok.get("id") == token_id:
        return tok.get("logprob"), True
    for t in (tok.get("top") or []):
        if t.get("id") == token_id:
            return t.get("logprob"), True
    floor = None
    top = tok.get("top") or []
    if top:
        floor = min(t.get("logprob") for t in top)
    return floor, False  # lower bound: gap is at least this large


def analyze(records):
    groups = defaultdict(dict)
    for (label, model, backend), rec in records.items():
        groups[(model, label)][backend] = rec

    per_backend = defaultdict(lambda: {"cells": 0, "exact": 0, "prompts": 0,
                                       "match": 0, "rst_match": 0, "gaps": [],
                                       "beyond_top5": 0, "ppl_deltas": [],
                                       "blind_diverge": 0, "stable_cells": 0,
                                       "stable_prompts": 0})
    per_quant = defaultdict(lambda: {"prompts": 0, "match": 0})
    per_category = defaultdict(lambda: {"prompts": 0, "match": 0})
    per_cell = {}
    control_pairs = defaultdict(lambda: {"prompts": 0, "match": 0})
    CONTROL_PAIRS = {
        ("radeon-rx-7900-xtx-linux", "radeon-rx-7900-xtx-2-linux"): ["rocm", "vulkan"],
        ("rtx-pro-6000-blackwell-linux", "rtx-pro-6000-blackwell-2-linux"): ["cuda", "vulkan"],
    }
    CUDA13_PAIRS = {"rtx-5060-ti-linux", "rtx-pro-6000-blackwell-linux",
                    "rtx-pro-6000-blackwell-2-linux"}

    for (model, label), backends in sorted(groups.items()):
        ref = backends.get("cpu")
        if ref is None or ref.get("status") != "OK":
            continue
        ref_gens = [p.get("gen") or "" for p in ref.get("prompts", [])]
        ref_toks = [p.get("tokens") or [] for p in ref.get("prompts", [])]
        cats = [p.get("category") for p in ref.get("prompts", [])]
        cell_rows = {}
        for backend, rec in backends.items():
            if rec.get("status") != "OK":
                continue
            gens = [p.get("gen") or "" for p in rec.get("prompts", [])]
            toks = [p.get("tokens") or [] for p in rec.get("prompts", [])]
            n = min(len(gens), len(ref_gens))
            if backend != "cpu":
                st = per_backend[backend]
                st["cells"] += 1
                st["prompts"] += n
                m = sum(1 for i in range(n) if gens[i] == ref_gens[i])
                rm = sum(1 for i in range(n) if gens[i].rstrip() == ref_gens[i].rstrip())
                st["match"] += m
                st["rst_match"] += rm
                cell_rows[backend] = {"exact": m, "n": n}
                q = quant_of(model)
                per_quant[(backend, q)]["prompts"] += n
                per_quant[(backend, q)]["match"] += m
                ppl = (rec.get("ppl") or {}).get("value")
                ref_ppl = (ref.get("ppl") or {}).get("value")
                pd = None
                if ppl is not None and ref_ppl is not None:
                    pd = ppl - ref_ppl
                    st["ppl_deltas"].append(pd)
                st["blind_diverge"] += sum(
                    1 for i in range(n)
                    if gens[i] != ref_gens[i] and (pd is not None and pd == 0)
                )
                if all(p.get("stable") for p in rec.get("prompts", []) if p.get("gens")):
                    st["stable_cells"] += 1
                    st["stable_prompts"] += len(rec.get("prompts", []))
                # near-tie gaps at first divergence
                for i in range(n):
                    if gens[i] == ref_gens[i]:
                        if cats[i]:
                            per_category[(backend, cats[i])]["prompts"] += 1
                            per_category[(backend, cats[i])]["match"] += 1
                        continue
                    if cats[i]:
                        per_category[(backend, cats[i])]["prompts"] += 1
                    bt = toks[i] if i < len(toks) else []
                    rt = ref_toks[i] if i < len(ref_toks) else []
                    pos = None
                    for k in range(min(len(bt), len(rt))):
                        if bt[k].get("id") != rt[k].get("id"):
                            pos = k
                            break
                    if pos is None:
                        continue
                    lp, in_top = token_logprob(rt, pos, bt[pos].get("id"))
                    if lp is not None and rt[pos].get("logprob") is not None:
                        gap = rt[pos]["logprob"] - lp
                        st["gaps"].append(gap)
                        if not in_top:
                            st["beyond_top5"] += 1
        per_cell[(model, label)] = cell_rows

    # ---- control passes (post-collection, counted exactly once each) ----
    models = sorted({m for (_l, m, _b) in records.keys()})
    for (l1, l2), bk in CONTROL_PAIRS.items():
        for backend in bk:
            for model in models:
                a, b = records.get((l1, model, backend)), records.get((l2, model, backend))
                if a is None or b is None or a.get("status") != "OK" or b.get("status") != "OK":
                    continue
                ga = [p.get("gen") or "" for p in a.get("prompts", [])]
                gb = [p.get("gen") or "" for p in b.get("prompts", [])]
                n = min(len(ga), len(gb))
                key = (l1, l2, backend)
                control_pairs[key]["prompts"] += n
                control_pairs[key]["match"] += sum(1 for i in range(n) if ga[i] == gb[i])
    for label in CUDA13_PAIRS:
        for model in models:
            c12, c13 = records.get((label, model, "cuda")), records.get((label, model, "cuda13"))
            if c12 is None or c13 is None or c12.get("status") != "OK" or c13.get("status") != "OK":
                continue
            g12 = [p.get("gen") or "" for p in c12.get("prompts", [])]
            g13 = [p.get("gen") or "" for p in c13.get("prompts", [])]
            n = min(len(g12), len(g13))
            key = (label, label, "cuda-vs-cuda13")
            control_pairs[key]["prompts"] += n
            control_pairs[key]["match"] += sum(1 for i in range(n) if g12[i] == g13[i])

    def summarize(st):
        gaps = sorted(st["gaps"]) if st["gaps"] else []
        def pct(a, b):
            return round(100.0 * a / b, 1) if b else None
        return {
            "cells": st["cells"],
            "prompts": st["prompts"],
            "gen_exact_pct": pct(st["match"], st["prompts"]),
            "gen_exact_rstrip_pct": pct(st["rst_match"], st["prompts"]),
            "first_divergences": len(gaps),
            "gap_median_nats": round(gaps[len(gaps) // 2], 4) if gaps else None,
            "gap_p10_nats": round(gaps[max(0, len(gaps) // 10)], 4) if gaps else None,
            "gap_p90_nats": round(gaps[min(len(gaps) - 1, 9 * len(gaps) // 10)], 4) if gaps else None,
            "gap_beyond_top5_pct": pct(st["beyond_top5"], len(gaps)),
            "ppl_delta_abs_median": round(sorted(abs(d) for d in st["ppl_deltas"])[len(st["ppl_deltas"]) // 2], 5) if st["ppl_deltas"] else None,
            "ppl_bit_identical_cells": sum(1 for d in st["ppl_deltas"] if d == 0),
            "divergent_gens_with_zero_ppl_delta": st["blind_diverge"],
        }

    return {
        "per_backend": {b: summarize(st) for b, st in sorted(per_backend.items())},
        "per_backend_quant": {
            f"{b}/{q}": {"prompts": v["prompts"], "match": v["match"],
                         "gen_exact_pct": round(100.0 * v["match"] / v["prompts"], 1) if v["prompts"] else None}
            for (b, q), v in sorted(per_quant.items())
        },
        "per_category": {
            f"{b}/{c}": {"prompts": v["prompts"], "match": v["match"],
                         "gen_exact_pct": round(100.0 * v["match"] / v["prompts"], 1) if v["prompts"] else None}
            for (b, c), v in sorted(per_category.items())
        },
        "control_pairs": {
            f"{a}|{b}|{bk}": {"prompts": v["prompts"], "match": v["match"],
                              "cross_box_exact_pct": round(100.0 * v["match"] / v["prompts"], 1) if v["prompts"] else None}
            for (a, b, bk), v in sorted(control_pairs.items())
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results/analysis.json")
    args = ap.parse_args()
    records = load_records(args.raw)
    result = analyze(records)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
    # headline print
    print(f"[analysis] {len(records)} records -> {args.out}")
    for b, s in result["per_backend"].items():
        print(f"  {b:8s} exact={s['gen_exact_pct']}%  gap_med={s['gap_median_nats']} nats  "
              f"beyond_top5={s['gap_beyond_top5_pct']}%  ppl|d|med={s['ppl_delta_abs_median']}  "
              f"bitid_ppl_cells={s['ppl_bit_identical_cells']}  blind_diverge={s['divergent_gens_with_zero_ppl_delta']}")
    for k, v in result["control_pairs"].items():
        print(f"  CONTROL {k}: cross_exact={v['cross_box_exact_pct']}% over {v['prompts']} prompts")


if __name__ == "__main__":
    main()
