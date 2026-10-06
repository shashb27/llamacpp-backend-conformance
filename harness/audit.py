#!/usr/bin/env python3
"""Phase-0 truth audit over the v2 core dataset.

Produces the canonical corrected numbers after the review round:
  A1  device-truth table: every (box, backend) classified as gpu-offloaded
      vs cpu-fallback from generation wall-time ratios vs the same-box CPU
  A2  true per-backend and per-machine N; pooled rates on clean populations
      (POP-GPU = verified-offloaded cells; POP-BUILDCPU = fallback cells)
      with prompt-clustered bootstrap CIs
  A3  gap-event accounting: reconcile total comparisons -> divergences ->
      first-divergence events -> computable reference-side and backend-side
      gaps (the 7,875 figure was backend-side; both sides now reported)
  A4  identical-stack dedup counts for mechanism events

Usage:
  python harness/audit.py            # writes results/audit-v2.json
"""
import glob
import json
import os
import random
from collections import defaultdict

RATIO_THRESHOLD = 0.85   # >= -> classified cpu-fallback
BOOTSTRAP_REPS = 1000

# Identical-stack groups for dedup: (gpu family, backend) -> [labels]
IDENTICAL_STACK_GROUPS = [
    ("vulkan", ["radeon-rx-7900-xtx-linux", "radeon-rx-7900-xtx-2-linux"]),
    ("vulkan", ["rtx-pro-6000-blackwell-linux", "rtx-pro-6000-blackwell-2-linux"]),
    ("cuda",   ["rtx-pro-6000-blackwell-linux", "rtx-pro-6000-blackwell-2-linux"]),
]
DEDUP_REPRESENTATIVE = {
    ("vulkan", "radeon-rx-7900-xtx-2-linux"),
    ("vulkan", "rtx-pro-6000-blackwell-2-linux"),
    ("cuda", "rtx-pro-6000-blackwell-2-linux"),
}

# rtx-pro-6000-blackwell-2-linux is the SAME physical box as
# rtx-pro-6000-blackwell-linux, re-run ~12h apart (bit-identical outputs;
# verification-report.md). Its cells are excluded from populations and
# reported separately as a same-box rerun-stability control.
RERUN_LABEL = "rtx-pro-6000-blackwell-2-linux"


def load_records(raw_dir="results/raw"):
    records = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        label, backend, model = parts
        records[(label, model, backend)] = json.load(open(path, encoding="utf-8"))
    return records


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def prompt_wall(rec):
    return [p.get("wall_s") for p in rec.get("prompts", []) if p.get("wall_s")]


def classify_devices(records):
    """A1: per (label, backend), median over models of the ratio of median
    per-prompt gen wall-time vs the same-box cpu record."""
    by_box_model = defaultdict(dict)
    for (label, model, backend), rec in records.items():
        by_box_model[(label, model)][backend] = rec
    ratios = defaultdict(list)
    for (label, model), bks in by_box_model.items():
        cpu = bks.get("cpu")
        if not cpu or cpu.get("status") != "OK":
            continue
        cw = median(prompt_wall(cpu))
        if not cw:
            continue
        for backend, rec in bks.items():
            if backend == "cpu" or rec.get("status") != "OK":
                continue
            bw = median(prompt_wall(rec))
            if bw:
                ratios[(label, backend)].append(bw / cw)
    table = {}
    for (label, backend), rs in sorted(ratios.items()):
        med = median(rs)
        table[f"{label}|{backend}"] = {
            "n_models": len(rs),
            "median_ratio": round(med, 3),
            "device": "cpu_fallback" if med >= RATIO_THRESHOLD else "gpu",
        }
    return table


def build_comparisons(records, device_table):
    """Every (label, model, backend) vs its same-box cpu reference."""
    groups = defaultdict(dict)
    for (label, model, backend), rec in records.items():
        groups[(label, model)][backend] = rec
    comps = []   # dicts: label, model, backend, device, per-prompt flags
    skipped = defaultdict(int)
    for (label, model), bks in groups.items():
        cpu = bks.get("cpu")
        if not cpu or cpu.get("status") != "OK":
            skipped["no_cpu_reference"] += 1
            continue
        cp = cpu.get("prompts", [])
        for backend, rec in bks.items():
            if backend == "cpu":
                continue
            if rec.get("status") != "OK":
                skipped[f"status_{rec.get('status')}"] += 1
                continue
            device = device_table.get(f"{label}|{backend}", {}).get("device", "unknown")
            bp = rec.get("prompts", [])
            n = min(len(bp), len(cp))
            flags = []
            for i in range(n):
                if (bp[i].get("prompt") or "") != (cp[i].get("prompt") or ""):
                    skipped["prompt_mismatch"] += 1
                    flags.append(None)
                    continue
                flags.append(bp[i].get("gen") == cp[i].get("gen"))
            comps.append({"label": label, "model": model, "backend": backend,
                          "device": device, "flags": flags})
    return comps, skipped


def rate(flags):
    f = [x for x in flags if x is not None]
    return (100.0 * sum(f) / len(f), len(f)) if f else (None, 0)


def bootstrap_ci(comps, reps=BOOTSTRAP_REPS, seed=7):
    """Prompt-clustered bootstrap over pooled match rate."""
    rng = random.Random(seed)
    n_prompts = max(len(c["flags"]) for c in comps)
    rates = []
    for _ in range(reps):
        idx = [rng.randrange(n_prompts) for _ in range(n_prompts)]
        m = t = 0
        for c in comps:
            for i in idx:
                if i < len(c["flags"]) and c["flags"][i] is not None:
                    t += 1
                    m += 1 if c["flags"][i] else 0
        rates.append(100.0 * m / t if t else 0)
    rates.sort()
    return (round(rates[int(0.025 * reps)], 1), round(rates[int(0.975 * reps)], 1))


def gap_accounting(records, comps):
    """A3: reconcile comparison counts with gap-event counts, both sides."""
    groups = defaultdict(dict)
    for (label, model, backend), rec in records.items():
        groups[(label, model)][backend] = rec
    acct = defaultdict(int)
    gap_events = []   # (label, model, backend, prompt_idx, gap_ref, gap_backend, in_top5)
    for c in comps:
        key = (c["label"], c["model"])
        cpu = groups[key].get("cpu")
        rec = groups[key].get(c["backend"])
        ctoks = [p.get("tokens") or [] for p in cpu.get("prompts", [])]
        btoks = [p.get("tokens") or [] for p in rec.get("prompts", [])]
        for i, ok in enumerate(c["flags"]):
            if ok is None:
                acct["prompt_mismatch"] += 1
                continue
            acct["comparisons"] += 1
            if ok:
                acct["exact_match"] += 1
                continue
            acct["divergent"] += 1
            a, b = ctoks[i], btoks[i]
            if not a or not b:
                acct["div_no_tokens"] += 1
                continue
            pos = None
            for k in range(min(len(a), len(b))):
                if (a[k].get("id") or -1) != (b[k].get("id") or -2):
                    pos = k
                    break
            if pos is None:
                acct["div_pos_not_found"] += 1
                continue
            acct["first_div_events"] += 1
            ref_choice = a[pos].get("id")
            bak_choice = b[pos].get("id")
            # reference-side gap: ref's preference for its argmax over the
            # backend's choice (within the reference's recorded top-5)
            gap_ref, in_top5 = None, False
            for t in (a[pos].get("top") or []):
                if t.get("id") == bak_choice:
                    gap_ref = a[pos].get("logprob") - t.get("logprob")
                    in_top5 = True
                    break
            if gap_ref is None:
                acct["gap_ref_not_in_top5"] += 1
            else:
                acct["gap_ref_computable"] += 1
            # backend-side gap: backend's preference for its choice over
            # the reference's argmax (within the backend's recorded top-5)
            gap_bak = None
            for t in (b[pos].get("top") or []):
                if t.get("id") == ref_choice:
                    gap_bak = b[pos].get("logprob") - t.get("logprob")
                    break
            if gap_bak is None:
                acct["gap_backend_not_in_top5"] += 1
            else:
                acct["gap_backend_computable"] += 1
            gap_events.append((c["label"], c["model"], c["backend"], i,
                               gap_ref, gap_bak, in_top5))
    return acct, gap_events


def main():
    records = load_records()
    device_table = classify_devices(records)
    comps, skipped = build_comparisons(records, device_table)

    out = {"devices": device_table, "skipped": dict(skipped)}

    # same-box rerun stability (pro-6000 re-run ~12h apart)
    rerun = {"cells": 0, "prompts": 0, "match": 0}
    base_by_key = {}
    for c in comps:
        if c["label"] == "rtx-pro-6000-blackwell-linux":
            base_by_key[(c["backend"], c["model"])] = c["flags"]
    for c in comps:
        if c["label"] == RERUN_LABEL and (c["backend"], c["model"]) in base_by_key:
            bf = base_by_key[(c["backend"], c["model"])]
            n = min(len(bf), len(c["flags"]))
            m = sum(1 for i in range(n) if bf[i] is not None and c["flags"][i] is not None and bf[i] == c["flags"][i])
            rerun["cells"] += 1
            rerun["prompts"] += n
            rerun["match"] += m
    if rerun["prompts"]:
        rerun["match_pct"] = round(100 * rerun["match"] / rerun["prompts"], 1)
    out["rerun_stability"] = rerun

    # populations exclude the rerun box (duplicate of the base box)
    pop_comps = [c for c in comps if c["label"] != RERUN_LABEL]

    # A2: true N per backend and per machine, split by device truth
    per_backend = {}
    for pop_name, want_device in (("POP-GPU", "gpu"), ("POP-BUILDCPU", "cpu_fallback")):
        pop = [c for c in pop_comps if c["device"] == want_device]
        by_backend = defaultdict(lambda: {"cells": 0, "prompts": 0, "match": 0})
        by_machine = defaultdict(lambda: {"cells": 0, "prompts": 0, "match": 0})
        for c in pop:
            r, n = rate(c["flags"])
            by_backend[c["backend"]]["cells"] += 1
            by_backend[c["backend"]]["prompts"] += n
            by_backend[c["backend"]]["match"] += int(sum(x for x in c["flags"] if x is not None))
            key = f"{c['label']}|{c['backend']}"
            by_machine[key]["cells"] += 1
            by_machine[key]["prompts"] += n
            by_machine[key]["match"] += int(sum(x for x in c["flags"] if x is not None))
        per_backend[pop_name] = {
            "by_backend": {b: {**v, "exact_pct": round(100 * v["match"] / v["prompts"], 1)}
                           for b, v in sorted(by_backend.items())},
            "by_machine": {k: {**v, "exact_pct": round(100 * v["match"] / v["prompts"], 1)}
                           for k, v in sorted(by_machine.items())},
            "pooled": {},
        }
        tp = sum(v["prompts"] for v in by_backend.values())
        tm = sum(v["match"] for v in by_backend.values())
        per_backend[pop_name]["pooled"] = {
            "cells": len(pop), "prompts": tp, "match": tm,
            "exact_pct": round(100 * tm / tp, 1),
        }
        per_backend[pop_name]["pooled"]["ci95_prompt_clustered"] = bootstrap_ci(pop)
    out["populations"] = per_backend

    # per-quant x population rates (deduped populations)
    def quant_of(model):
        for q in ("fp16", "f16", "q8_0", "q6_k", "q4_k_m"):
            if model.endswith(q):
                return "f16" if q in ("fp16", "f16") else q
        return "?"
    per_quant = {}
    for pop_name, want_device in (("POP-GPU", "gpu"), ("POP-BUILDCPU", "cpu_fallback")):
        agg = defaultdict(lambda: [0, 0])
        for c in pop_comps:
            if c["device"] != want_device:
                continue
            q = quant_of(c["model"])
            for f in c["flags"]:
                if f is not None:
                    agg[q][1] += 1
                    agg[q][0] += 1 if f else 0
        per_quant[pop_name] = {q: {"prompts": v[1], "match": v[0],
                                   "exact_pct": round(100 * v[0] / v[1], 1)}
                               for q, v in sorted(agg.items())}
    out["per_quant"] = per_quant

    # A3: gap accounting
    acct, gap_events = gap_accounting(records, comps)
    out["gap_accounting"] = {k: int(v) for k, v in sorted(acct.items())}

    # quantiles of both-side gaps over all divergent events
    import statistics
    for side, idx in (("gap_ref", 4), ("gap_backend", 5)):
        vals = sorted(e[idx] for e in gap_events if e[idx] is not None)
        if vals:
            out[f"{side}_quantiles"] = {
                "n": len(vals),
                "median": round(statistics.median(vals), 4),
                "p90": round(vals[int(0.9 * len(vals))], 4),
                "p99": round(vals[int(0.99 * len(vals))], 4),
                "max": round(vals[-1], 4),
            }
    out["in_top5_pct_of_first_div"] = round(
        100 * sum(1 for e in gap_events if e[6]) / len(gap_events), 2) if gap_events else None

    # A4: identical-stack dedup
    deduped = [e for e in gap_events
               if (e[2], e[0]) not in DEDUP_REPRESENTATIVE or e[0] in
               {r for b, r in [] }]
    # simpler: drop events whose (backend, label) is a non-representative dup
    drop = set()
    for backend, labels in [(b, ls) for b, ls in
                            [(g[0], g[1]) for g in IDENTICAL_STACK_GROUPS]]:
        for lb in labels:
            if (backend, lb) in DEDUP_REPRESENTATIVE:
                drop.add((backend, lb))
    n_before = len(gap_events)
    dedup_events = [e for e in gap_events if (e[2], e[0]) not in drop]
    out["dedup"] = {
        "first_div_events_all": n_before,
        "first_div_events_dedup": len(dedup_events),
        "dropped_duplicate_stack_events": n_before - len(dedup_events),
    }

    with open("results/audit-v2.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)

    print("=== A1 device truth ===")
    for k, v in device_table.items():
        print(f"  {k:55s} ratio={v['median_ratio']:.2f} -> {v['device']}")
    for pop in ("POP-GPU", "POP-BUILDCPU"):
        p = out["populations"][pop]
        print(f"=== A2 {pop}: pooled {p['pooled']['exact_pct']}% "
              f"({p['pooled']['match']}/{p['pooled']['prompts']}) "
              f"CI95 {p['pooled']['ci95_prompt_clustered']} ===")
        for b, v in p["by_backend"].items():
            print(f"  {b:7s} cells={v['cells']:3d} prompts={v['prompts']:5d} exact={v['exact_pct']}%")
    print("=== A3 gap accounting ===")
    for k, v in out["gap_accounting"].items():
        print(f"  {k:28s} {v}")
    for side in ("gap_ref", "gap_backend"):
        if side in out:
            print(f"  {side}: {out[side]}")
    print("  in_top5_pct_of_first_div:", out["in_top5_pct_of_first_div"])
    print("=== A4 dedup ===", out["dedup"])


if __name__ == "__main__":
    main()
