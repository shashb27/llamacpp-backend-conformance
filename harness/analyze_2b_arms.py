#!/usr/bin/env python3
"""Analysis of the 2b supplemental arms: extended prompts (__ext) and
length-drift (__n512).

Extended prompts: per-backend exact-match vs the CPU reference across the
16 extended prompts (categories: code, summarization, translation),
compared against the core-prompt divergence rates for the same models.

Length drift: agreement survival by generation position — the fraction of
prompts for which a backend matches the CPU reference through token k,
for k up to n_predict (512 for the arm, 128 for core).

Usage:
  python harness/analyze_2b_arms.py
"""
import glob
import json
import os
from collections import defaultdict


def load(raw_dir, suffix):
    """Load 4-part-stem records (label__backend__model__suffix) -> {(label, model, backend): rec}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("__")
        if len(parts) != 4 or parts[3] != suffix:
            continue
        label, backend, model = parts[0], parts[1], parts[2]
        with open(path, "r", encoding="utf-8") as fh:
            out[(label, model, backend)] = json.load(fh)
    return out


def core_of(raw_dir="results/raw"):
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        label, backend, model = parts
        out[(model, label)][backend] = json.load(open(path, "r", encoding="utf-8"))
    return out


def main():
    report = {"extended": {}, "length_drift": {}}

    # ---- extended prompts arm ----
    ext = load("results/raw-2b", "ext")
    groups = defaultdict(dict)
    for (label, model, backend), rec in ext.items():
        groups[(model, label)][backend] = rec
    per_backend = defaultdict(lambda: {"prompts": 0, "match": 0})
    per_cat = defaultdict(lambda: {"prompts": 0, "match": 0})
    for (model, label), bks in groups.items():
        ref = bks.get("cpu")
        if not ref or ref.get("status") != "OK":
            continue
        rg = [p.get("gen") or "" for p in ref.get("prompts", [])]
        cats = [p.get("category") for p in ref.get("prompts", [])]
        for backend, rec in bks.items():
            if backend == "cpu" or rec.get("status") != "OK":
                continue
            g = [p.get("gen") or "" for p in rec.get("prompts", [])]
            n = min(len(g), len(rg))
            per_backend[backend]["prompts"] += n
            m = sum(1 for i in range(n) if g[i] == rg[i])
            per_backend[backend]["match"] += m
            for i in range(n):
                if cats[i]:
                    per_cat[(backend, cats[i])]["prompts"] += 1
                    if g[i] == rg[i]:
                        per_cat[(backend, cats[i])]["match"] += 1
    report["extended"]["per_backend"] = {
        b: {"prompts": v["prompts"], "match": v["match"],
            "exact_pct": round(100 * v["match"] / v["prompts"], 1) if v["prompts"] else None}
        for b, v in sorted(per_backend.items())}
    report["extended"]["per_category"] = {
        f"{b}/{c}": {"prompts": v["prompts"], "match": v["match"],
                     "exact_pct": round(100 * v["match"] / v["prompts"], 1) if v["prompts"] else None}
        for (b, c), v in sorted(per_cat.items())}

    # ---- length-drift arm (__n512) ----
    n512 = load("results/raw-2b", "n512")
    groups = defaultdict(dict)
    for (label, model, backend), rec in n512.items():
        groups[(model, label)][backend] = rec
    # tokens give the generated ids; compare id-prefixes per position
    survival = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # backend -> pos -> [match, total]
    per_backend_512 = defaultdict(lambda: {"prompts": 0, "match": 0})
    for (model, label), bks in groups.items():
        ref = bks.get("cpu")
        if not ref or ref.get("status") != "OK":
            continue
        rtoks = [p.get("tokens") or [] for p in ref.get("prompts", [])]
        for backend, rec in bks.items():
            if backend == "cpu" or rec.get("status") != "OK":
                continue
            btoks = [p.get("tokens") or [] for p in rec.get("prompts", [])]
            n = min(len(btoks), len(rtoks))
            per_backend_512[backend]["prompts"] += n
            per_backend_512[backend]["match"] += sum(
                1 for i in range(n)
                if [t.get("id") for t in btoks[i]] == [t.get("id") for t in rtoks[i]])
            for i in range(n):
                a = [t.get("id") for t in btoks[i]]
                b = [t.get("id") for t in rtoks[i]]
                # agreement survival: prefix match up to first divergence
                k = 0
                while k < min(len(a), len(b)) and a[k] == b[k]:
                    k += 1
                for pos in range(0, min(len(a), len(b), 512) + 1, 16):
                    if pos <= k:
                        survival[backend][pos][0] += 1
                    survival[backend][pos][1] += 1
    report["length_drift"]["per_backend_full"] = {
        b: {"prompts": v["prompts"], "match": v["match"],
            "exact_pct": round(100 * v["match"] / v["prompts"], 1) if v["prompts"] else None}
        for b, v in sorted(per_backend_512.items())}
    report["length_drift"]["survival"] = {
        b: {str(pos): {"agree": v[0], "n": v[1], "pct": round(100 * v[0] / v[1], 1)}
            for pos, v in sorted(survival[b].items())}
        for b in sorted(survival.keys())}

    with open("results/analysis-2b-arms.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
    print("[2b-arms] extended per-backend:", {b: v["exact_pct"] for b, v in report["extended"]["per_backend"].items()})
    print("[2b-arms] extended per-category:", {k: v["exact_pct"] for k, v in report["extended"]["per_category"].items()})
    print("[2b-arms] n512 full-agreement:", {b: v["exact_pct"] for b, v in report["length_drift"]["per_backend_full"].items()})
    for b in report["length_drift"]["survival"]:
        s = report["length_drift"]["survival"][b]
        pts = [f"{p}:{s[p]['pct']}" for p in ("0", "64", "128", "256", "512") if p in s]
        print(f"  survival {b}: " + " ".join(pts))


if __name__ == "__main__":
    main()
