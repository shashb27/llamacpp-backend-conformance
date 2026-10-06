#!/usr/bin/env python3
"""Regenerate the conformance matrix (matrix.md + matrix.json) from a record directory.

Usage:
  python harness/make_matrix.py --raw results/raw --out results/matrix
  python harness/make_matrix.py --raw results/raw-2b --out results/matrix-2b

Reference backend is cpu within each (label, model) group. Rows are emitted
per record; non-OK records show '-' for gen/ppl columns.
"""
import argparse
import datetime
import glob
import json
import os

BACKEND_ORDER = ["cpu", "cuda", "cuda13", "rocm", "sycl", "vulkan"]


def first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def load_records(raw_dir: str):
    records = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("__")
        if len(parts) != 3:
            continue  # suffixed supplemental records are excluded from core matrices
        label, backend, model = parts
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        records[(label, model, backend)] = rec
    return records


def build_rows(records: dict) -> list:
    groups = {}
    for (label, model, backend), rec in records.items():
        groups.setdefault((model, label), {})[backend] = rec

    rows = []
    for (model, label) in sorted(groups.keys()):
        backends = groups[(model, label)]
        ordered = [b for b in BACKEND_ORDER if b in backends] + sorted(
            b for b in backends if b not in BACKEND_ORDER
        )
        ref = backends.get("cpu")
        for backend in ordered:
            rec = backends[backend]
            row = {
                "model": model,
                "label": label,
                "backend": backend,
                "status": rec.get("status", "?"),
                "reference": "cpu",
            }
            if row["status"] == "OK" and ref is not None and ref.get("status") == "OK":
                gens = [p.get("gen", "") for p in rec.get("prompts", [])]
                ref_gens = [p.get("gen", "") for p in ref.get("prompts", [])]
                n = len(gens)
                exact = sum(1 for g, r in zip(gens, ref_gens) if g == r)
                exact_rstrip = sum(1 for g, r in zip(gens, ref_gens) if g.rstrip() == r.rstrip())
                diffs = [first_diff(g, r) for g, r in zip(gens, ref_gens)]
                ppl = rec.get("ppl", {}).get("value")
                ref_ppl = ref.get("ppl", {}).get("value")
                row["gen_exact"] = f"{exact}/{n}"
                row["gen_exact_rstrip"] = f"{exact_rstrip}/{n}"
                row["first_diffs"] = diffs
                row["ppl"] = ppl
                row["ppl_delta"] = (
                    None if (ppl is None or ref_ppl is None) else round(ppl - ref_ppl, 4)
                )
            else:
                row["gen_exact"] = None
                row["gen_exact_rstrip"] = None
                row["first_diffs"] = []
                row["ppl"] = None
                row["ppl_delta"] = None
            rows.append(row)
    return rows


def fmt(value, fmt_str: str) -> str:
    if value is None:
        return "-"
    return fmt_str.format(value)


def write_md(rows: list, out_path: str, generated: str, release_tag: str, reference_backend: str):
    lines = [
        "# Conformance matrix",
        "",
        f"generated: {generated} | llama.cpp release: {release_tag} | reference backend: {reference_backend}",
        "",
        "| model | hardware | backend | gen exact | first diff (chars) | ppl | ppl delta | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        diffs = (
            ", ".join(str(d) for d in r["first_diffs"])
            if r["first_diffs"] is not None
            else "-"
        )
        ppl = fmt(r["ppl"], "{:.4f}")
        delta = fmt(r["ppl_delta"], "{:.4f}")
        lines.append(
            f"| {r['model']} | {r['label']} | {r['backend']} | "
            f"{r['gen_exact'] or '-'} | {diffs} | {ppl} | {delta} | {r['status']} |"
        )
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results/matrix")
    ap.add_argument("--config", default="harness/config.json")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    generated = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = load_records(args.raw)
    rows = build_rows(records)

    payload = {
        "generated": generated,
        "release_tag": config.get("release_tag", "?"),
        "reference_backend": config.get("reference_backend", "cpu"),
        "rows": rows,
    }
    with open(args.out + ".json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    write_md(
        rows,
        args.out + ".md",
        generated,
        config.get("release_tag", "?"),
        config.get("reference_backend", "cpu"),
    )
    print(f"[matrix] {len(rows)} rows -> {args.out}.md / {args.out}.json")


if __name__ == "__main__":
    main()
