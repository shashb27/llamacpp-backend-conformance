# llamacpp-backend-conformance

A reproducible cross-backend numerical conformance study for llama.cpp.

**Question this repo answers:** given the *same* model file, the *same* prompts, and fully deterministic (greedy) decoding, do llama.cpp's compute backends agree with each other?

Throughput benchmarks already exist elsewhere. This repo deliberately measures **correctness, not speed**: exact-match of greedy generations across backends, and perplexity deltas against a reference backend, across CUDA, Vulkan, ROCm, SYCL and CPU — on hardware spanning several GPU generations and vendors.

Where backends disagree, that's either a real numerical bug worth reporting upstream, or a quantified precision boundary worth knowing about. Both are useful.

## Status

v1.0 dataset complete: 425 measured (model, backend, hardware) cells across 13 hardware labels (NVIDIA, AMD, Intel), pinned to a single llama.cpp release. Canonical summary: `results/audit-v2.json` and `results/matrix.md`. Full write-up: [paper/technote.pdf](paper/technote.pdf).

## Correction (Oct 2026)

An earlier comment I posted on llama.cpp issue #26964 said the "ROCm"
generations came back bit-identical to the CPU backend's on the same host.
That part is wrong, and I want it on the record here. What matched exactly
was **perplexity** — 68/68 cells, at the tool's printed precision. The
fallback generations matched the standalone CPU build's generations at
**46.8%** overall: the ROCm build fell back to its *own* bundled CPU
implementation, which is a different build's CPU code than the reference
build. The evidence for the fallback itself is unaffected — wall-time ratio
~1.0 vs the CPU build, no HIP library in the server's process maps, idle
VRAM with an explicit `-ngl 999`. Corrected numbers:
[`results/audit-v2.json`](results/audit-v2.json)
(`populations.POP-BUILDCPU.by_backend.rocm`).

Since then, a source-level check of the b11327 release turned up a
second imprecision in that comment: the `--gpu-layers option will be
ignored` warning it calls "the only hint" never actually prints in
this configuration. That warning lives only inside the `-ngl` option
handler, and even when `-ngl` is passed it is suppressed whenever the
bundled RPC backend is registered — which the official prebuilts
always load. Release builds also swallow the backend-load failure
itself silently. In runs like mine there was no console hint at all.
The full breakdown, with the relevant b11327 source, is in
[`docs/evidence/02-silent-warning-mechanism.md`](docs/evidence/02-silent-warning-mechanism.md).

## Data layout

In-repo: the harness, the analysis outputs (`results/analysis*.json`, `results/matrix*`), the device-truth audit (`results/audit-v2.json`), hardware/driver versions (`results/hardware-versions.md`), the pilot batch (`results/raw-v0/`), quarantined cells (`results/raw-excluded-4090-vulkan/`), and pinned-device verifications (`results/vk-verify/`).

The full raw run records (`results/raw/` and `results/raw-2b/`, ~2.4 GB) are packaged as tarballs attached to the [v1.0 release](../../releases). Extract them under `results/` and every summary table can be regenerated with `python harness/run_sweep.py --report-only`.

## What is measured

For every (model, backend, hardware) cell:

1. **Greedy generation exact-match** — deterministic completion of a fixed prompt set; output compared byte-for-byte against the reference backend (CPU). Mismatches record the first divergence position.
2. **Perplexity delta** — perplexity of a fixed corpus, reported as the delta vs the reference backend. Absolute PPL of a small fixed corpus is not meaningful on its own; the *difference between backends on identical input* is the signal.

## What is deliberately NOT measured

Speed. Throughput/latency on this hardware is covered by other benchmarking efforts; mixing it in would blur the correctness signal. This repo is about numerical agreement only.

## Reproduce

Requires Python 3.8+ (stdlib only) and ~1 GB disk for the pinned model.

```
python harness/run_sweep.py                    # full sweep for this machine
python harness/run_sweep.py --backends cpu,cuda
python harness/run_sweep.py --report-only      # recompute matrix from existing raw records
```

The harness downloads the pinned llama.cpp release binaries and model from their
official sources, records SHA256 of every artifact, and never builds anything —
so every number in `results/` traces to a public, bit-identical binary.

See [METHODOLOGY.md](METHODOLOGY.md) for the full protocol: pinning policy,
determinism controls, hardware disclosure, and the fail-closed rules.

## Hardware policy

Runs are labeled by GPU model and platform only (e.g. `rtx-5060-ti-linux`). Host
names, owners, and cluster details are not part of the dataset.

## License

MIT. Model files are downloaded from their official publishers and remain under
their own licenses.
