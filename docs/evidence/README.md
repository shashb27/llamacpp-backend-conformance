# Evidence bundle — silent CPU fallback in llama.cpp b11327

First-party diagnostic evidence backing the claims in
[paper/technote.tex](../paper/technote.tex) (Finding 1) and the
device-truth audit method. Everything here was captured on-box during
the study (2026-10-02/04) or read directly from the pinned release's
source; nothing is reconstructed from memory.

All artifacts reference the **official llama.cpp release b11327**
prebuilt Linux tarballs (SHA256 recorded per run record in the
[dataset](../../releases)). Hosts are identified by dataset label
only (GPU model + OS), per the repo's hardware policy.

| File | What it establishes |
|---|---|
| [01-rocm-backend-load-failure.md](01-rocm-backend-load-failure.md) | The ROCm backend library cannot load on a stock ROCm 6.3.0 host: it wants SONAMEs one major version newer, the release bundles no Linux ROCm runtime, `--list-devices` prints `(none)`, and the server runs on CPU even with explicit `-ngl 999`. |
| [02-silent-warning-mechanism.md](02-silent-warning-mechanism.md) | Why there is **no console indication at all**: the engine's only "no usable GPU" warning fires solely inside the `-ngl` option handler and is suppressed whenever the bundled RPC backend is registered; release builds additionally swallow backend-load failures. Source-level analysis with verbatim b11327 code. |
| [03-cuda13-environment-ab.md](03-cuda13-environment-ab.md) | The CUDA-13 fallback was our own launch-environment mistake: a matched `-ngl 999` A/B on the same box (602 vs 203 tok/s; 456 MiB vs 2 MiB VRAM) with `ldd` both ways. |
| [04-4090-guest-no-gpu.md](04-4090-guest-no-gpu.md) | The "RTX 4090" host is a VM with no GPU passthrough: no NVIDIA device on the guest PCI bus, no driver library, no `/dev/nvidia*`. |
| [05-vk-verify.md](05-vk-verify.md) | Vulkan device-selection integrity: pinned re-runs reproduce the core records byte-for-byte on three iGPU-first hosts. |
| [06-host-cpu-fingerprint-map.md](06-host-cpu-fingerprint-map.md) | Anonymized host inventory: per-label CPU ISA fingerprints, thread counts, run windows, and the empirical output-equivalence classes — with the operator-confirmed host identities (12 machines; two separate Arc A770 machines; the two PRO 6000 labels = one machine measured twice). |

The per-run wall-time ratios that first flagged every fallback
configuration are in
[results/audit-v2.json](../results/audit-v2.json) (`devices`).
