# ROCm backend load failure — official b11327 ROCm build on a stock host

**Host:** `radeon-rx-7900-xtx-2-linux` (RX 7900 XTX, Ubuntu 24.04.3,
ROCm 6.3.0 userspace, amdgpu 6.10.5) — the host diagnosed directly;
the three other AMD hosts show the identical wall-time signature
(ratio 1.000–1.002 vs the same-box CPU build).

**Binary:** `llama-b11327-bin-ubuntu-rocm-10.0-x64.tar.gz` (official
release asset; the asset name declares rocm-10.0).

## 1. `--list-devices` (verbatim)

```
Available devices:
  (none)
```

## 2. The backend library cannot load — `ldd` (verbatim)

```
$ ldd ./libggml-hip.so
	libhipblas.so.3 => not found
	librocblas.so.5 => not found
	libamdhip64.so.7 => not found
```

`ldd ./llama-server` itself reports no missing libraries — the failure
is in the backend shared library that is `dlopen`'d at runtime. The
tarball bundles no ROCm runtime libraries (only `libggml-hip.so`), and
the b11327 release ships no Linux ROCm runtime asset (only a Windows
ROCm zip exists).

## 3. The host's installed ROCm provides the previous major version

`/opt/rocm/lib` on this host contains `libamdhip64.so.6`,
`libhipblas.so.2`, `librocblas.so.4` — the build demands
`.so.7/.so.3/.so.5`, one major version newer. The SONAMEs do not match
even with `/opt/rocm/lib` on `LD_LIBRARY_PATH`. (`rocm-smi` is healthy:
Device 0, DID 0x744c, Navi 31.)

## 4. The server runs on the CPU — even with explicit `-ngl 999`

Startup log with `-ngl 999` after 20 s: normal server init lines only
(threadpool `n_threads=6`, model loaded, listening). **Zero**
backend/offload/GPU lines — no "offloaded X/Y layers", no error, no
warning. A grep of the server log for
`backend|offload|hip|gpu|device` returns nothing. See
[02-silent-warning-mechanism.md](02-silent-warning-mechanism.md) for
why no warning appears.

Timed generation with `-ngl 999`: 64 tokens in 322.90 ms
(5.13 ms/token, 195.10 tok/s) — CPU class. VRAM during the run:
~27 MB (idle).

## 5. Process maps confirm the backend was never loaded

`/proc/<pid>/maps` for the running server shows only
`libggml-cpu-haswell.so` and `libggml-rpc.so` mapped —
`libggml-hip.so` was never mapped (the `dlopen` failed).

## 6. Consequence for quality benchmarking

The release's `llama-perplexity` suffers the same backend failure, so
its values are host-CPU values: across the four AMD hosts the "ROCm"
perplexity matched the standalone CPU build's in 68/68 cells (at the
tool's printed precision). The "ROCm" *generations* did not match the
standalone CPU build's (46.8% pooled) — the fallback runs executed the
GPU builds' bundled CPU implementation, which differs from the
standalone CPU build's code. See
[results/audit-v2.json](../../results/audit-v2.json)
(`populations.POP-BUILDCPU.by_backend.rocm`).
