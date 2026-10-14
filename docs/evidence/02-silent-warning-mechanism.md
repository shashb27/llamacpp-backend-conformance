# Why the fallback is silent — source-level analysis at b11327

The claim this document backs (technote Finding 1): in runs like ours
the CPU fallback produces **no console output at all** — no error, no
warning, no offload summary, even with an explicit `-ngl 999`. This is
not luck; three mechanisms in the b11327 source stack, each verified by
reading the tagged source and by our on-box logs.

## Mechanism 1 — the backend-load failure is swallowed in release builds

`ggml/src/ggml-backend-reg.cpp`, `ggml_backend_load_all_from_path()`:

```c
void ggml_backend_load_all_from_path(const fs::path & dir_path) {
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif
    ...
    ggml_backend_load_best("hip", silent, dir_path);
```

When `libggml-hip.so` fails to load, `load_backend()`'s error branches
are all guarded by `if (!silent)`. Official prebuilts are release
builds (`NDEBUG` defined), so the load failure produces **no output**.

## Mechanism 2 — the only "no usable GPU" warning lives inside the `-ngl` option handler

`common/arg.cpp`, inside the handler for the `-ngl/--gpu-layers`
option (and its `-ngl-draft` twin):

```c
} else {
    params.n_gpu_layers = std::stoi(value);
}
if (!llama_supports_gpu_offload()) {
    fprintf(stderr, "warning: no usable GPU found, --gpu-layers option will be ignored\n");
    fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
    fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
}
```

If the server is launched **without** an explicit `-ngl`, this code
path never executes. Our sweep harness never passes `-ngl`
(`harness/run_sweep.py` server launch).

## Mechanism 3 — the warning's guard is defeated by the bundled RPC backend

`src/llama.cpp`:

```c
bool llama_supports_gpu_offload(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}
```

The official prebuilts bundle `libggml-rpc.so`, and the loader loads
it; the RPC backend being registered makes `llama_supports_rpc()` — and
therefore `llama_supports_gpu_offload()` — return **true** even on a
host with no usable GPU. The warning is then suppressed **even when
`-ngl` is passed explicitly**.

## Empirical confirmation (our runs)

- Process maps of every directly-diagnosed fallback server show
  `libggml-rpc.so` mapped (see
  [01-rocm-backend-load-failure.md](01-rocm-backend-load-failure.md)
  and [04-4090-guest-no-gpu.md](04-4090-guest-no-gpu.md)).
- A grep of the failing servers' logs for
  `backend|offload|hip|gpu|device|warning` returns **nothing** —
  including a run launched with explicit `-ngl 999`.
- Note for anyone auditing logs on this release: b11327's server prints
  no "offloaded X/Y layers" lines even when offload **succeeds**, so
  log-grepping alone cannot distinguish GPU from CPU runs — VRAM
  readings and process maps are required (see
  [03-cuda13-environment-ab.md](03-cuda13-environment-ab.md) for what
  successful offload looks like).

## Practical upshot

The warning the engine does have is dead on arrival for exactly the
configuration that needs it: a prebuilt (release, RPC-bundled) binary
on a host where the GPU backend cannot load. Anyone relying on seeing
it will see nothing. The checks that actually work are the wall-time
ratio against a CPU baseline, followed by verifying the backend library
is mapped in the server process.
