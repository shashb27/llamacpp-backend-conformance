# CUDA-13 fallback — our launch-environment mistake, isolated by a matched A/B

**Host:** `rtx-5060-ti-linux` (RTX 5060 Ti 16 GB, driver 580.126.20,
CUDA 13.0 system runtime, Ubuntu 24.04.4).

**Binaries:** `llama-b11327-bin-ubuntu-cuda-13.4-x64.tar.gz` (+ its
cudart runtime tarball) and the 12.8 build. This failure class was our
own harness bug — the runtime-library path was passed to device
enumeration and to the perplexity tool but not to the server process
(fixed in `harness/run_sweep.py`: the server `Popen` now inherits the
runtime env). We disclose it as a retraction in the paper; the evidence
below is what established it.

## 1. The backend loads only with the runtime env

```
$ LD_LIBRARY_PATH=<runtime-dir> ldd ./libggml-cuda.so
	libcudart.so.13    => resolves from runtime dir
	libcublas.so.13    => resolves from runtime dir
	libcublasLt.so.13  => resolves from runtime dir
	libcuda.so.1       => resolves from system driver

$ ldd ./libggml-cuda.so            # without the runtime dir
	libcudart.so.13 => not found
	libcublas.so.13 => not found
```

The 12.8 build's `libggml-cuda.so` resolves **without** the env from
the system's `/usr/local/cuda-12.9` — which is why the 12.8 columns
were unaffected on this host.

## 2. `--list-devices` reflects the environment of the process that ran it

- WITH the runtime dir on `LD_LIBRARY_PATH`:
  `CUDA0: NVIDIA GeForce RTX 5060 Ti (15847 MiB, 15710 MiB free)`
- WITHOUT it: `(none)`

Our harness ran enumeration **with** the env and the server **without**
it — so the metadata listed the GPU while the server fell back to the
CPU.

## 3. Matched A/B at identical settings (`-ngl 999` both times)

SmolLM2-135M-Q4_K_M, 64-token generation, same box and model:

| Run | Build | Runtime env | `-ngl` | VRAM in use | Speed | Execution |
|---|---|---|---|---|---|---|
| 1 | cuda-13.4 | set | 999 | 456 MiB | 1.66 ms/tok (602 tok/s) | **GPU** |
| A | cuda-13.4 | unset | 999 | 2 MiB | 4.93 ms/tok (203 tok/s) | **CPU** |
| B | cuda-13.4 | set | auto | 456 MiB | 1.50 ms/tok (667 tok/s) | GPU |
| D | cuda-12.8 | unset | auto | 456 MiB | — | GPU (self-resolves via system CUDA 12.9) |

The Run-1/Run-A pair isolates the single variable (the runtime
environment): with it, explicit `-ngl 999` offloads and `libcublasLt`
is mapped in the server; without it, the same command runs on the CPU
at 203 tok/s with 2 MiB of VRAM.

## 4. What successful offload looks like on this release

With the env set: `libcublasLt.so.13` appears in the server's process
maps and `nvidia-smi` lists `llama-server` at ~446 MiB on the GPU.
(The server prints no offload lines even on success — see
[02-silent-warning-mechanism.md](02-silent-warning-mechanism.md).)
