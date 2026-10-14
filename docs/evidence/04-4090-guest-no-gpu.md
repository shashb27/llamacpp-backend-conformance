# The "RTX 4090" host — a VM with no GPU passthrough

**Host:** `rtx-4090-linux` (KVM guest, Ubuntu 24.04.3). Labeled as a
virtualized RTX 4090; the advertised GPU does not exist in the guest.

**Binary:** `llama-b11327-bin-ubuntu-cuda-12.8-x64.tar.gz` (+ runtime).

## 1. No NVIDIA presence in the guest

- `nvidia-smi`: command not found
- `ls /dev/nvidia*`: no such file or directory
- no `libcuda*` anywhere under `/usr/lib/x86_64-linux-gnu/`
- no `/proc/driver/nvidia`
- `lspci`: **no NVIDIA device at all** — the only GPU on the guest PCI
  bus is `Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770]`

The RTX 4090 is not passed through to this guest.

## 2. The CUDA backend cannot load

```
$ ldd ./libggml-cuda.so
	libcudart.so.12  => resolves from the runtime tarball
	libcublas.so.12  => resolves from the runtime tarball
	libcuda.so.1     => not found          # no driver library in the guest
```

`--list-devices` prints `(none)`.

## 3. Silent CPU fallback, same signature

Startup log with `-ngl 999`: normal init lines only — no
backend/offload/error lines (see
[02-silent-warning-mechanism.md](02-silent-warning-mechanism.md)).
Process maps show only `libggml-cpu-alderlake.so` and
`libggml-rpc.so`. Timed generation: 333.36 ms / 64 tokens
(5.29 ms/token, 188.98 tok/s) — CPU class.

## 4. How this host is used in the study

Its CUDA cells are classified CPU-fallback and retained only in the
build-CPU population; its Vulkan cells are **excluded** entirely (the
Vulkan backend enumerated and used the guest-visible Intel iGPU —
quarantined in `results/raw-excluded-4090-vulkan/`). Its earlier
"virtualized 4090" hardware label was wrong, and the host doubles as a
negative control demonstrating that the wall-time audit catches
even this case.
