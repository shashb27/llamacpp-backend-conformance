# Hardware/Software Versions (v2 dataset)

Driver and OS versions per label, captured via host probes on 2026-10-02.
CUDA/OpenCL runtime versions are pinned by the release assets (CUDA 12.4 win / 12.8 linux / 13.4; ROCm 10.0; SYCL fp16 build) and recorded per record via asset SHA256.

## NVIDIA (Linux)

| Label | GPU | Driver | OS |
|---|---|---|---|
| titan-rtx-linux | NVIDIA TITAN RTX | 580.105.08 | Ubuntu 24.04.3 LTS |
| rtx-4090-linux | NVIDIA GeForce RTX 4090 | not visible from guest (virtualized stack: no nvidia-smi, no /proc/driver/nvidia; libcuda injected — CUDA sweeps fully functional) | Ubuntu 24.04.3 LTS |
| rtx-5060-ti-linux | NVIDIA GeForce RTX 5060 Ti | 580.126.20 | Ubuntu 24.04.4 LTS |
| rtx-pro-6000-blackwell-linux | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 580.178.04 | Ubuntu 24.04.3 LTS |
| rtx-pro-6000-blackwell-2-linux | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 580.178.04 | Ubuntu 24.04.3 LTS |
| geforce-rtx-3060-ti-win (local) | NVIDIA GeForce RTX 3060 Ti | 596.21 (Windows, per prior session note) | Windows 11 |

Notes:
- The two PRO 6000 units run identical drivers (580.178.04) and OS — clean conditions for the identical-unit control comparison.
- The two units' records are kept under distinct labels and never mixed.

## AMD

| Label | GPU | Driver | OS |
|---|---|---|---|
| radeon-rx-7900-xtx-linux | AMD Radeon RX 7900 XTX | amdgpu 6.10.5, kernel 6.8.0-85-generic (captured 2026-10-04 via interactive session) | Ubuntu 24.04.3 LTS |
| radeon-rx-7900-xtx-2-linux | AMD Radeon RX 7900 XTX (2nd unit) | amdgpu 6.10.5, kernel 6.8.0-85 | Ubuntu 24.04.3 LTS |
| radeon-ai-pro-r9700-linux | AMD Radeon AI PRO R9700 | amdgpu 6.10.5, kernel 6.8.0-85 | Ubuntu 24.04.3 LTS |
| radeon-8060s-ryzen-ai-max-linux | AMD Radeon 8060S (Ryzen AI Max integrated, Strix Halo APU) | amdgpu 6.12.12, kernel 6.14.0-27 | Ubuntu 24.04.2 LTS |

Note: the two 7900 XTX units run IDENTICAL amdgpu driver (6.10.5), kernel (6.8.0-85-generic), and OS (Ubuntu 24.04.3) — the identical-GPU control pair is matched on GPU model + full driver stack, differing only in host platform. The 2nd 7900 XTX and the R9700 also share the same amdgpu driver and kernel.

## Intel

| Label | GPU | Driver | OS |
|---|---|---|---|
| intel-arc-a770-linux | Intel Arc A770 | i915 (in-kernel, 6.8.0-106); mesa-vulkan-drivers 25.0.7; intel-opencl-icd (compute runtime) 26.18.38308.1 | Ubuntu 24.04.3 LTS |
| intel-arc-b60-linux | Intel Arc B60 | xe + i915 kernel modules (kernel 6.17.0-1008-oem); mesa-vulkan-drivers 25.0.7; intel-opencl-icd 26.18.38308.1 | Ubuntu 24.04.3 LTS |
| intel-arc-a770-win | Intel Arc A770 (Windows) | 32.0.101.6790 (iGPU UHD 750: 31.0.101.5592) | Windows 11 Pro |

Note (a770-win device audit, 2026-10-04): the box hosts an Intel UHD 750 iGPU that enumerates as Vulkan0 ahead of the A770 (Vulkan1) — same pattern as the other iGPU-first boxes. Vulkan generation timings are dGPU-class (median 6.8s for 7B-q8 vs ~60s+ expected on the iGPU), and a pinned (GGML_VK_VISIBLE_DEVICES=1) byte-verify re-run was staged to results/vk-verify/ for confirmation.

## Device-list capture (2b records onward)

From batch 2b onward, every record embeds the backend's `--list-devices` output (`devices` field) and any `GGML_VK_VISIBLE_DEVICES` pin — making device provenance per-record verifiable instead of audit-inferred. Core (batch 2) records predate this field; fleet vulkan provenance for those is established by the pinned byte-identical verifications (titan-rtx-linux, radeon-ai-pro-r9700-linux; intel-arc-a770-win pending confirmation) kept under `results/vk-verify/`.

## Exclusions

- rtx-4090-linux VULKAN cells (17 records, quarantined to results/raw-excluded-4090-vulkan/): the box's virtualized NVIDIA guest exposes NO NVIDIA Vulkan device - llama's vulkan backend enumerated only the host Intel iGPU (RPL-S), so all its vulkan records were computed on the Intel iGPU (confirmed by --list-devices audit 2026-10-02 and 3x-slow gen timings). CUDA cells unaffected (CUDA sees only the 4090). Vulkan-on-Ada is not claimable from this box.
