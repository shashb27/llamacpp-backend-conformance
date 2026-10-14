# Vulkan device-selection integrity — pinned re-runs

Seven of the fleet's hosts enumerate an integrated GPU as Vulkan
device 0 with the study GPU as device 1. `llama.cpp`'s auto-selection
picks the discrete GPU in that situation, but we did not take that on
faith: on three iGPU-first hosts we re-ran the Vulkan backend with the
device explicitly pinned (`GGML_VK_VISIBLE_DEVICES=1`) and compared
against the original unpinned core record.

Model: Qwen2.5-0.5B-Instruct Q4_K_M (34 prompts). Records:
[`results/vk-verify/`](../../results/vk-verify/).

| Host | Result vs unpinned core record | Pin evidence |
|---|---|---|
| `intel-arc-a770-win` | generations byte-identical **34/34**; perplexity identical (24.2127) | strongest: the pin and an A770-only device list are recorded **inside** the verification record itself |
| `titan-rtx-linux` | generations byte-identical **34/34** | pin set by the orchestration script (the b11327-era record predates the in-record pin field) |
| `radeon-ai-pro-r9700-linux` | generations byte-identical **34/34** | pin set by the orchestration script (same) |

Combined with dGPU-class generation timings on every other iGPU-first
host (median per-prompt wall ratios 0.07–0.66 vs the same-box CPU
build — see `devices` in
[results/audit-v2.json](../../results/audit-v2.json)), the fleet's
Vulkan cells are established as discrete-GPU-computed. The one host
where Vulkan ran on an iGPU — the 4090 guest — is excluded outright
(see [04-4090-guest-no-gpu.md](04-4090-guest-no-gpu.md)).
