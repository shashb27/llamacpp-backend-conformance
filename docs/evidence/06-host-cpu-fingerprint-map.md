# Host inventory — CPU fingerprints, run windows, and equivalence classes

Anonymized per the repo's hardware policy: hosts are identified by
dataset label (GPU model + OS). CPU fingerprints are the
`system_info` ISA lines the engine prints into every run record
(verifiable in the [dataset](../../releases)); run windows are the
record-internal `created` timestamps.

## Per-label fingerprints

| Label | Platform | Threads | AVX-512 | AMX | Notes |
|---|---|---|---|---|---|
| `titan-rtx-linux` | Linux | 8 | yes | no | |
| `rtx-pro-6000-blackwell-linux` | Linux | 8 | yes | no | |
| `rtx-pro-6000-blackwell-2-linux` | Linux | 8 | yes | no | **same physical machine as the row above**, re-run ~12 h later |
| `radeon-rx-7900-xtx-linux` | Linux | 8 | yes | no | |
| `radeon-rx-7900-xtx-2-linux` | Linux | **6** | **no** | no | different host CPU from the XTX row above |
| `radeon-ai-pro-r9700-linux` | Linux | 8 | yes | no | |
| `radeon-8060s-ryzen-ai-max-linux` | Linux | 16 | yes | no | AMD Zen 5 (Strix Halo APU) |
| `intel-arc-a770-linux` | Linux | 8 | yes | no | |
| `intel-arc-b60-linux` | Linux | 8 | yes | no | |
| `rtx-5060-ti-linux` | Linux | **16** | yes | **yes (AMX-INT8)** | |
| `rtx-4090-linux` | Linux (VM) | 8 | **no** (AVX-VNNI only) | no | Raptor Lake guest |
| `geforce-rtx-3060-ti-win` | Windows | 8 | yes | no | |
| `intel-arc-a770-win` | Windows | 8 | yes | no | |

**12 distinct physical machines, 13 labels** (the two PRO 6000 labels
are one machine measured twice). The two Arc A770 labels are **two
separate machines**, not one dual-boot box: both were swept in the same
batches on different OSes a few hours apart (Linux core run Oct 2
17:30–18:27; Windows core run Oct 2 22:57–00:17; both labels active in
the supplemental batch on Oct 3), with independently recorded driver
inventories. Host-identity statements are the operator's fleet record.

## Output equivalence classes (empirical)

Grouping hosts by byte-identical standalone-CPU-build outputs across
all 578 model–prompt combinations:

- **Class A (7 Linux hosts):** `titan-rtx-linux`,
  `rtx-pro-6000-blackwell-linux`, `radeon-rx-7900-xtx-linux`,
  `radeon-ai-pro-r9700-linux`, `radeon-8060s-ryzen-ai-max-linux`,
  `intel-arc-a770-linux`, `intel-arc-b60-linux` — every pair
  578/578 identical.
- **Class B (2 hosts):** `radeon-rx-7900-xtx-2-linux`,
  `rtx-4090-linux` — 578/578 with each other.
- **Singleton:** `rtx-5060-ti-linux` — differs from every other Linux
  host.
- **Windows pair:** `geforce-rtx-3060-ti-win` and `intel-arc-a770-win`
  agree 578/578 under the Windows CPU build (a different binary;
  excluded from the Linux cross-host rows).

The classes line up with CPU instruction-set tiers rather than with
any physical-host property: class A hosts all expose AVX-512 without
AMX (Rocket Lake–signature Intel hosts plus the Zen 5 APU), class B is
AVX2-only, and the singleton carries AMX. Consistently, the process
maps of the two directly-diagnosed class-B fallback runs show
different runtime-selected CPU kernel variants (`libggml-cpu-haswell`,
`libggml-cpu-alderlake`).

## Run windows (record-internal `created` timestamps, core batch)

| Label | Core run (local) | Supplemental run |
|---|---|---|
| `titan-rtx-linux` | Oct 1, 19:23–20:22 | Oct 3 |
| `rtx-pro-6000-blackwell-linux` | Oct 2, 00:20–03:02 | Oct 3 |
| `radeon-rx-7900-xtx-2-linux` | Oct 2, 13:37–15:58 | Oct 3 |
| `radeon-ai-pro-r9700-linux` | Oct 2, 15:59–17:28 | Oct 3 |
| `intel-arc-a770-linux` | Oct 2, 17:30–18:27 | Oct 3, 16:29–18:44 |
| `intel-arc-b60-linux` | Oct 2, 18:31–19:29 | Oct 3 |
| `intel-arc-a770-win` | Oct 2, 22:57–00:17 | Oct 3, 20:20–21:23 |
| `rtx-4090-linux` | Oct 2 | Oct 3 |
| `rtx-5060-ti-linux` | Oct 2 | Oct 3 |
| `geforce-rtx-3060-ti-win` | local, Oct 1–2 | Oct 3 |
| `radeon-8060s-ryzen-ai-max-linux` | Oct 2 | Oct 3 |
| `radeon-rx-7900-xtx-linux` | Oct 3, 12:02–13:23 (manual run) | same |
