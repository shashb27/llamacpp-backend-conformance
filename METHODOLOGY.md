# Methodology

## 1. Pinned artifacts

- **llama.cpp binaries**: official release assets from
  `https://github.com/ggml-org/llama.cpp/releases`, pinned to a single release
  tag recorded in `harness/config.json`. No source builds. The exact asset
  filenames per (platform, backend) are recorded in `config.json`; SHA256 of
  every downloaded asset and model is recorded in the raw run records and the
  artifact manifest. Anyone can reproduce a cell of the matrix by downloading
  the same public asset and running the same command.
  Note: llama.cpp marks its `b*` release tags as "pre-release" on GitHub; these
  tags are the project's actual release channel and are the only source of
  official binaries. The Windows CUDA runtime assets (`cudart-llama-bin-win-cuda-*.zip`)
  are shared across releases and carry no tag; their SHA256 is recorded in every
  run record that uses them.
- **Model**: GGUF file from the model publisher's official repository, pinned
  by URL. SHA256 recorded on first download and verified on every subsequent
  run.

## 2. Determinism controls

- Sampling temperature 0.0 (pure greedy argmax), fixed seed recorded, top-k
  sampling disabled by temperature; `n_predict` fixed per config.
- Identical prompt strings from `harness/prompts.json` for every backend,
  passed via `-p` through llama-cli's conversation mode: the model's chat
  template is applied by the same common code path on every backend.
  `--single-turn` and `--simple-io` force non-interactive single-turn
  execution. The user-turn echo (`> <prompt>`) is the parsing anchor; the
  generation is the text between it and the timing line, recorded raw.
- Identical context size and batch parameters for every backend.
- Reference backend is CPU on the same release build; every other backend is
  compared against it, and pairwise against each other via the raw records.
  The reference itself is verified run-stable by repeat execution (byte-
  identical generations, identical perplexity) before cross-backend claims.

## 3. Metrics

- **gen_exact_match** (per prompt, then aggregated): byte-for-byte equality of
  the generated continuation vs the reference backend. v0 comparison is at the
  text level; token-level divergence indexing is planned once logit-export
  tooling is wired in (tracked as a limitation, not hidden).
- **first_diff_offset**: character offset of the first divergence when
  `gen_exact_match` is false; -1 when true.
- **ppl** and **ppl_delta**: perplexity of the fixed corpus
  (`corpus/fixed_corpus.txt`), delta computed against the reference backend on
  the same hardware run. The corpus is project-authored fixed English prose
  (license-clean, no third-party text), ~6,800 words / ~4,700 model tokens,
  satisfying the >=4,096-token minimum required for a 2,048-token context
  window. The tool prints its final estimate to stderr
  (`Final estimate: PPL = ...`); it is parsed from the combined output
  streams. Absolute PPL values are not the signal — the cross-backend delta
  on identical input is.

## 4. Fail-closed rules

- Every subprocess invocation records: command line, exit code, captured
  stderr/stdout tail, wall time. Failed or missing runs are recorded with
  status `FAILED` / `UNAVAILABLE` in the matrix — never silently dropped.
- If any backend on a machine cannot execute (missing driver, incompatible
  runtime), that cell is reported as such, with the captured error, as a
  first-class result.
- Raw records under `results/raw/` are the source of truth; summary tables are
  generated from them only (`--report-only` regenerates without running).

## 5. Hardware disclosure

Runs are labeled by GPU model string and OS/platform only. No host names, no
owner identity, no infrastructure details. The harness detects the GPU via
standard tools (`nvidia-smi` / `rocminfo` / `clinfo` / `vulkaninfo`) where
available, or accepts a `--label` override.

## 6. Known limitations (v0)

- Text-level comparison, not token-level; a divergence that produces identical
  bytes is invisible (and harmless), a divergence is localized only to byte
  offset.
- One model family and quant in the initial matrix; more quants (q8_0, fp16)
  are planned to separate "quant noise" from "backend divergence".
- Perplexity is computed over a handful of chunks on the fixed corpus; the
  estimate is deterministic for a given backend but coarse — deltas smaller
  than its resolution are treated as agreement and reported as such.
- Conversation mode applies the model's chat template to prompts; this is a
  deliberate protocol choice (identical templating code on every backend),
  and raw-completion mode is left to a later protocol version.
- Backends that require driver/runtime versions not present on a given machine
  are reported UNAVAILABLE rather than worked around; the conformance matrix
  reflects what actually runs.
