# Conformance matrix

generated: 2026-10-02T02:00:28.161787+00:00 | llama.cpp release: b11327 | reference backend: cpu

| model | hardware | backend | gen exact | first diff (chars) | ppl | ppl delta | status |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct-q4_k_m | geforce-rtx-3060-ti-win | cpu | 8/8 | -1, -1, -1, -1, -1, -1, -1, -1 | 24.2326 | 0.0 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | geforce-rtx-3060-ti-win | cuda | 4/8 | -1, -1, 318, 0, -1, 166, -1, 261 | 24.1957 | -0.0369 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | geforce-rtx-3060-ti-win | vulkan | 3/8 | -1, -1, 381, 0, -1, 166, 258, 261 | 24.0814 | -0.1512 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | geforce-rtx-5090-linux | cpu | 8/8 | -1, -1, -1, -1, -1, -1, -1, -1 | 24.2654 | 0.0 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | geforce-rtx-5090-linux | cuda | - | - | - | - | UNAVAILABLE |
| qwen2.5-0.5b-instruct-q4_k_m | geforce-rtx-5090-linux | cuda13 | - | - | - | - | UNAVAILABLE |
| qwen2.5-0.5b-instruct-q4_k_m | geforce-rtx-5090-linux | vulkan | 5/8 | -1, -1, -1, 84, -1, 481, -1, 225 | 24.0784 | -0.187 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | intel-arc-a770-linux | cpu | 8/8 | -1, -1, -1, -1, -1, -1, -1, -1 | 24.2203 | 0.0 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | intel-arc-a770-linux | sycl | - | - | - | - | FAILED |
| qwen2.5-0.5b-instruct-q4_k_m | intel-arc-a770-linux | vulkan | 5/8 | -1, -1, 381, -1, -1, 303, -1, 219 | 24.2055 | -0.0148 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | radeon-rx-7900-xtx-linux | cpu | 8/8 | -1, -1, -1, -1, -1, -1, -1, -1 | 24.2203 | 0.0 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | radeon-rx-7900-xtx-linux | rocm | 3/8 | -1, -1, 344, 521, -1, 303, 238, 219 | 24.2203 | 0.0 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | radeon-rx-7900-xtx-linux | vulkan | 6/8 | -1, -1, 381, -1, -1, -1, -1, 219 | 24.1919 | -0.0284 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | rtx-5060-ti-linux | cpu | 8/8 | -1, -1, -1, -1, -1, -1, -1, -1 | 24.2146 | 0.0 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | rtx-5060-ti-linux | cuda | 2/8 | -1, -1, 310, 0, 293, 132, 238, 208 | 24.2044 | -0.0102 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | rtx-5060-ti-linux | cuda13 | 2/8 | -1, -1, 310, 0, 293, 132, 238, 208 | 24.2044 | -0.0102 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | rtx-5060-ti-linux | vulkan | 3/8 | -1, -1, 310, 0, -1, 130, 258, 225 | 24.0925 | -0.1221 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | titan-rtx-linux | cpu | 8/8 | -1, -1, -1, -1, -1, -1, -1, -1 | 24.2203 | 0.0 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | titan-rtx-linux | cuda | 5/8 | -1, -1, 169, 22, -1, -1, -1, 104 | 24.1457 | -0.0746 | OK |
| qwen2.5-0.5b-instruct-q4_k_m | titan-rtx-linux | vulkan | 3/8 | -1, -1, 578, 22, -1, 464, 238, 219 | 24.0841 | -0.1362 | OK |
| smollm2-135m-instruct-q4_k_m | geforce-rtx-3060-ti-win | cpu | 34/34 | -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 | 29.5176 | 0.0 | OK |

gen exact: byte-identical greedy continuations vs reference backend on same hardware.
first diff: character offset of first divergence per prompt (-1 = identical).
ppl delta: perplexity difference vs reference backend; same-input cross-backend delta is the signal.
