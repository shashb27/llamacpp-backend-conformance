#!/usr/bin/env python3
"""Cross-backend numerical conformance harness for llama.cpp.

Stdlib only. Downloads pinned official release binaries and the pinned model,
runs greedy generation over a fixed prompt set plus perplexity on a fixed
corpus for each backend, writes fail-closed raw records under results/raw/,
and builds the conformance matrix (results/matrix.{md,json}) comparing every
backend against the reference backend on the same hardware.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "harness"
CORPUS = ROOT / "corpus" / "fixed_corpus.txt"
MODELS = ROOT / "models"
BINS = ROOT / "bins"
RAW = ROOT / "results" / "raw"
MATRIX_MD = ROOT / "results" / "matrix.md"
MATRIX_JSON = ROOT / "results" / "matrix.json"
UA = {"User-Agent": "llamacpp-backend-conformance/0.1"}
RUN_TIMEOUT_S = 1800
CAPTURE_CAP = 20000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    if dest.exists():
        return sha256_file(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dest) + ".part")
    print(f"[download] {url}")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(tmp, dest)
    except OSError as e:
        print(f"[download] failed: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    return sha256_file(dest)


def extract(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        else:
            with tarfile.open(archive) as t:
                try:
                    t.extractall(dest, filter="data")
                except TypeError:
                    t.extractall(dest)
        return True
    except OSError as e:
        print(f"[extract] failed: {e}")
        shutil.rmtree(dest, ignore_errors=True)
        return False


def find_binary(root, name):
    exe = name + (".exe" if os.name == "nt" else "")
    for cand in sorted(root.rglob(exe)):
        if cand.is_file():
            return cand
    return None


def detect_gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            is_gpu = False
            for line in r.stdout.splitlines():
                if "Device Type:" in line:
                    is_gpu = "GPU" in line
                elif is_gpu and "Marketing Name:" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        r = subprocess.run(["clinfo"], capture_output=True, text=True, timeout=60)
        m = re.search(r"Device Name:\s*(.+)", r.stdout)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    try:
        r = subprocess.run(["lspci"], capture_output=True, text=True, timeout=60)
        for line in r.stdout.splitlines():
            if re.search(r"VGA compatible controller|3D controller|Display controller", line):
                return line.split(":", 1)[1].rsplit(":", 1)[-1].strip()
    except Exception:
        pass
    return platform.processor() or "cpu"


def slugify(text):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    for prefix in ("nvidia-", "geforce-", "radeon-", "intel-r-", "intel-", "amd-", "apple-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s or "unknown"


def sanitize(text):
    if not isinstance(text, str):
        return text
    text = text.replace(str(ROOT), ".")
    text = re.sub(r"/home/[A-Za-z0-9_.-]+/[\w.-]+", "<run-dir>", text)
    text = re.sub(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+\\[\w.-]+", "<run-dir>", text)
    return text


def run_cmd(cmd, env=None, timeout=RUN_TIMEOUT_S):
    t0 = time.time()
    real_argv = [str(c) for c in cmd]
    r = subprocess.run(real_argv, capture_output=True, text=True, env=env, timeout=timeout, errors="replace")
    return {
        "cmd": [sanitize(a) for a in real_argv],
        "rc": r.returncode,
        "stdout": sanitize(r.stdout[-CAPTURE_CAP:]),
        "stderr": sanitize(r.stderr[-CAPTURE_CAP:]),
        "wall_s": round(time.time() - t0, 2),
    }


def ensure_model(model_cfg):
    dest = MODELS / model_cfg["url"].rsplit("/", 1)[-1]
    sha = download(model_cfg["url"], dest)
    if sha is None:
        return None, None
    return dest, sha


def asset_paths(cfg, asset_name):
    tag = cfg["release_tag"]
    name = asset_name.replace("{tag}", tag)
    url = f"https://github.com/{cfg['release_repo']}/releases/download/{tag}/{name}"
    return url, BINS / name


def list_devices(server_bin, env):
    if server_bin is None:
        return None
    try:
        p = subprocess.run([str(server_bin), "--list-devices"], capture_output=True,
                           text=True, timeout=60, env=env or None)
        out = ((p.stdout or "") + (p.stderr or "")).strip()
        return out[-2000:] or None
    except Exception as e:
        return f"list-devices failed: {e}"


def ensure_backend(cfg, backend, plat):
    spec = cfg["backends"].get(backend, {}).get(plat)
    if not spec:
        return None
    url, archive = asset_paths(cfg, spec)
    sha = download(url, archive)
    if sha is None:
        return {"error": f"asset download failed: {archive.name}"}
    dest = Path(str(archive) + ".d")
    if not (dest / ".extracted").exists():
        free = shutil.disk_usage(BINS).free
        need = archive.stat().st_size * 3
        if free < need:
            return {"error": f"insufficient disk: need ~{need >> 20} MB, free {free >> 20} MB"}
        if not extract(archive, dest):
            return {"error": "asset extraction failed (disk or archive error)"}
        (dest / ".extracted").write_text("")
        try:
            os.remove(archive)
        except OSError:
            pass

    runtime = cfg["backends"][backend].get(plat + "_runtime")
    env = None
    if runtime:
        rt_url, rt_archive = asset_paths(cfg, runtime)
        rt_sha = download(rt_url, rt_archive)
        if rt_sha is None:
            return {"error": f"runtime download failed: {rt_archive.name}"}
        rt_dest = Path(str(rt_archive) + ".d")
        if not (rt_dest / ".extracted").exists():
            free = shutil.disk_usage(BINS).free
            need = rt_archive.stat().st_size * 3
            if free < need:
                return {"error": f"insufficient disk for runtime: need ~{need >> 20} MB, free {free >> 20} MB"}
            if not extract(rt_archive, rt_dest):
                return {"error": "runtime extraction failed (disk or archive error)"}
            (rt_dest / ".extracted").write_text("")
            try:
                os.remove(rt_archive)
            except OSError:
                pass
        if plat == "win":
            cli = find_binary(dest, "llama-cli")
            if cli is not None:
                for dll in rt_dest.rglob("*.dll"):
                    shutil.copy2(dll, cli.parent / dll.name)
        else:
            lib_dirs = sorted({p.parent for p in rt_dest.rglob("*.so*")})
            env = os.environ.copy()
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(str(d) for d in lib_dirs) + (":" + existing if existing else "")

    return {
        "cli": find_binary(dest, "llama-cli"),
        "perp": find_binary(dest, "llama-perplexity"),
        "server": find_binary(dest, "llama-server"),
        "asset": archive.name,
        "sha256": sha,
        "env": env,
        "devices": list_devices(find_binary(dest, "llama-server"), env),
    }


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def http_json(method, url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def server_generations(b, model_path, prompts, cfg, repeats):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    topk = cfg.get("server_logprobs", 5)
    argv = [str(b["server"]), "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port),
            "-c", str(cfg["ctx_size"]), "-b", str(cfg["batch_size"])]
    lf = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
    lpath = lf.name
    lf.close()
    with open(lpath, "wb") as out:
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=b.get("env"))
    try:
        ready = False
        deadline = time.time() + 240
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                if http_json("GET", base + "/health", timeout=5).get("status") == "ok":
                    ready = True
                    break
            except Exception:
                time.sleep(1)
        if not ready:
            tail = Path(lpath).read_text(errors="replace")[-2000:]
            return None, [], f"llama-server not ready (rc={proc.poll()}): {tail}"
        failed = []
        entries = []
        for pinfo in prompts:
            prompt = pinfo["prompt"]
            gens = []
            tokens = None
            ok = True
            t0 = time.time()
            for _ in range(repeats):
                try:
                    payload = {"prompt": prompt, "max_tokens": cfg["n_predict"], "temperature": 0,
                               "logprobs": topk, "seed": cfg["seed"], "cache_prompt": False}
                    resp = http_json("POST", base + "/v1/completions", payload, timeout=1800)
                    ch = resp["choices"][0]
                    gens.append(ch.get("text") or "")
                    lp = (ch.get("logprobs") or {}).get("content") or []
                    tokens = [
                        {"id": t.get("id"), "logprob": t.get("logprob"),
                         "top": [{"id": x.get("id"), "logprob": x.get("logprob")} for x in (t.get("top_logprobs") or [])]}
                        for t in lp
                    ]
                except Exception:
                    ok = False
                    gens.append(None)
            if not ok:
                failed.append(prompt)
            entries.append({
                "category": pinfo.get("category"),
                "prompt": prompt,
                "rc": 0 if ok else 1,
                "gen": gens[0],
                "gens": gens,
                "stable": gens[0] is not None and all(g == gens[0] for g in gens),
                "tokens": tokens,
                "wall_s": round(time.time() - t0, 2),
            })
            print(f"[server] gen ok={ok} {entries[-1]['wall_s']}s: {prompt[:40]}")
        return entries, failed, None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        try:
            os.remove(lpath)
        except OSError:
            pass


def write_record(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2))


def run_generation(cli, model_path, prompt, cfg, env):
    cmd = [
        cli, "-m", model_path, "-p", prompt,
        "-n", cfg["n_predict"], "-c", cfg["ctx_size"], "-b", cfg["batch_size"],
        "--temp", "0", "--seed", cfg["seed"],
        "--no-display-prompt", "--simple-io", "--single-turn",
    ]
    return run_cmd(cmd, env=env)


def parse_generation(stdout, prompt):
    marker = "> " + prompt + "\n"
    start = stdout.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = stdout.find("\n[ Prompt:", start)
    if end < 0:
        end = stdout.find("\n[", start)
    if end < 0:
        end = len(stdout)
    return stdout[start:end].rstrip("\n")


def run_perplexity(perp, model_path, cfg, env):
    cmd = [
        perp, "-m", model_path, "-f", CORPUS,
        "-c", cfg["ctx_size"], "-b", cfg["batch_size"],
        "--chunks", "16",
    ]
    run = run_cmd(cmd, env=env)
    text = run["stdout"] + "\n" + run["stderr"]
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", text)
    if m is None:
        m = re.search(r"PPL = ([0-9.]+)", text)
    return {
        "rc": run["rc"],
        "value": float(m.group(1)) if m else None,
        "stdout_tail": run["stdout"][-2000:],
        "stderr_tail": run["stderr"][-2000:],
        "wall_s": run["wall_s"],
    }


def sweep(cfg, prompts, backends, label_override, models_filter=None, repeats=1, protocol="cli", record_suffix=""):
    plat = "win" if os.name == "nt" else "linux"
    gpu = detect_gpu()
    label = label_override or (slugify(gpu) + "-" + plat)
    print(f"[sweep] hardware: {gpu} | label: {label} | protocol: {protocol} | backends: {', '.join(backends)} | repeats: {repeats}")

    models = cfg["models"]
    if models_filter:
        models = [m for m in models if m["name"] in models_filter]

    for model_cfg in models:
        model_path, model_sha = ensure_model(model_cfg)
        for backend in backends:
            rec_path = RAW / f"{label}__{backend}__{model_cfg['name']}{record_suffix}.json"
            rec = {
                "schema": 1,
                "created": datetime.now(timezone.utc).isoformat(),
                "label": label,
                "platform": plat,
                "gpu_detected": gpu,
                "backend": backend,
                "model": model_cfg["name"],
                "model_sha256": model_sha,
                "release_tag": cfg["release_tag"],
                "config": {k: cfg[k] for k in ("n_predict", "ctx_size", "batch_size", "seed", "temperature")},
                "status": "OK",
                "error": None,
                "protocol": "server-v1" if protocol == "server" else "cli-v0",
                "prompts": [],
                "ppl": None,
            }
            if model_path is None:
                rec["status"] = "UNAVAILABLE"
                rec["error"] = "model download failed"
                write_record(rec_path, rec)
                print(f"[{backend}] UNAVAILABLE: model download failed")
                continue
            b = ensure_backend(cfg, backend, plat)
            if b is None:
                rec["status"] = "UNAVAILABLE"
                rec["error"] = f"no {plat} asset for backend"
                write_record(rec_path, rec)
                print(f"[{backend}] UNAVAILABLE: no {plat} asset")
                continue
            if "error" in b:
                rec["status"] = "UNAVAILABLE"
                rec["error"] = b["error"]
                write_record(rec_path, rec)
                print(f"[{backend}] UNAVAILABLE: {b['error']}")
                continue
            rec["asset"] = {"name": b["asset"], "sha256": b["sha256"]}
            rec["devices"] = b.get("devices")
            if os.environ.get("GGML_VK_VISIBLE_DEVICES"):
                rec["ggml_vk_visible_devices"] = os.environ["GGML_VK_VISIBLE_DEVICES"]
            if b["cli"] is None or b["perp"] is None:
                rec["status"] = "UNAVAILABLE"
                rec["error"] = "binaries not found in extracted asset"
                write_record(rec_path, rec)
                print(f"[{backend}] UNAVAILABLE: binaries missing from asset")
                continue

            failed = []
            if protocol == "server":
                if b["server"] is None:
                    rec["status"] = "UNAVAILABLE"
                    rec["error"] = "llama-server binary not found in asset"
                    write_record(rec_path, rec)
                    print(f"[{backend}] UNAVAILABLE: llama-server missing from asset")
                    continue
                entries, sfailed, serr = server_generations(b, model_path, prompts, cfg, repeats)
                if entries is None:
                    rec["status"] = "UNAVAILABLE"
                    rec["error"] = serr
                    write_record(rec_path, rec)
                    print(f"[{backend}] UNAVAILABLE: {serr}")
                    continue
                rec["prompts"] = entries
                failed = sfailed
            else:
                for pinfo in prompts:
                    prompt = pinfo["prompt"]
                    gens = []
                    run_ok = True
                    run = None
                    for _ in range(repeats):
                        run = run_generation(b["cli"], model_path, prompt, cfg, b["env"])
                        gen = parse_generation(run["stdout"], prompt) if run["rc"] == 0 else None
                        if run["rc"] != 0 or gen is None:
                            run_ok = False
                        gens.append(gen)
                    entry = {
                        "category": pinfo.get("category"),
                        "prompt": prompt,
                        "rc": run["rc"] if run else 1,
                        "gen": gens[0],
                        "gens": gens,
                        "stable": gens[0] is not None and all(g == gens[0] for g in gens),
                        "stderr_tail": run["stderr"][-1000:] if run else "",
                        "wall_s": run["wall_s"] if run else 0.0,
                    }
                    if not run_ok:
                        failed.append(prompt)
                    rec["prompts"].append(entry)
                    print(f"[{backend}] gen ok={run_ok} {entry['wall_s']}s: {prompt[:40]}")

            ppl = run_perplexity(b["perp"], model_path, cfg, b["env"])
            rec["ppl"] = ppl
            if ppl["rc"] != 0 or ppl["value"] is None:
                failed.append("perplexity")
            print(f"[{backend}] ppl rc={ppl['rc']} value={ppl['value']}")

            if failed:
                rec["status"] = "FAILED"
                rec["error"] = f"failed steps: {failed}"
            write_record(rec_path, rec)
            print(f"[{backend}] -> {rec['status']} ({rec_path.name})")


def first_diff_offset(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def build_report(cfg):
    records = []
    for p in sorted(RAW.glob("*.json")):
        if len(p.stem.split("__")) > 3:
            continue
        try:
            records.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[report] unreadable record {p.name}: {e}")
    rows = []
    ref_backend = cfg["reference_backend"]
    for model in sorted({r["model"] for r in records}):
        recs_m = [r for r in records if r["model"] == model]
        for label in sorted({r["label"] for r in recs_m}):
            group = [r for r in recs_m if r["label"] == label]
            ref = next((r for r in group if r["backend"] == ref_backend and r["status"] == "OK"), None)
            for r in sorted(group, key=lambda x: x["backend"]):
                row = {
                    "model": model,
                    "label": label,
                    "backend": r["backend"],
                    "status": r["status"],
                    "gen_exact": None,
                    "gen_exact_rstrip": None,
                    "first_diffs": [],
                    "ppl": None,
                    "ppl_delta": None,
                    "reference": ref["backend"] if ref else None,
                }
                if r["status"] == "OK" and ref is not None:
                    rprompts = ref.get("prompts", [])
                    exact = 0
                    exact_rs = 0
                    diffs = []
                    for i, pr in enumerate(r.get("prompts", [])):
                        if i >= len(rprompts):
                            break
                        a = pr.get("gen") or ""
                        b = rprompts[i].get("gen") or ""
                        if a == b:
                            exact += 1
                            diffs.append(-1)
                        else:
                            diffs.append(first_diff_offset(a, b))
                        if a.rstrip() == b.rstrip():
                            exact_rs += 1
                    total = len(r.get("prompts", []))
                    row["gen_exact"] = f"{exact}/{total}"
                    row["gen_exact_rstrip"] = f"{exact_rs}/{total}"
                    row["first_diffs"] = diffs
                    if r.get("ppl") and ref.get("ppl"):
                        row["ppl"] = r["ppl"].get("value")
                        if ref["ppl"].get("value") is not None and r["ppl"].get("value") is not None:
                            row["ppl_delta"] = round(r["ppl"]["value"] - ref["ppl"]["value"], 4)
                rows.append(row)

    MATRIX_JSON.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "release_tag": cfg["release_tag"],
        "reference_backend": ref_backend,
        "rows": rows,
    }, indent=2))

    lines = [
        "# Conformance matrix",
        "",
        f"generated: {datetime.now(timezone.utc).isoformat()} | llama.cpp release: {cfg['release_tag']} | reference backend: {ref_backend}",
        "",
        "| model | hardware | backend | gen exact | first diff (chars) | ppl | ppl delta | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        diffs = ", ".join(str(d) for d in row["first_diffs"]) or "-"
        lines.append(
            f"| {row['model']} | {row['label']} | {row['backend']} | {row['gen_exact'] or '-'} | "
            f"{diffs} | {row['ppl'] if row['ppl'] is not None else '-'} | "
            f"{row['ppl_delta'] if row['ppl_delta'] is not None else '-'} | {row['status']} |"
        )
    lines += [
        "",
        "gen exact: byte-identical greedy continuations vs reference backend on same hardware.",
        "first diff: character offset of first divergence per prompt (-1 = identical).",
        "ppl delta: perplexity difference vs reference backend; same-input cross-backend delta is the signal.",
    ]
    MATRIX_MD.write_text("\n".join(lines) + "\n")
    print(f"[report] {len(rows)} rows -> {MATRIX_MD}")


def main():
    ap = argparse.ArgumentParser(description="llama.cpp cross-backend conformance sweep")
    ap.add_argument("--backends", default=None, help="comma-separated backend list (default: all with an asset for this platform)")
    ap.add_argument("--models", default=None, help="comma-separated model-name filter (default: all in config)")
    ap.add_argument("--label", default=None, help="hardware label override (default: detected GPU + platform)")
    ap.add_argument("--repeats", type=int, default=1, help="generation repeats per prompt for within-backend stability")
    ap.add_argument("--protocol", choices=["cli", "server"], default="cli", help="generation protocol: cli (conversation mode) or server (raw completions + logprobs)")
    ap.add_argument("--prompts-file", default=None, help="prompts JSON file (default: harness/prompts.json)")
    ap.add_argument("--n-predict", type=int, default=None, help="override generation length (n_predict)")
    ap.add_argument("--record-suffix", default="", help="suffix appended to record filenames (e.g. __ext, __n512) to namespace supplemental runs; suffixed records are excluded from the core matrix")
    ap.add_argument("--report-only", action="store_true", help="rebuild results/matrix.* from raw records without running")
    args = ap.parse_args()

    cfg = json.loads((HARNESS / "config.json").read_text())
    if args.n_predict is not None:
        cfg["n_predict"] = args.n_predict
    prompts_path = Path(args.prompts_file) if args.prompts_file else (HARNESS / "prompts.json")
    raw_prompts = json.loads(prompts_path.read_text())["prompts"]
    prompts = []
    for entry in raw_prompts:
        if isinstance(entry, str):
            prompts.append({"category": None, "prompt": entry})
        else:
            prompts.append(entry)

    if not args.report_only:
        plat = "win" if os.name == "nt" else "linux"
        if args.backends:
            backends = [b.strip() for b in args.backends.split(",") if b.strip()]
        else:
            backends = [b for b, spec in cfg["backends"].items() if plat in spec]
        mf = set(x.strip() for x in args.models.split(",")) if args.models else None
        sweep(cfg, prompts, backends, args.label, mf, args.repeats, args.protocol, args.record_suffix)
    build_report(cfg)


if __name__ == "__main__":
    main()
