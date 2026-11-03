"""Packing tracr lib programs into one model, over vocab size V and length N.

For each (V, N): compile every program, shrink each alone (the separate-model
baseline), pack the first K programs for K = 1..len(programs), and compare
every task of the full pack with its program on every input (small domains)
or on random inputs (large ones). Settings live in config.toml [pack_report];
inputs follow [shrink_report]. Writes results.jsonl (one row per config,
flushed), run.log, pack_report.png, summary.md and summary.json to out_dir.

Usage:
  uv run --extra experiments python experiments/pack_report.py
"""

import json
import os
import resource
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import shrink_report

ROOT = shrink_report.ROOT
# Another config file (e.g. a tiny smoke-test grid) can be set in the env;
# child processes inherit it. Set SHRINK_REPORT_CONFIG to the same file.
CONFIG_PATH = os.environ.get("PACK_REPORT_CONFIG",
                             os.path.join(ROOT, "config.toml"))
with open(CONFIG_PATH, "rb") as _f:
  CFG = tomllib.load(_f)["pack_report"]
OUT_DIR = os.path.join(ROOT, CFG["out_dir"])
# Caps how many heavy configs hold their memory at once.
HEAVY_SLOTS = threading.Semaphore(CFG["heavy_workers"])

HEADER = (f"{'V':>4}{'N':>4}{'K':>4}{'d_pack↓':>8}{'d_max':>7}{'d_sum':>7}"
          f"{'bound':>7}{'solver':>8}{'packed↓':>12}{'block↓':>13}"
          f"{'separate':>12}{'x_block↑':>9}{'seqs':>7}{'mism↓':>6}{'ties':>6}"
          f"{'nbits':>6}{'maxdiff↓':>10}{'max_gate':>10}{'pack_s':>7}{'total_s':>8}")


def child(V, N):
  from tracr.compiler import compiling
  from tracr.compiler import packing
  from tracr.compiler import shrinking
  t_start = time.perf_counter()
  names, models, inputs, programs = [], [], [], []
  for prog in CFG["programs"]:
    program, vocab = shrink_report.build(prog, V, N)
    t0 = time.perf_counter()
    model = compiling.compile_rasp_to_model(
        program, vocab=vocab, max_seq_len=N, compiler_bos=CFG["compiler_bos"])
    compile_s = time.perf_counter() - t0
    trim = shrinking.shrink(model, reuse=False)
    reuse = shrinking.shrink(model, reuse=True)
    programs.append(dict(
        prog=prog, compile_s=compile_s, dense=trim.stats.params_before,
        d_trim=trim.stats.d_model_after, params_trim=trim.num_params,
        d_reuse=reuse.stats.d_model_after, params_reuse=reuse.num_params))
    del trim, reuse
    seqs, mode, _ = shrink_report.inputs_for(vocab, N, shrink_report.CFG["seed"])
    names.append(prog)
    models.append(model)
    inputs.append((seqs, mode))
    print(f"V={V} N={N} compiled {prog} in {compile_s:.1f} s", file=sys.stderr,
          flush=True)
  options = dict(gate_scale=CFG["gate_scale"], gate_offset=CFG["gate_offset"],
                 ilp_time_limit_s=CFG["ilp_time_limit_s"],
                 ilp_max_nonzeros=CFG["ilp_max_nonzeros"])
  sweep = []
  packed = None
  for k in range(1, len(models) + 1):
    t0 = time.perf_counter()
    packed = packing.pack(models[:k], names[:k], **options)
    stats = dict(vars(packed.stats), K=k, pack_s=time.perf_counter() - t0)
    stats["separate_trim"] = sum(p["params_trim"] for p in programs[:k])
    stats["separate_reuse"] = sum(p["params_reuse"] for p in programs[:k])
    sweep.append(stats)
    print(f"V={V} N={N} packed K={k} in {stats['pack_s']:.1f} s",
          file=sys.stderr, flush=True)
  verify = []
  assert packed is not None
  for i, (model, (seqs, mode)) in enumerate(zip(models, inputs)):
    cmp = shrinking.compare(model, packed.task(i), seqs, CFG["batch_size"],
                            CFG["tie_tol"])
    verify.append(dict(prog=names[i], verify=mode, sequences=cmp.sequences,
                       mismatches=cmp.decoded_mismatches,
                       tie_flips=cmp.tie_flips, max_abs_diff=cmp.max_abs_diff,
                       categorical=model.categorical_output))
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  row: dict[str, Any] = dict(
      V=V, N=N, ok=True, programs=programs, sweep=sweep, verify=verify,
      total_s=time.perf_counter() - t_start, peak_rss_mb=rss / 2**20)
  print(json.dumps(row), flush=True)


def run_one(V, N):
  cmd = [sys.executable, os.path.abspath(__file__), "--one", str(V), str(N)]
  if N == CFG["heavy_n"]:
    with HEAVY_SLOTS:
      return _run(cmd, V, N)
  return _run(cmd, V, N)


def _run(cmd, V, N):
  try:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       timeout=CFG["timeout_s"])
  except subprocess.TimeoutExpired:
    return dict(V=V, N=N, ok=False, err=f"timeout>{CFG['timeout_s']}s")
  lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
  if lines:
    return json.loads(lines[-1])
  tail = (r.stderr.strip().splitlines() or ["?"])[-1]
  return dict(V=V, N=N, ok=False, err=tail[:160])


def fmt_rows(r):
  if not r["ok"]:
    return [f"{r['V']:>4}{r['N']:>4}  FAIL {r['err']}"]
  v = r["verify"]
  seqs = sum(x["sequences"] for x in v)
  mism = sum(x["mismatches"] for x in v if x["categorical"])
  ties = sum(x["tie_flips"] for x in v)
  bits = sum(x["mismatches"] for x in v if not x["categorical"])
  diff = max(x["max_abs_diff"] for x in v)
  out = []
  for s in r["sweep"]:
    last = s["K"] == len(r["sweep"])
    check = (f"{seqs:>7,}{mism:>6}{ties:>6}{bits:>6}{diff:>10.1e}" if last
             else f"{'-':>7}{'-':>6}{'-':>6}{'-':>6}{'-':>10}")
    bound = s["slots_lower_bound"] + s["gate_slots"] + s["position_slots"]
    out.append(
        f"{r['V']:>4}{r['N']:>4}{s['K']:>4}{s['d_model']:>8}"
        f"{max(s['d_programs']):>7}{sum(s['d_programs']):>7}{bound:>7}"
        f"{s['solver']:>8}{s['params']:>12,}{s['params_block_diagonal']:>13,}"
        f"{s['separate_trim']:>12,}"
        f"{s['params_block_diagonal'] / s['params']:>9.2f}{check}"
        f"{s['max_gate']:>10.1e}{s['pack_s']:>7.1f}"
        f"{r['total_s'] if last else 0:>8.0f}")
  return out


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  jobs = [(V, N) for V in CFG["vocab_sizes"] for N in CFG["max_seq_lens"]]
  jobs = sorted(jobs, key=lambda j: j[1] != CFG["heavy_n"])
  results_path = os.path.join(OUT_DIR, "results.jsonl")
  log_path = os.path.join(OUT_DIR, "run.log")
  rows = []
  with open(results_path, "w") as out, open(log_path, "w") as log:

    def emit(line):
      print(line, flush=True)
      log.write(line + "\n")
      log.flush()

    emit(f"packing {len(CFG['programs'])} programs at {len(jobs)} (V, N) "
         f"configs on {CFG['workers']} workers, timeout {CFG['timeout_s']} s "
         f"each -> {OUT_DIR}")
    emit("K = programs packed; d_pack = packed residual width; d_max / d_sum "
         "= widest / summed width of the K trimmed programs; bound = clique "
         "lower bound on d_pack; packed / block / separate = dense params of "
         "the packed model, of block-diagonal stacking, and of K trimmed "
         "models; x_block = block / packed. The full pack (last K) is checked: "
         "mism = categorical sequences whose decoded output differs where the "
         "original is decided; ties = differ only where its top-2 scores are "
         f"within {CFG['tie_tol']:g}; nbits = numerical sequences not bitwise "
         "equal; maxdiff = largest output value difference.")
    emit("started: " + ", ".join(f"V={V} N={N}" for V, N in jobs))
    emit(HEADER)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(CFG["workers"]) as ex:
      futs = [ex.submit(run_one, *j) for j in jobs]
      for fu in as_completed(futs):
        r = fu.result()
        rows.append(r)
        out.write(json.dumps(r) + "\n")
        out.flush()
        for line in fmt_rows(r):
          emit(line)
    emit(f"done {len(rows):,} configs in {time.perf_counter() - t0:.0f} s")
    import pack_report_summary
    for line in pack_report_summary.write(rows, OUT_DIR, CFG):
      emit(line)


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--one":
    child(int(sys.argv[2]), int(sys.argv[3]))
  else:
    main()
