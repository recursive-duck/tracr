"""Lossless shrinking across tracr lib programs, vocab size V and length N.

For each (program, V, N): compile, shrink with trim only and with dim reuse,
then compare each shrunk model with the original on every input (small
domains) or on random inputs (large ones), and the original and each shrunk
model with the RASP program itself. Settings live in config.toml
[shrink_report]. Writes results.jsonl (one row per config, flushed), run.log,
shrink_report.png and summary.md to out_dir.

Usage:
  uv run --extra experiments python experiments/shrink_report.py
"""

import itertools
import json
import os
import random
import resource
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Another config file (e.g. a tiny smoke-test grid) can be set in the env;
# child processes inherit it.
CONFIG_PATH = os.environ.get("SHRINK_REPORT_CONFIG",
                             os.path.join(ROOT, "config.toml"))
with open(CONFIG_PATH, "rb") as _f:
  CFG = tomllib.load(_f)["shrink_report"]
OUT_DIR = os.path.join(ROOT, CFG["out_dir"])

HEADER = (f"{'program':<16}{'V':>4}{'N':>4}{'params':>15}{'trim↓':>12}"
          f"{'reuse↓':>12}{'x_trim↑':>9}{'x_reuse↑':>10}{'d':>6}{'d_trim↓':>8}"
          f"{'d_reuse↓':>9}{'seqs':>7}{'mism↓':>6}{'ties':>6}{'rasp↓':>6}"
          f"{'maxdiff↓':>10}"
          f"{'compile_s':>10}{'shrink_s':>9}")


def build(prog, V, N):
  from tracr.compiler import lib
  from tracr.rasp import rasp
  ints = set(range(V))
  extra = max(0, V - len(CFG["dyck_tokens"]))
  strs = set(CFG["dyck_tokens"]) | {f"a{i}" for i in range(extra)}
  makers = {
      "length": lambda: (lib.make_length(), ints),
      "reverse": lambda: (lib.make_reverse(rasp.tokens), ints),
      "hist": lambda: (lib.make_hist(), ints),
      "sort_unique": lambda: (lib.make_sort_unique(rasp.tokens, rasp.tokens),
                              ints),
      "sort": lambda: (lib.make_sort(rasp.tokens, rasp.tokens, max_seq_len=N,
                                     min_key=1), ints),
      "sort_freq": lambda: (lib.make_sort_freq(max_seq_len=N), ints),
      "frac_prevs": lambda: (lib.make_frac_prevs(rasp.tokens == "("), strs),
      "pair_balance": lambda: (lib.make_pair_balance(rasp.tokens, "(", ")"),
                               strs),
      "shuffle_dyck2": lambda: (lib.make_shuffle_dyck2(), strs),
      "count_less_freq": lambda: (lib.make_count_less_freq(n=2), ints),
  }
  return makers[prog]()


def inputs_for(vocab, N, seed):
  vocab = sorted(vocab, key=str)
  domain = sum(len(vocab)**n for n in range(1, N + 1))
  if domain <= CFG["exhaustive_limit"]:
    seqs = [list(x) for n in range(1, N + 1)
            for x in itertools.product(vocab, repeat=n)]
    return seqs, "exhaustive", domain
  rng = random.Random(seed)
  seqs = []
  for _ in range(CFG["samples"]):
    length = rng.randint(1, N)
    seqs.append([rng.choice(vocab) for _ in range(length)])
  return seqs, "sampled", domain


def child(prog, V, N):
  import numpy as np
  from tracr.compiler import compiling
  from tracr.compiler import shrinking
  import rasp_check
  program, vocab = build(prog, V, N)
  t0 = time.perf_counter()
  model = compiling.compile_rasp_to_model(
      program, vocab=vocab, max_seq_len=N, compiler_bos=CFG["compiler_bos"])
  compile_s = time.perf_counter() - t0
  nnz = sum(int(np.count_nonzero(w)) for mod in model.params.values()
            for w in mod.values())
  seqs, mode, domain = inputs_for(vocab, N, CFG["seed"])
  row: dict[str, Any] = dict(
      prog=prog, V=V, N=N, ok=True, compile_s=compile_s, nnz=nnz,
      verify=mode, domain=domain, layers=model.model_config.num_layers)
  expected = rasp_check.expected_outputs(program, seqs)
  tol = dict(batch_size=CFG["batch_size"], atol=CFG["rasp_atol"],
             rtol=CFG["rasp_rtol"])
  vs_rasp = rasp_check.check(rasp_check.original_output_values(model), model,
                             seqs, expected, **tol)
  row["rasp_mismatches"] = vs_rasp.mismatches
  row["rasp_max_abs_err"] = vs_rasp.max_abs_err
  for reuse in (False, True):
    key = "reuse" if reuse else "trim"
    t0 = time.perf_counter()
    shrunk = shrinking.shrink(model, reuse=reuse)
    shrink_s = time.perf_counter() - t0
    cmp = shrinking.compare(model, shrunk, seqs, CFG["batch_size"],
                            CFG["tie_tol"])
    vs_rasp = rasp_check.check(shrunk.output_values, model, seqs, expected,
                               **tol)
    row[key] = dict(vars(shrunk.stats), shrink_s=shrink_s,
                    sequences=cmp.sequences,
                    mismatches=cmp.decoded_mismatches,
                    tie_flips=cmp.tie_flips,
                    max_abs_diff=cmp.max_abs_diff,
                    rasp_mismatches=vs_rasp.mismatches,
                    rasp_max_abs_err=vs_rasp.max_abs_err,
                    categorical=shrunk.categorical_output)
    del shrunk
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  row["peak_rss_mb"] = rss / 2**20  # bytes on macOS
  print(json.dumps(row), flush=True)


def run_one(prog, V, N):
  cmd = [sys.executable, os.path.abspath(__file__), "--one", prog, str(V),
         str(N)]
  try:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       timeout=CFG["timeout_s"])
  except subprocess.TimeoutExpired:
    return dict(prog=prog, V=V, N=N, ok=False,
                err=f"timeout>{CFG['timeout_s']}s")
  lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
  if lines:
    return json.loads(lines[-1])
  tail = (r.stderr.strip().splitlines() or ["?"])[-1]
  return dict(prog=prog, V=V, N=N, ok=False, err=tail[:160])


def fmt_row(r):
  if not r["ok"]:
    return f"{r['prog']:<16}{r['V']:>4}{r['N']:>4}  FAIL {r['err']}"
  t, u = r["trim"], r["reuse"]
  mism = t["mismatches"] + u["mismatches"]
  ties = t["tie_flips"] + u["tie_flips"]
  diff = max(t["max_abs_diff"], u["max_abs_diff"])
  rasp = r["rasp_mismatches"] + t["rasp_mismatches"] + u["rasp_mismatches"]
  return (f"{r['prog']:<16}{r['V']:>4}{r['N']:>4}{t['params_before']:>15,}"
          f"{t['params_after']:>12,}{u['params_after']:>12,}"
          f"{t['params_before'] / t['params_after']:>9.1f}"
          f"{t['params_before'] / u['params_after']:>10.1f}"
          f"{t['d_model_before']:>6}{t['d_model_after']:>8}"
          f"{u['d_model_after']:>9}{t['sequences']:>7,}{mism:>6}{ties:>6}{rasp:>6}"
          f"{diff:>10.1e}"
          f"{r['compile_s']:>10.1f}{t['shrink_s'] + u['shrink_s']:>9.1f}")


def known_params():
  """Dense params per config from an earlier run, to start heavy ones first."""
  path = os.path.join(OUT_DIR, "results.jsonl")
  sizes = {}
  if os.path.exists(path):
    for line in open(path):
      r = json.loads(line)
      if r.get("ok"):
        sizes[(r["prog"], r["V"], r["N"])] = r["trim"]["params_before"]
  return sizes


def heavy_first(jobs):
  sizes = known_params()
  big_n = max(CFG["max_seq_lens"])

  def heavy(job):
    size = sizes.get(job)
    if size is not None:
      return size >= CFG["heavy_params"]
    return job[0] == "shuffle_dyck2" and job[2] == big_n

  return sorted(jobs, key=lambda j: not heavy(j))


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  jobs = [(p, V, N) for p in CFG["programs"] for V in CFG["vocab_sizes"]
          for N in CFG["max_seq_lens"]]
  jobs = heavy_first(jobs)
  results_path = os.path.join(OUT_DIR, "results.jsonl")
  log_path = os.path.join(OUT_DIR, "run.log")
  rows = []
  with open(results_path, "w") as out, open(log_path, "w") as log:

    def emit(line):
      print(line, flush=True)
      log.write(line + "\n")
      log.flush()

    emit(f"shrinking {len(jobs):,} configs on {CFG['workers']} workers, "
         f"timeout {CFG['timeout_s']} s each -> {OUT_DIR}")
    emit("x_trim / x_reuse = dense params before / after; mism = sequences "
         "whose output differs where the original is decided (both modes): "
         "decoded token for categorical programs, any bit for numerical ones; "
         f"ties = sequences differing only where the original's top-2 scores "
         f"are within {CFG['tie_tol']:g}; rasp = sequences whose decoded "
         "output differs from the RASP program, summed over the original, "
         "trim and reuse (positions where RASP gives None skipped; numerical "
         f"within {CFG['rasp_atol']:g} + {CFG['rasp_rtol']:g} * |RASP|); "
         "maxdiff = largest output value difference")
    emit(HEADER)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(CFG["workers"]) as ex:
      futs = [ex.submit(run_one, *j) for j in jobs]
      for fu in as_completed(futs):
        r = fu.result()
        rows.append(r)
        out.write(json.dumps(r) + "\n")
        out.flush()
        emit(fmt_row(r))
    emit(f"done {len(rows):,} configs in {time.perf_counter() - t0:.0f} s")
    import shrink_report_summary
    for line in shrink_report_summary.write(rows, OUT_DIR, CFG):
      emit(line)


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--one":
    child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
  else:
    main()
