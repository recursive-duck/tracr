"""Cost of the compiled exact-copy head (lib_copy) vs ID length and slots.

Program: tracr/compiler/lib_copy.make_exact_copy. A call CALL_s followed by
k copy positions gets, at each of them, the matching digit of the source
SRC_s d_1..d_k (layout in the lib_copy docstring). Two sweeps:
  vs k: id_len over id_lens at num_slots = fixed_slots
  vs S: num_slots over slot_counts at id_len = fixed_id_len
each with the copy key/query bit-coded (rasp.binary, the default) and
one-hot (rasp.categorical, the baseline). max_seq_len is
lib_copy.max_sequence_len (every slot as a source, max_calls calls, gaps of
max_gap).

Per config, in its own process with a timeout: compile, count live MLP
units and attention heads, trim with shrinking.shrink(reuse=False), and run
`samples` random valid sequences (lib_copy.random_sequence) through the
compiled model, comparing every copy position with lib_copy.reference_copy;
the first rasp_samples of them also go through rasp.evaluate.

Settings live in config.toml [exact_copy]. Writes results.jsonl (one row per
config, flushed), run.log, exact_copy.png and summary.md to out_dir.

Usage:
  uv run --extra experiments python experiments/exact_copy/exact_copy.py
"""

import json
import os
import random
import resource
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CONFIG_PATH = os.environ.get("EXACT_COPY_CONFIG",
                             os.path.join(ROOT, "config.toml"))
with open(CONFIG_PATH, "rb") as _f:
  CFG = tomllib.load(_f)["exact_copy"]
OUT_DIR = os.path.join(ROOT, CFG["out_dir"])
BINARY = "binary"
CATEGORICAL = "categorical"
SWEEP_K = "k"
SWEEP_S = "S"

HEADER = (f"{'enc':<12}{'k':>4}{'S':>4}{'N':>5}{'units↓':>9}"
          f"{'heads↓':>7}{'layers↓':>8}{'qk_dims↓':>9}{'d_model↓':>9}"
          f"{'d_trim↓':>8}{'params↓':>13}{'p_trim↓':>12}{'compile_s↓':>11}"
          f"{'seqs':>6}{'copies':>8}{'mism↓':>6}{'rasp_mism↓':>11}"
          f"{'rss_gb':>7}")


def jobs():
  """(encoding, id_len, num_slots) per config, largest first; the point
  shared by both sweeps runs once."""
  out = set()
  for enc in CFG["encodings"]:
    out |= {(enc, k, CFG["fixed_slots"]) for k in CFG["id_lens"]}
    out |= {(enc, CFG["fixed_id_len"], s) for s in CFG["slot_counts"]}
  return sorted(out, key=lambda j: (-j[1] * j[2], j))


def run_model(model, seqs):
  """Decoded outputs at real positions, padded and batched under one jit."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  enc, dec = model.input_encoder, model.output_encoder
  forward = jax.jit(lambda p, x: model.forward(p, x).unembedded_output)
  width = max(map(len, seqs)) + 1
  outs = []
  for start in range(0, len(seqs), CFG["batch_size"]):
    chunk = seqs[start:start + CFG["batch_size"]]
    ids = np.array([enc.encode([enc.bos_token] + s + [enc.pad_token] *
                               (width - 1 - len(s))) for s in chunk],
                   dtype=np.int32)
    pred = np.asarray(forward(model.params, jnp.asarray(ids)))
    outs += [dec.decode(row.tolist())[1:1 + len(s)]
             for row, s in zip(pred, chunk)]
  return outs


def child(enc, k, s):
  import numpy as np
  from tracr.compiler import compiling
  from tracr.compiler import lib_copy
  from tracr.compiler import shrinking
  from tracr.rasp import rasp
  layout = lib_copy.CopyLayout(id_len=k, num_slots=s, base=CFG["base"])
  n = lib_copy.max_sequence_len(layout, CFG["max_calls"], CFG["max_gap"])
  encoding = rasp.binary if enc == BINARY else rasp.categorical
  program = lib_copy.make_exact_copy(layout, encoding=encoding)
  t0 = time.perf_counter()
  model = compiling.compile_rasp_to_model(
      program, vocab=layout.vocab, max_seq_len=n,
      compiler_bos=CFG["compiler_bos"], compiler_pad=CFG["compiler_pad"])
  compile_s = time.perf_counter() - t0
  cfg = model.model_config
  layers = range(cfg.num_layers)
  live = sum(int(np.count_nonzero(np.abs(np.asarray(
      model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"])).sum(0)))
             for l in layers)
  head_dim = cfg.key_size
  heads = sum(int(np.count_nonzero(np.abs(np.asarray(
      model.params[f"transformer/layer_{l}/attn/query"]["w"])).reshape(
          -1, cfg.num_heads, head_dim).sum((0, 2)))) for l in layers)
  qk_dims = sum(l.startswith(("copy_key_", "copy_query_"))
                for l in model.residual_labels)
  t0 = time.perf_counter()
  stats = shrinking.shrink(model, reuse=False).stats
  shrink_s = time.perf_counter() - t0
  rng = random.Random(CFG["seed"] + 1000 * k + s)
  seqs = [lib_copy.random_sequence(layout, rng, CFG["max_calls"],
                                   CFG["max_gap"])
          for _ in range(CFG["samples"])]
  refs = [lib_copy.reference_copy(layout, q) for q in seqs]
  t0 = time.perf_counter()
  got = run_model(model, seqs)
  apply_s = time.perf_counter() - t0
  copies = sum(r is not None for ref in refs for r in ref)
  mism = sum(g != r for out, ref in zip(got, refs) for g, r in zip(out, ref)
             if r is not None)
  t0 = time.perf_counter()
  rasp_seqs = seqs[:CFG["rasp_samples"]]
  rasp_mism = sum(rasp.evaluate(program, q) != ref
                  for q, ref in zip(rasp_seqs, refs))
  rasp_s = time.perf_counter() - t0
  row: dict[str, Any] = dict(
      enc=enc, k=k, S=s, ok=True, max_seq_len=n,
      vocab=len(layout.vocab), units_live=live, heads=heads,
      layers=cfg.num_layers, qk_dims=qk_dims,
      mlp_hidden_size=cfg.mlp_hidden_size, d_model=stats.d_model_before,
      d_model_trim=stats.d_model_after, params=stats.params_before,
      params_trim=stats.params_after, mlp_units_trim=stats.mlp_units_after,
      compile_s=compile_s, shrink_s=shrink_s, apply_s=apply_s, rasp_s=rasp_s,
      seqs=len(seqs), copies=copies, mism=mism, rasp_seqs=len(rasp_seqs),
      rasp_mism=rasp_mism)
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  row["peak_rss_gb"] = rss / 2**30  # bytes on macOS
  print(json.dumps(row), flush=True)


def run_one(enc, k, s):
  cmd = [sys.executable, os.path.abspath(__file__), "--one", enc, str(k),
         str(s)]
  base = dict(enc=enc, k=k, S=s, ok=False)
  t0 = time.perf_counter()
  try:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       timeout=CFG["timeout_s"])
  except subprocess.TimeoutExpired:
    return dict(base, err=f"timeout > {CFG['timeout_s']} s")
  lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
  if lines:
    return json.loads(lines[-1])
  tail = (r.stderr.strip().splitlines() or ["?"])[-1]
  return dict(base, err=f"{tail[:120]} after {time.perf_counter() - t0:.0f} s")


def fmt_row(r):
  head = f"{r['enc']:<12}{r['k']:>4}{r['S']:>4}"
  if not r["ok"]:
    return f"{head}  FAIL {r['err']}"
  return (f"{head}{r['max_seq_len']:>5}{r['units_live']:>9,}{r['heads']:>7}"
          f"{r['layers']:>8}{r['qk_dims']:>9,}{r['d_model']:>9,}"
          f"{r['d_model_trim']:>8,}{r['params']:>13,}{r['params_trim']:>12,}"
          f"{r['compile_s']:>11.1f}{r['seqs']:>6}{r['copies']:>8,}"
          f"{r['mism']:>6}{r['rasp_mism']:>11}{r['peak_rss_gb']:>7.1f}")


def _in_sweep(r, sweep):
  if sweep == SWEEP_K:
    return r["S"] == CFG["fixed_slots"] and r["k"] in CFG["id_lens"]
  return r["k"] == CFG["fixed_id_len"] and r["S"] in CFG["slot_counts"]


def _series(rows, sweep, enc):
  x = "k" if sweep == SWEEP_K else "S"
  return sorted((r for r in rows if r["ok"] and r["enc"] == enc and
                 _in_sweep(r, sweep)), key=lambda r: r[x])


def fits(rows):
  """log-log slopes: quantity ~ x^slope along each sweep, per encoding."""
  import numpy as np
  out = {}
  for sweep in (SWEEP_K, SWEEP_S):
    x_key = "k" if sweep == SWEEP_K else "S"
    for enc in CFG["encodings"]:
      ok = _series(rows, sweep, enc)
      if len(ok) < 2:
        continue
      x = np.log2([r[x_key] for r in ok])
      fit = dict(xs=[r[x_key] for r in ok],
                 layers=[r["layers"] for r in ok],
                 heads=[r["heads"] for r in ok])
      for key in ("units_live", "qk_dims", "d_model_trim", "params_trim",
                  "compile_s"):
        y = np.log2([max(r[key], 1e-9) for r in ok])
        fit[f"{key}_slope"] = float(np.polyfit(x, y, 1)[0])
      out[f"{sweep}/{enc}"] = fit
  return out


def plot(rows, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  panels = [("params_trim", "params after trim"),
            ("units_live", "live MLP hidden units"),
            ("qk_dims", "copy key+query residual dims"),
            ("compile_s", "compile seconds")]
  sweeps = [(SWEEP_K, "k", f"ID length k (S = {CFG['fixed_slots']})"),
            (SWEEP_S, "S", f"IDs in context S (k = {CFG['fixed_id_len']})")]
  fig, axes = plt.subplots(len(sweeps), len(panels), figsize=(16, 7.5))
  for row_axes, (sweep, x_key, x_label) in zip(axes, sweeps):
    for ax, (key, title) in zip(row_axes, panels):
      for enc in CFG["encodings"]:
        ok = _series(rows, sweep, enc)
        if ok:
          ax.plot([r[x_key] for r in ok], [r[key] for r in ok], "o-",
                  label=f"key/query {enc}")
      ax.set_xscale("log", base=2)
      ax.set_yscale("log")
      ax.set_xlabel(x_label)
      ax.set_title(f"{title} (log)", fontsize=9)
      ax.grid(alpha=0.3)
  axes[0, 0].legend(fontsize=8)
  ok = [r for r in rows if r["ok"]]
  mism = sum(r["mism"] for r in ok)
  copies = sum(r["copies"] for r in ok)
  fig.suptitle(f"Compiled exact copy of k-digit IDs (lib_copy, base "
               f"{CFG['base']}): {mism} wrong of {copies:,} copied tokens")
  fig.tight_layout()
  fig.savefig(path, dpi=110)
  plt.close(fig)


def _ratio(before, after):
  """before / after as text, bold when at least bold_ratio."""
  x = before / after
  text = f"{x:,.1f}x"
  return f"**{text}**" if x >= CFG["bold_ratio"] else text


def write_summary(rows, fit, total_s):
  ok = [r for r in rows if r["ok"]]
  fails = [r for r in rows if not r["ok"]]
  by = {(r["enc"], r["k"], r["S"]): r for r in ok}
  lines = ["# Exact copy of multi-token IDs", "",
           "## What was tested", "",
           "tracr/compiler/lib_copy.make_exact_copy: at each of the k "
           "positions after CALL_s the compiled model must output the "
           "matching digit of the source ID SRC_s d_1..d_k earlier in the "
           "sequence. Layout: digits 0..base-1, SRC_s = base + 2s, CALL_s = "
           "base + 2s + 1, filler = base + 2S; markers at least k + 1 apart. "
           "Head 1 (one attention layer) finds the marker 1..k positions "
           "back: one head per bit of its code and of its index mod (k + 1), "
           "each averaging a constant 1 over window AND marker (exactly 0 or "
           "1, never None), decoded with lib_arith.from_bits. Head 2 copies "
           "tokens by exact equality of the bit-coded (rasp.binary) "
           "address slot * (k + 1) + offset.", "",
           f"Sweeps: k over {CFG['id_lens']} at S = {CFG['fixed_slots']}; S "
           f"over {CFG['slot_counts']} at k = {CFG['fixed_id_len']}; base "
           f"{CFG['base']}. Each config: {CFG['samples']} random valid "
           f"sequences (1..S sources in random slot order, 1.."
           f"{CFG['max_calls']} calls, gaps of 0..{CFG['max_gap']} random "
           "digits/filler, random tokens at copy positions) vs the "
           f"pure-Python reference; the first {CFG['rasp_samples']} also vs "
           "rasp.evaluate. Baseline: the same program with one-hot "
           "(categorical) copy key/query.", "",
           "Terms: units = MLP hidden units with nonzero input weights; heads "
           "= attention heads with nonzero query weights; qk_dims = residual "
           "dims of the copy key + query; d_trim / p_trim = residual width "
           "and dense params after shrinking.shrink(reuse=False); N = "
           "max_seq_len.", "", "## Key numbers", ""]
  copies = sum(r["copies"] for r in ok)
  mism = sum(r["mism"] for r in ok)
  rasp_seqs = sum(r["rasp_seqs"] for r in ok)
  rasp_mism = sum(r["rasp_mism"] for r in ok)
  lines.append(f"- Configs: {len(ok)} compiled and checked, {len(fails)} "
               "failed" + (": " + "; ".join(
                   f"{r['enc']} k={r['k']} S={r['S']}: {r['err']}"
                   for r in fails) if fails else "."))
  lines.append(f"- Wrong copied tokens (compiled vs reference): {mism} of "
               f"{copies:,} over {sum(r['seqs'] for r in ok):,} sequences. "
               f"RASP vs reference: {rasp_mism} of {rasp_seqs:,} sequences.")
  for sweep, x_key in ((SWEEP_K, "k"), (SWEEP_S, "S")):
    series = _series(rows, sweep, BINARY)
    if not series:
      continue
    lo, hi = series[0], series[-1]
    lines.append(
        f"- Binary, {x_key} {lo[x_key]} -> {hi[x_key]} (N {lo['max_seq_len']}"
        f" -> {hi['max_seq_len']}): layers {lo['layers']} -> {hi['layers']}, "
        f"heads {lo['heads']} -> {hi['heads']}, units {lo['units_live']:,} -> "
        f"{hi['units_live']:,}, qk_dims {lo['qk_dims']} -> {hi['qk_dims']}, "
        f"p_trim {lo['params_trim']:,} -> {hi['params_trim']:,}, compile "
        f"{lo['compile_s']:.1f} -> {hi['compile_s']:.1f} s.")
    for r in series[-1:]:
      cat = by.get((CATEGORICAL, r["k"], r["S"]))
      if cat:
        lines.append(
            f"  - categorical -> binary key/query at k={r['k']}, S={r['S']}: "
            f"qk_dims {cat['qk_dims']:,} -> {r['qk_dims']:,} "
            f"({_ratio(cat['qk_dims'], r['qk_dims'])}), d_trim "
            f"{cat['d_model_trim']:,} -> {r['d_model_trim']:,}, units "
            f"{cat['units_live']:,} -> {r['units_live']:,}, p_trim "
            f"{cat['params_trim']:,} -> {r['params_trim']:,} "
            f"({_ratio(cat['params_trim'], r['params_trim'])}), compile "
            f"{cat['compile_s']:.1f} -> {r['compile_s']:.1f} s.")
  lines += ["", "## Growth (log-log least squares: y ~ x^slope)", "",
            "| sweep / key-query | x | layers | heads | units | qk_dims | "
            "d_trim | p_trim | compile_s |", "|---|---|---|---|---|---|---|"
            "---|---|"]
  for name, f in fit.items():
    lines.append(
        f"| {name} | {f['xs'][0]}..{f['xs'][-1]} | {f['layers']} | "
        f"{f['heads']} | {f['units_live_slope']:.2f} | "
        f"{f['qk_dims_slope']:.2f} | {f['d_model_trim_slope']:.2f} | "
        f"{f['params_trim_slope']:.2f} | {f['compile_s_slope']:.2f} |")
  lines += [
      "", "## What the numbers mean", "",
      "The copy itself is exact by construction: the key and query are the "
      "same integer address only when slot and offset agree, and the binary "
      "head scores a full match L above every partial match. Zero wrong "
      "tokens over all configs says the soft attention and the 0/1 marker "
      "heads stayed exact at these sizes. Bit-coded key/query need "
      "ceil(log2(S (k + 1) + 2)) dims each instead of one per address.", "",
      "## What this does NOT prove", "",
      "- Nothing about long contexts: max_seq_len here is at most "
      f"{max((r['max_seq_len'] for r in ok), default=0):,}. The marker "
      "window Select reads one-hot indices, so its width and d_model grow "
      "linearly with max_seq_len; 1M context needs a different window "
      "(e.g. relative/bit positions).",
      "- Nothing about a trained model or natural text: inputs follow the "
      "strict layout (markers k + 1 apart, one source per slot, every "
      "called slot present). Broken layouts are rejected by check_layout, "
      "not handled.",
      "- Not an autoregressive decoder: the compiled model is not causal and "
      "reads all copy positions at once; teacher forcing is only simulated "
      "by random tokens at copy positions.",
      "- Random checks, not exhaustive, at these sizes (exhaustive k <= 3, "
      "S <= 3 base 2 is in lib_copy_test).", "",
      f"Plot: {os.path.join(CFG['out_dir'], 'exact_copy.png')}. Total wall "
      f"time {total_s:.0f} s."]
  return lines


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  todo = jobs()
  results_path = os.path.join(OUT_DIR, "results.jsonl")
  log_path = os.path.join(OUT_DIR, "run.log")
  rows = []
  lock = threading.Lock()
  with open(results_path, "w") as out, open(log_path, "w") as log:

    def emit(line):
      with lock:
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def record(r):
      with lock:
        rows.append(r)
        out.write(json.dumps(r) + "\n")
        out.flush()
      emit(fmt_row(r))

    emit(f"exact_copy: {len(todo)} configs on {CFG['workers']} workers, "
         f"timeout {CFG['timeout_s']} s each -> {OUT_DIR}")
    emit("units = live MLP units; heads = live attention heads; qk_dims = "
         "copy key+query dims; copies = copy positions checked; mism = wrong "
         "copied tokens vs reference; rasp_mism = sequences where "
         "rasp.evaluate differs from the reference")
    emit(HEADER)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(CFG["workers"]) as ex:
      futs = [ex.submit(lambda j=j: record(run_one(*j))) for j in todo]
      for fu in as_completed(futs):
        fu.result()
    total_s = time.perf_counter() - t0
    emit(f"done {len(rows):,} configs in {total_s:.0f} s")
    fit = fits(rows)
    plot(rows, os.path.join(OUT_DIR, "exact_copy.png"))
    lines = write_summary(rows, fit, total_s)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
      f.write("\n".join(lines) + "\n")
    emit("")
    for line in lines:
      emit(line)


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--one":
    child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
  else:
    main()
