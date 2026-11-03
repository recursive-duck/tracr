"""Cost of a + b as one lookup table vs as bit circuits (lib_arith), w = 2..10.

Each program reads a = tokens (vocab 0..2^w-1) and b = indices (max_seq_len
2^w) and outputs a + b as one categorical value:
  lookup:       SequenceMap(lambda a, b: a + b, tokens, indices)
  ripple_*:     from_bits(ripple_add(to_bits(a), to_bits(b), fusion))
  kogge_stone:  from_bits(kogge_stone_add(to_bits(a), to_bits(b)))
Per (method, w), in its own process with a timeout: compile, count MLP units,
trim with shrinking.shrink(reuse=False), and compare the compiled model with
rasp.evaluate and with integer a + b on random full-length inputs.

Units are split by part (from basis inference value sets, the same sizes
expr_to_craft_graph builds): encode = the to_bits Maps, circuit = the adder
(or the lookup table), decode = from_bits. units_live counts MLP units with a
nonzero input weight in the compiled model.

Settings live in config.toml [bit_arith]. Writes results.jsonl (one row per
config, flushed), run.log, bit_arith.png and summary.md to out_dir.

Usage:
  uv run --extra experiments python experiments/bit_arith/bit_arith.py
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
CONFIG_PATH = os.environ.get("BIT_ARITH_CONFIG",
                             os.path.join(ROOT, "config.toml"))
with open(CONFIG_PATH, "rb") as _f:
  CFG = tomllib.load(_f)["bit_arith"]
OUT_DIR = os.path.join(ROOT, CFG["out_dir"])
LOOKUP = "lookup"
PARTS = ("encode", "circuit", "decode")

HEADER = (f"{'method':<13}{'w':>3}{'units↓':>11}{'live↓':>11}{'circuit↓':>10}"
          f"{'decode↓':>9}{'layers↓':>8}{'d_model':>8}{'d_trim↓':>8}"
          f"{'params':>15}{'p_trim↓':>13}{'compile_s↓':>11}{'seqs':>6}"
          f"{'mism_rasp↓':>11}{'mism_int↓':>10}{'rss_gb':>7}")


def build(method, w):
  """(program, encode S-Ops, circuit output S-Ops) of a + b at width w."""
  from tracr.compiler import lib_arith
  from tracr.rasp import rasp
  if method == LOOKUP:
    program = rasp.categorical(
        rasp.SequenceMap(lambda x, y: x + y, rasp.tokens, rasp.indices))
    return program, [], [program]
  a = lib_arith.to_bits(rasp.tokens, w)
  b = lib_arith.to_bits(rasp.indices, w)
  if method == "kogge_stone":
    out = lib_arith.kogge_stone_add(a, b)
  else:
    out = lib_arith.ripple_add(a, b, fusion=method.removeprefix("ripple_"))
  return lib_arith.from_bits(out), a + b, out


def units_by_part(program, encode, circuit_out, vocab, n):
  """Hidden units of each MLP from basis inference, summed per part."""
  import networkx as nx
  from tracr.compiler import basis_inference
  from tracr.compiler import nodes
  from tracr.compiler import rasp_to_graph
  from tracr.rasp import rasp
  extracted = rasp_to_graph.extract_rasp_graph(program)
  graph = extracted.graph
  basis_inference.infer_bases(graph, extracted.sink, vocab, n)
  encode_labels = {s.label for s in encode}
  circuit_labels = set()
  for sop in circuit_out:
    circuit_labels |= nx.ancestors(graph, sop.label) | {sop.label}
  units = dict.fromkeys(PARTS, 0)

  def width(sop):
    return len(graph.nodes[sop.label][nodes.OUTPUT_BASIS])

  for label, node in graph.nodes.items():
    expr = node[nodes.EXPR]
    if isinstance(expr, rasp.SequenceMap):
      hidden = (width(expr.fst) if width(expr.fst) == width(expr.snd) and
                expr.fst is expr.snd else width(expr.fst) * width(expr.snd))
    elif isinstance(expr, rasp.Map):
      hidden = width(expr)
    else:
      continue
    if label in encode_labels:
      units["encode"] += hidden
    elif label in circuit_labels:
      units["circuit"] += hidden
    else:
      units["decode"] += hidden
  return units


def random_inputs(w, seed):
  """Full-length random inputs: `samples` of them, capped so all of them
  together have at most dense_token_budget tokens (the dense forward pass
  costs about 0.5 GFLOP per token at w=10)."""
  n = 2**w
  count = min(CFG["samples"], max(1, CFG["dense_token_budget"] // (n + 1)))
  rng = random.Random(seed)
  return [[rng.randrange(n) for _ in range(n)] for _ in range(count)]


def run_model(model, seqs):
  """Decoded outputs at real positions, batched under one jit."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  enc, dec = model.input_encoder, model.output_encoder
  # Returning only the output lets XLA drop the per-layer residuals and
  # attention logits that model.forward also returns.
  forward = jax.jit(lambda p, x: model.forward(p, x).unembedded_output)
  outs = []
  for start in range(0, len(seqs), CFG["batch_size"]):
    chunk = seqs[start:start + CFG["batch_size"]]
    ids = np.array([enc.encode([enc.bos_token] + s) for s in chunk],
                   dtype=np.int32)
    pred = np.asarray(forward(model.params, jnp.asarray(ids)))
    outs += [dec.decode(row.tolist())[1:] for row in pred]
  return outs


def child(method, w):
  import numpy as np
  from tracr.compiler import compiling
  from tracr.compiler import shrinking
  from tracr.rasp import rasp
  n = 2**w
  vocab = set(range(n))
  program, encode, circuit_out = build(method, w)
  units = units_by_part(program, encode, circuit_out, vocab, n)
  t0 = time.perf_counter()
  model = compiling.compile_rasp_to_model(
      program, vocab=vocab, max_seq_len=n, compiler_bos=CFG["compiler_bos"])
  compile_s = time.perf_counter() - t0
  cfg = model.model_config
  live = sum(int(np.count_nonzero(np.abs(np.asarray(
      model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"])).sum(0)))
             for l in range(cfg.num_layers))
  t0 = time.perf_counter()
  stats = shrinking.shrink(model, reuse=False).stats
  shrink_s = time.perf_counter() - t0
  seqs = random_inputs(w, CFG["seed"] + w)
  t0 = time.perf_counter()
  got = run_model(model, seqs)
  apply_s = time.perf_counter() - t0
  t0 = time.perf_counter()
  want = [rasp.evaluate(program, s) for s in seqs]
  rasp_s = time.perf_counter() - t0
  ints = [[a + b for b, a in enumerate(s)] for s in seqs]
  row: dict[str, Any] = dict(
      method=method, w=w, ok=True, units=sum(units.values()),
      units_live=live, **{f"units_{p}": units[p] for p in PARTS},
      layers=cfg.num_layers, mlp_hidden_size=cfg.mlp_hidden_size,
      d_model=stats.d_model_before, d_model_trim=stats.d_model_after,
      params=stats.params_before, params_trim=stats.params_after,
      mlp_units_trim=stats.mlp_units_after, compile_s=compile_s,
      shrink_s=shrink_s, apply_s=apply_s, rasp_s=rasp_s, seqs=len(seqs),
      mism_rasp=sum(g != r for g, r in zip(got, want)),
      mism_int=sum(g != i for g, i in zip(got, ints)),
      rasp_vs_int=sum(r != i for r, i in zip(want, ints)))
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  row["peak_rss_gb"] = rss / 2**30  # bytes on macOS
  print(json.dumps(row), flush=True)


def lookup_estimate_gb(w):
  """Dense MLP weights of the lookup table: 4^w units x (2 D) floats."""
  d = 2**w + 2**w + 2**(w + 1)  # token, index and output dims
  return 4**w * 2 * d * CFG["bytes_per_param"] / 2**30


def run_one(method, w):
  if method == LOOKUP and lookup_estimate_gb(w) > CFG["mem_budget_gb"]:
    return dict(method=method, w=w, ok=False,
                err=f"skipped: est. {lookup_estimate_gb(w):,.1f} GB of MLP "
                    f"weights > {CFG['mem_budget_gb']} GB budget")
  cmd = [sys.executable, os.path.abspath(__file__), "--one", method, str(w)]
  timeout = CFG["timeout_s"] if method == LOOKUP else CFG["bit_timeout_s"]
  t0 = time.perf_counter()
  try:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       timeout=timeout)
  except subprocess.TimeoutExpired:
    return dict(method=method, w=w, ok=False, err=f"timeout > {timeout} s")
  lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
  if lines:
    return json.loads(lines[-1])
  tail = (r.stderr.strip().splitlines() or ["?"])[-1]
  return dict(method=method, w=w, ok=False,
              err=f"{tail[:120]} after {time.perf_counter() - t0:.0f} s")


def run_lookup_chain():
  """Lookup widths in increasing order, stopping at the first failure."""
  for w in CFG["widths"]:
    r = run_one(LOOKUP, w)
    yield r
    if not r["ok"]:
      return


def fmt_row(r):
  if not r["ok"]:
    return f"{r['method']:<13}{r['w']:>3}  FAIL {r['err']}"
  return (f"{r['method']:<13}{r['w']:>3}{r['units']:>11,}{r['units_live']:>11,}"
          f"{r['units_circuit']:>10,}{r['units_decode']:>9,}{r['layers']:>8}"
          f"{r['d_model']:>8,}{r['d_model_trim']:>8,}{r['params']:>15,}"
          f"{r['params_trim']:>13,}{r['compile_s']:>11.1f}{r['seqs']:>6}"
          f"{r['mism_rasp']:>11}{r['mism_int']:>10}{r['peak_rss_gb']:>7.1f}")


def fits(rows):
  """Growth of units, circuit units and compile time with w per method."""
  import numpy as np
  out = {}
  for method in [LOOKUP] + CFG["methods"]:
    ok = sorted((r for r in rows if r["method"] == method and r["ok"]),
                key=lambda r: r["w"])
    if len(ok) < 2:
      continue
    w = np.array([r["w"] for r in ok], float)
    fit = dict(widths=[int(x) for x in w])
    # Exponential growth: log2(y) = slope * w + c, so y ~ 2^(slope w).
    for key in ("units", "units_circuit", "params_trim", "compile_s"):
      y = np.array([r[key] for r in ok], float)
      fit[f"{key}_log2_slope"] = float(np.polyfit(w, np.log2(y), 1)[0])
    # Linear growth of the circuit alone: units = a * w + b (or w log2 w).
    xs = w * np.log2(w) if method == "kogge_stone" else w
    y = np.array([r["units_circuit"] for r in ok], float)
    a, b = np.polyfit(xs, y, 1)
    fit["circuit_lin"] = dict(x="w*log2(w)" if method == "kogge_stone" else "w",
                              a=float(a), b=float(b),
                              max_abs_resid=float(np.abs(
                                  a * xs + b - y).max()))
    layers = np.array([r["layers"] for r in ok], float)
    fit["layers"] = [int(x) for x in layers]
    out[method] = fit
  return out


def plot(rows, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  methods = [LOOKUP] + CFG["methods"]
  fig, axes = plt.subplots(2, 2, figsize=(12, 8))
  panels = [("units", "MLP hidden units (total, log)", True),
            ("params_trim", "params after trim (log)", True),
            ("layers", "layers", False),
            ("compile_s", "compile seconds (log)", True)]
  for ax, (key, title, log) in zip(axes.flat, panels):
    for m in methods:
      ok = sorted((r for r in rows if r["method"] == m and r["ok"]),
                  key=lambda r: r["w"])
      if not ok:
        continue
      line, = ax.plot([r["w"] for r in ok], [r[key] for r in ok], "o-",
                      label=m)
      if key == "units" and m != LOOKUP:
        ax.plot([r["w"] for r in ok], [r["units_circuit"] for r in ok], "--",
                color=line.get_color(), alpha=0.6)
    ax.set_title(title + (" ; dashed = adder circuit only"
                          if key == "units" else ""), fontsize=9)
    ax.set_xlabel("bit width w")
    if log:
      ax.set_yscale("log")
    ax.grid(alpha=0.3)
  axes[0, 0].legend(fontsize=8)
  fig.suptitle("a + b on 0..2^w-1: lookup table vs bit circuits (lib_arith)")
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
  by = {(r["method"], r["w"]): r for r in ok}
  lookup_ws = sorted(r["w"] for r in ok if r["method"] == LOOKUP)
  fails = [r for r in rows if not r["ok"]]
  w_cmp = lookup_ws[-1] if lookup_ws else None
  lines = ["# Bit arithmetic vs lookup table", "",
           "## What was tested", "",
           "a + b for a = tokens (vocab 0..2^w-1) and b = indices "
           "(max_seq_len 2^w), compiled as one lookup-table SequenceMap and "
           "as bit circuits from tracr/compiler/lib_arith.py, for w = "
           f"{CFG['widths'][0]}..{CFG['widths'][-1]}. Every bit program "
           "reads its bits with to_bits Maps (encode) and returns one "
           "categorical value through from_bits (decode). Each compiled model "
           f"ran on {CFG['samples']} random full-length inputs (seed "
           f"{CFG['seed']} + w; fewer when they would exceed "
           f"{CFG['dense_token_budget']:,} tokens, see seqs) against "
           "rasp.evaluate and integer a + b.", "",
           "Terms: units = MLP hidden units (one ReLU each) from the value "
           "sets; circuit = units of the adder alone (or of the table); "
           "layers = attention+MLP layer pairs; d_trim / p_trim = residual "
           "width and dense params after shrinking.shrink(reuse=False).", "",
           "## Key numbers", ""]
  lines.append(f"- Configs: {len(ok)} compiled and checked, {len(fails)} "
               "not: " + "; ".join(f"{r['method']} w={r['w']}: {r['err']}"
                                   for r in fails))
  mism = sum(r["mism_rasp"] + r["mism_int"] for r in ok)
  seqs = sum(r["seqs"] for r in ok)
  lines.append(f"- Mismatching sequences vs RASP + vs integers: {mism} of "
               f"{2 * seqs:,} comparisons.")
  if w_cmp is not None:
    lk = by[(LOOKUP, w_cmp)]
    lines.append(f"- Largest lookup that compiled: w={w_cmp}, "
                 f"{lk['units']:,} units, {lk['params_trim']:,} trimmed "
                 f"params, {lk['compile_s']:.1f} s.")
    for m in CFG["methods"]:
      r = by.get((m, w_cmp))
      if r:
        lines.append(
            f"  - {m} at w={w_cmp}: units {lk['units']:,} -> {r['units']:,} "
            f"({_ratio(lk['units'], r['units'])}), circuit "
            f"{lk['units_circuit']:,} -> {r['units_circuit']:,}, params_trim "
            f"{lk['params_trim']:,} -> {r['params_trim']:,} "
            f"({_ratio(lk['params_trim'], r['params_trim'])}), layers "
            f"{lk['layers']} -> {r['layers']}, compile {lk['compile_s']:.1f} "
            f"-> {r['compile_s']:.1f} s.")
  w_max = CFG["widths"][-1]
  for m in CFG["methods"]:
    r = by.get((m, w_max))
    if r:
      lines.append(f"- {m} at w={w_max}: {r['units']:,} units (circuit "
                   f"{r['units_circuit']:,}, decode {r['units_decode']:,}, "
                   f"encode {r['units_encode']:,}), {r['layers']} layers, "
                   f"d_trim {r['d_model_trim']:,}, p_trim "
                   f"{r['params_trim']:,}, compile {r['compile_s']:.1f} s.")
  lines += ["", "## Growth fits (least squares over the compiled widths)", "",
            "| method | widths | units log2-slope | circuit log2-slope | "
            "circuit units ~ a*x + b | params_trim log2-slope | compile_s "
            "log2-slope | layers |", "|---|---|---|---|---|---|---|---|"]
  for m, f in fit.items():
    c = f["circuit_lin"]
    lines.append(
        f"| {m} | {f['widths'][0]}..{f['widths'][-1]} | "
        f"{f['units_log2_slope']:.2f} | {f['units_circuit_log2_slope']:.2f} | "
        f"{c['a']:.1f}*{c['x']} + {c['b']:.1f} (max resid "
        f"{c['max_abs_resid']:.1f}) | {f['params_trim_log2_slope']:.2f} | "
        f"{f['compile_s_log2_slope']:.2f} | {f['layers']} |")
  lines += ["", "A log2-slope s means the quantity grows like 2^(s w): 2 is "
            "the 4^w of a two-operand table, 1 the 2^w of a w-bit output.", "",
            f"Plot: {os.path.join(CFG['out_dir'], 'bit_arith.png')}. "
            f"Total wall time {total_s:.0f} s."]
  return lines


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  # Widest first: they take longest, so the pool finishes sooner.
  jobs = [(m, w) for w in reversed(CFG["widths"]) for m in CFG["methods"]]
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

    emit(f"bit_arith: {len(jobs)} bit-circuit configs + lookup chain over "
         f"w={CFG['widths']} on {CFG['workers']} workers, timeout "
         f"{CFG['timeout_s']} s per lookup and {CFG['bit_timeout_s']} s per bit "
         f"circuit config -> {OUT_DIR}")
    emit("units = MLP hidden units from value sets; live = units with nonzero "
         "input weights; circuit/decode = adder/table vs from_bits units; "
         "mism_* = sequences (of seqs) differing from rasp.evaluate / integer "
         "a + b")
    emit(HEADER)
    t0 = time.perf_counter()

    def lookup_job():
      for r in run_lookup_chain():
        record(r)

    with ThreadPoolExecutor(CFG["workers"]) as ex:
      futs = [ex.submit(lookup_job)]
      futs += [ex.submit(lambda j=j: record(run_one(*j))) for j in jobs]
      for fu in as_completed(futs):
        fu.result()
    total_s = time.perf_counter() - t0
    emit(f"done {len(rows):,} configs in {total_s:.0f} s")
    fit = fits(rows)
    plot(rows, os.path.join(OUT_DIR, "bit_arith.png"))
    lines = write_summary(rows, fit, total_s)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
      f.write("\n".join(lines) + "\n")
    emit("")
    for line in lines:
      emit(line)


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--one":
    child(sys.argv[2], int(sys.argv[3]))
  else:
    main()
