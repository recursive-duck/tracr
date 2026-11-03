"""One-hot vs binary (bit-coded) tokens and indices at large V = N.

For each program, size V = N and encoding: compile in a child process (compile
must finish within compile_timeout_s), shrink with trim only, then run the
model on `samples` random inputs and count mismatches against rasp.evaluate.
The RASP outputs are computed once per (program, V) in a process pool and
shared by both encodings. Settings live in config.toml [bit_encoding].
Writes results.jsonl (one row per config, flushed), run.log,
bit_encoding.png and summary.md to out_dir.

Usage:
  uv run --extra experiments python experiments/bit_encoding.py
"""

import json
import math
import os
import random
import resource
import signal
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.environ.get("BIT_ENCODING_CONFIG",
                             os.path.join(ROOT, "config.toml"))
with open(CONFIG_PATH, "rb") as _f:
  CFG = tomllib.load(_f)["bit_encoding"]
OUT_DIR = os.path.join(ROOT, CFG["out_dir"])

HEADER = (f"{'program':<11}{'enc':<12}{'V=N':>6}{'d_model↓':>10}"
          f"{'d_trim↓':>9}{'params↓':>15}{'params_trim↓':>15}{'layers↓':>8}"
          f"{'compile_s↓':>11}{'shrink_s':>9}{'eval_s':>8}{'inputs':>7}"
          f"{'mism↓':>6}{'peak_GB':>8}")


def build(prog, enc_name, V):
  """The program with tokens, indices and every S-Op in one encoding."""
  from tracr.rasp import rasp
  enc = {"categorical": rasp.categorical, "binary": rasp.binary}[enc_name]
  tokens, indices = enc(rasp.tokens), enc(rasp.indices)
  eq = rasp.Comparison.EQ
  if prog == "agg_eq":
    return enc(rasp.Aggregate(rasp.Select(tokens, tokens, eq), indices))
  if prog == "map_lookup":
    rng = random.Random(CFG["seed"])
    table = [rng.randrange(V) for _ in range(V)]
    return enc(rasp.Map(lambda x: table[x], tokens))
  if prog == "hist":
    return enc(rasp.SelectorWidth(rasp.Select(tokens, tokens, eq)))
  raise ValueError(prog)


def make_inputs(prog, V):
  """Seeded random inputs, lengths uniform in 1..N (N = V).

  agg_eq needs distinct tokens (a categorical Aggregate copies one value), so
  its inputs sample tokens without replacement.
  """
  rng = random.Random(f"{CFG['seed']}-{prog}-{V}")
  out = []
  for _ in range(CFG["samples"]):
    length = rng.randint(1, V)
    if prog == "agg_eq":
      out.append(rng.sample(range(V), length))
    else:
      out.append([rng.randrange(V) for _ in range(length)])
  return out


def rasp_eval(args):
  prog, V, xs = args
  # Encoding does not change what RASP computes.
  return build(prog, "categorical", V)(xs)


def child(prog, enc_name, V):
  """Compiles, shrinks and runs one config; prints one JSON line."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  from tracr.compiler import compiling
  from tracr.compiler import shrinking
  jax.config.update("jax_default_matmul_precision", "float32")

  program = build(prog, enc_name, V)

  def on_alarm(signum, frame):
    raise TimeoutError(f"compile exceeded {CFG['compile_timeout_s']} s")

  signal.signal(signal.SIGALRM, on_alarm)
  signal.alarm(CFG["compile_timeout_s"])
  t0 = time.time()
  model = compiling.compile_rasp_to_model(program, set(range(V)), V,
                                          compiler_bos=CFG["compiler_bos"])
  compile_s = time.time() - t0
  signal.alarm(0)

  t0 = time.time()
  shrunk = shrinking.shrink(model, reuse=False)
  shrink_s = time.time() - t0
  stats = shrunk.stats

  enc = model.input_encoder
  pad = enc.pad_encoding
  forward = jax.jit(model.forward)
  max_len = V + 1
  outputs = []
  t0 = time.time()
  for xs in make_inputs(prog, V):
    ids = enc.encode([CFG["compiler_bos"]] + xs)
    # Pad to a power-of-two bucket so jit compiles few shapes; PAD is masked.
    bucket = min(max_len, 1 << (len(ids) - 1).bit_length())
    ids = ids + [pad] * (bucket - len(ids))
    out = forward(model.params, jnp.array([ids])).unembedded_output[0]
    decoded = model.output_encoder.decode(
        np.asarray(out[1:1 + len(xs)]).tolist())
    outputs.append(decoded)
  eval_s = time.time() - t0

  peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
  cfg = model.model_config
  print(json.dumps(dict(
      program=prog, encoding=enc_name, V=V, N=V,
      d_model=len(model.residual_labels), d_model_trim=stats.d_model_after,
      params=stats.params_before, params_trim=stats.params_after,
      layers=cfg.num_layers, compile_s=compile_s, shrink_s=shrink_s,
      eval_s=eval_s, peak_gb=peak_gb, outputs=outputs)))


def fmt_row(r):
  if r.get("error"):
    return (f"{r['program']:<11}{r['encoding']:<12}{r['V']:>6,}  "
            f"FAILED: {r['error']}")
  return (f"{r['program']:<11}{r['encoding']:<12}{r['V']:>6,}"
          f"{r['d_model']:>10,}{r['d_model_trim']:>9,}{r['params']:>15,}"
          f"{r['params_trim']:>15,}{r['layers']:>8}{r['compile_s']:>11.2f}"
          f"{r['shrink_s']:>9.2f}{r['eval_s']:>8.1f}{r['inputs']:>7}"
          f"{r['mismatches']:>6}{r['peak_gb']:>8.2f}")


def mismatches(outputs, expected):
  bad = 0
  for out, exp in zip(outputs, expected):
    if any(e is not None and o != e for o, e in zip(out, exp)):
      bad += 1
  return bad


def fit_exponent(rows, key):
  """Slope of log(key) against log(V), per (program, encoding)."""
  import numpy as np
  fits = {}
  for r in rows:
    if not r.get("error"):
      fits.setdefault((r["program"], r["encoding"]), []).append(
          (r["V"], r[key]))
  out = {}
  for k, pts in fits.items():
    if len(pts) >= 2:
      v, y = np.array(pts, dtype=float).T
      out[k] = float(np.polyfit(np.log(v), np.log(y), 1)[0])
  return out


def plot(rows, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
  keys = [("d_model_trim", "trimmed d_model ↓"),
          ("params_trim", "trimmed params ↓"), ("compile_s", "compile s ↓")]
  # Colour by program, dashed one-hot vs solid binary.
  styles = {"categorical": "--", "binary": "-"}
  colors = {p: f"C{i}" for i, p in enumerate(CFG["programs"])}
  for ax, (key, title) in zip(axes, keys):
    for prog in CFG["programs"]:
      for enc in CFG["encodings"]:
        pts = sorted((r["V"], r[key]) for r in rows if not r.get("error") and
                     r["program"] == prog and r["encoding"] == enc)
        if pts:
          ax.plot(*zip(*pts), styles[enc], color=colors[prog], marker="o",
                  label=f"{prog} {enc}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("V = N")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
  axes[0].legend(fontsize=7)
  fig.tight_layout()
  fig.savefig(path, dpi=120)


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  log = open(os.path.join(OUT_DIR, "run.log"), "w")

  def emit(line=""):
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()

  emit(f"bit_encoding: programs={CFG['programs']} encodings="
       f"{CFG['encodings']} sizes={CFG['sizes']} samples={CFG['samples']}")
  results = open(os.path.join(OUT_DIR, "results.jsonl"), "w")
  rows = []
  pool = ProcessPoolExecutor(CFG["rasp_workers"])
  emit(HEADER)
  for V in CFG["sizes"]:
    # RASP outputs for this size, computed while the children compile.
    pending = {prog: pool.map(rasp_eval,
                              [(prog, V, xs) for xs in make_inputs(prog, V)])
               for prog in CFG["programs"]}
    expected = {}
    for prog in CFG["programs"]:
      for enc in CFG["encodings"]:
        row = dict(program=prog, encoding=enc, V=V, N=V)
        try:
          proc = subprocess.run(
              [sys.executable, __file__, "--child", prog, enc, str(V)],
              capture_output=True, text=True, timeout=CFG["child_timeout_s"],
              check=False)
          last = proc.stdout.strip().splitlines()[-1:] or [""]
          if proc.returncode != 0 or not last[0].startswith("{"):
            err = (proc.stderr.strip().splitlines() or ["no output"])[-1]
            raise RuntimeError(err[:200])
          row.update(json.loads(last[0]))
        except (subprocess.TimeoutExpired, RuntimeError) as e:
          row["error"] = str(e)[:200]
        if not row.get("error"):
          if prog not in expected:
            expected[prog] = list(pending[prog])
          outputs = row.pop("outputs")
          row["inputs"] = len(outputs)
          row["mismatches"] = mismatches(outputs, expected[prog])
        rows.append(row)
        results.write(json.dumps(row) + "\n")
        results.flush()
        emit(fmt_row(row))
  pool.shutdown()

  emit()
  ratio_lines = []
  for V in CFG["sizes"]:
    for prog in CFG["programs"]:
      by_enc = {r["encoding"]: r for r in rows if r["V"] == V and
                r["program"] == prog and not r.get("error")}
      if len(by_enc) == 2:
        c, b = by_enc["categorical"], by_enc["binary"]
        ratio_lines.append(
            f"{prog:<11}{V:>6,}  d_trim {c['d_model_trim']:,} -> "
            f"{b['d_model_trim']:,} ({c['d_model_trim'] / b['d_model_trim']:.1f}x)"
            f"  params_trim {c['params_trim']:,} -> {b['params_trim']:,} "
            f"({c['params_trim'] / b['params_trim']:.1f}x)  compile "
            f"{c['compile_s']:.1f} -> {b['compile_s']:.1f} s")
  emit("categorical -> binary (trimmed):")
  for line in ratio_lines:
    emit("  " + line)
  fits = {key: fit_exponent(rows, key)
          for key in ("compile_s", "d_model_trim", "params_trim")}
  emit("log-log slope vs V (exponent):")
  emit(f"  {'program':<11}{'enc':<12}{'compile_s':>10}{'d_trim':>8}"
       f"{'params_trim':>12}")
  for k in fits["compile_s"]:
    emit(f"  {k[0]:<11}{k[1]:<12}{fits['compile_s'][k]:>10.2f}"
         f"{fits['d_model_trim'].get(k, math.nan):>8.2f}"
         f"{fits['params_trim'].get(k, math.nan):>12.2f}")
  png = os.path.join(OUT_DIR, "bit_encoding.png")
  plot(rows, png)
  emit(f"plot: {png}")
  with open(os.path.join(OUT_DIR, "fits.json"), "w") as f:
    json.dump({key: {f"{p}/{e}": v for (p, e), v in fit.items()}
               for key, fit in fits.items()}, f, indent=1)


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--child":
    child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
  else:
    main()
