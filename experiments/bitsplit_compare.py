"""lib.make_sort / make_sort_freq against their bit-split versions (lib_bits).

For each (program, form, V, N): compile, trim with shrinking.shrink(reuse=
False), and compare the compiled outputs with rasp.evaluate on random inputs.
The bit-split RASP outputs are also compared with the lib program's. Settings
live in config.toml [bitsplit_compare]. Writes results.jsonl (one row per job,
flushed), run.log, bitsplit_compare.png and summary.md to out_dir.

Usage:
  uv run --extra experiments python experiments/bitsplit_compare.py
"""

import json
import os
import random
import resource
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.environ.get("BITSPLIT_COMPARE_CONFIG",
                             os.path.join(ROOT, "config.toml"))
with open(CONFIG_PATH, "rb") as _f:
  CFG = tomllib.load(_f)["bitsplit_compare"]
OUT_DIR = os.path.join(ROOT, CFG["out_dir"])
PLOT_NAME = "bitsplit_compare.png"
LIB_FORM = "lib"
SUM_MODE_OF_FORM = {"bits_num": "numerical", "bits_cat": "categorical"}
# Ratios at least this far from 1 are bold in summary.md.
BOLD_RATIO = 3.0

HEADER = (f"{'program':<10}{'form':<9}{'V':>4}{'N':>4}{'d_trim↓':>8}"
          f"{'params_trim↓':>14}{'x_lib↑':>8}{'layers↓':>8}{'heads↓':>7}"
          f"{'mlp_hidden↓':>12}{'compile_s↓':>11}{'mism↓':>6}{'ref_mism↓':>10}")


def build(prog, form, V, N):
  from tracr.compiler import lib
  from tracr.compiler import lib_bits
  from tracr.rasp import rasp
  ref = {
      "sort": lambda: lib.make_sort(rasp.tokens, rasp.tokens, max_seq_len=N,
                                    min_key=1),
      "sort_freq": lambda: lib.make_sort_freq(max_seq_len=N),
  }[prog]()
  if form == LIB_FORM:
    return ref, ref
  mode = SUM_MODE_OF_FORM[form]
  bits = lib_bits.num_bits_for(V)
  program = {
      "sort": lambda: lib_bits.make_sort_bits(
          rasp.tokens, rasp.tokens, max_seq_len=N, num_bits=bits,
          sum_mode=mode),
      "sort_freq": lambda: lib_bits.make_sort_freq_bits(N, bits,
                                                        sum_mode=mode),
  }[prog]()
  return program, ref


def inputs_for(V, N):
  rng = random.Random(CFG["seed"])
  return [[rng.randrange(V) for _ in range(rng.randint(1, N))]
          for _ in range(CFG["samples"])]


def run_compiled(model, inputs):
  """Decoded outputs (without BOS), one jitted forward per input length."""
  import jax
  import numpy as np
  forward = jax.jit(model.forward)
  enc = model.input_encoder
  by_len = {}
  for i, x in enumerate(inputs):
    by_len.setdefault(len(x), []).append(i)
  outputs = [None] * len(inputs)
  for idx in by_len.values():
    ids = np.array([enc.encode([CFG["compiler_bos"]] + inputs[i]) for i in idx])
    out = np.asarray(forward(model.params, ids).unembedded_output)
    for row, i in enumerate(idx):
      outputs[i] = model.output_encoder.decode(out[row].tolist())[1:]
  return outputs


def differs(want, got):
  return any(w is not None and w != g for w, g in zip(want, got))


def child(prog, form, V, N):
  from tracr.compiler import compiling
  from tracr.compiler import shrinking
  from tracr.rasp import rasp
  program, ref = build(prog, form, V, N)
  t0 = time.perf_counter()
  model = compiling.compile_rasp_to_model(
      program, vocab=set(range(V)), max_seq_len=N,
      compiler_bos=CFG["compiler_bos"])
  compile_s = time.perf_counter() - t0
  shrunk = shrinking.shrink(model, reuse=False)
  stats = shrunk.stats
  layers = shrunk.layers
  breakdown = dict(
      embed_params=int(shrunk.token_embed.size + shrunk.pos_embed.size),
      qk_params=sum(int(l.wq.size + l.wk.size) for l in layers),
      ov_params=sum(int(l.wv.size + l.wo.size) for l in layers),
      mlp_params=sum(int(l.w1.size + l.w2.size) for l in layers),
      qk_dims=sum(sum(l.qk_sizes) for l in layers))
  del shrunk
  seqs = inputs_for(V, N)
  want = [rasp.evaluate(program, x) for x in seqs]
  ref_out = [rasp.evaluate(ref, x) for x in seqs]
  got = run_compiled(model, seqs)
  row = dict(prog=prog, form=form, V=V, N=N, ok=True, compile_s=compile_s,
             layers=model.model_config.num_layers, sequences=len(seqs),
             mismatches=sum(differs(w, g) for w, g in zip(want, got)),
             ref_mismatches=sum(differs(r, w) for r, w in zip(ref_out, want)),
             **vars(stats), **breakdown)
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  row["peak_rss_mb"] = rss / 2**20  # bytes on macOS
  print(json.dumps(row), flush=True)


def run_one(prog, form, V, N):
  cmd = [sys.executable, os.path.abspath(__file__), "--one", prog, form,
         str(V), str(N)]
  try:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       timeout=CFG["timeout_s"])
  except subprocess.TimeoutExpired:
    return dict(prog=prog, form=form, V=V, N=N, ok=False,
                err=f"timeout>{CFG['timeout_s']}s")
  lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
  if lines:
    return json.loads(lines[-1])
  tail = (r.stderr.strip().splitlines() or ["?"])[-1]
  return dict(prog=prog, form=form, V=V, N=N, ok=False, err=tail[:160])


def lib_params(rows, r):
  for o in rows:
    if (o["ok"] and o["form"] == LIB_FORM and o["prog"] == r["prog"] and
        o["V"] == r["V"] and o["N"] == r["N"]):
      return o["params_after"]
  return None


def fmt_row(r, rows):
  head = f"{r['prog']:<10}{r['form']:<9}{r['V']:>4}{r['N']:>4}"
  if not r["ok"]:
    return f"{head}  FAIL {r['err']}"
  base = lib_params(rows, r)
  ratio = f"{base / r['params_after']:>8.2f}" if base else f"{'?':>8}"
  return (f"{head}{r['d_model_after']:>8,}{r['params_after']:>14,}{ratio}"
          f"{r['layers']:>8}{r['heads_after']:>7}{r['mlp_units_after']:>12,}"
          f"{r['compile_s']:>11.1f}{r['mismatches']:>6}"
          f"{r['ref_mismatches']:>10}")


def plot(rows, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  colors = {"lib": "#2a78d6", "bits_num": "#eb6834", "bits_cat": "#1baf7a"}
  configs = [tuple(c) for c in CFG["configs"]]
  forms = CFG["forms"]
  fig, axes = plt.subplots(2, len(CFG["programs"]), figsize=(12, 7),
                           squeeze=False)
  width = 0.8 / len(forms)
  for col, prog in enumerate(CFG["programs"]):
    for row_i, (key, label) in enumerate(
        (("params_after", "trimmed params (log)"),
         ("d_model_after", "trimmed d_model (log)"))):
      ax = axes[row_i][col]
      for k, form in enumerate(forms):
        xs, ys = [], []
        for c_i, (V, N) in enumerate(configs):
          hit = [r for r in rows if r["ok"] and r["prog"] == prog and
                 r["form"] == form and r["V"] == V and r["N"] == N]
          if hit:
            xs.append(c_i + (k - (len(forms) - 1) / 2) * width)
            ys.append(hit[0][key])
        ax.bar(xs, ys, width * 0.9, color=colors.get(form), label=form)
      ax.set_yscale("log")
      ax.set_xticks(range(len(configs)))
      ax.set_xticklabels([f"V={V}\nN={N}" for V, N in configs])
      ax.set_title(f"{prog}: {label}  (lower is better)", fontsize=10)
      ax.grid(axis="y", alpha=0.3)
      for side in ("top", "right"):
        ax.spines[side].set_visible(False)
  axes[0][0].legend(frameon=False)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def ratio(a, b):
  """a / b formatted, bold when at least BOLD_RATIO either way."""
  x = a / b
  s = f"{x:.2f}"
  return f"**{s}**" if x >= BOLD_RATIO or x <= 1 / BOLD_RATIO else s


SUMMARY_HEAD = """# bitsplit_compare: lib sort vs bit-split sort

Generated by experiments/bitsplit_compare.py from results.jsonl.

**What was tested.** `lib.make_sort` / `lib.make_sort_freq` ("lib") against
`lib_bits.make_sort_bits` / `make_sort_freq_bits`. The bit-split programs
count the (key, index) pairs below a position's own as one head per bit
level j (pairs equal above bit j, key bit j < query bit j); each head is an
AND of one-bit Selects, so it needs 2 query/key dims per bit instead of about
one per key value. The per-level counts are summed as numerical
SelectorWidths by a balanced tree of LinearSequenceMaps plus one categorical
Map ("bits_num"), or as categorical SelectorWidths by a tree of categorical
SequenceMaps ("bits_cat").

**Method.** Compile each (program, form, V, N), trim with
`shrinking.shrink(model, reuse=False)`, run random inputs (lengths uniform in
1..N) through the compiled model and compare with `rasp.evaluate`; compare the
bit-split RASP output with the lib one. Settings: config.toml
`[bitsplit_compare]`. Plot: bitsplit_compare.png. x = lib params / form params
(above 1: smaller than lib); bold when at least 3x either way. qk = query/key
weights, mlp = MLP weights, both after trimming.
"""

SUMMARY_TAIL = """
**What it does NOT prove.** Random inputs are a sample (the exhaustive check
is tracr/compiler/lib_bits_test.py at V=4, N=5: 1,364 inputs per program).
Nothing here runs at the 100k-vocab / 1M-context target: the query/key saving
should grow with V (lib needs about V dims, bit split about
(log2 V + log2 N)^2), but each bit level adds a SelectorWidth MLP whose width
grows with N, and the one-hot token input and sorted output stay in the
residual in every form. Compile seconds are wall time with several jobs in
parallel on a shared laptop.
"""


def write_summary(rows, path):
  """Writes summary.md (method, table, findings computed from rows)."""
  ok = {(r["prog"], r["form"], r["V"], r["N"]): r for r in rows if r["ok"]}
  configs = [tuple(c) for c in CFG["configs"]]
  bits_forms = [f for f in CFG["forms"] if f != LIB_FORM]
  cols = [LIB_FORM] + bits_forms
  lines = [SUMMARY_HEAD]
  total = sum(r["sequences"] for r in ok.values())
  mism = sum(r["mismatches"] for r in ok.values())
  ref_mism = sum(r["ref_mismatches"] for r in ok.values())
  lines.append(f"**Correctness.** {len(ok):,} of {len(rows):,} jobs compiled; "
               f"{mism:,} of {total:,} inputs differ between compiled model and "
               f"RASP; {ref_mism:,} differ between bit-split and lib RASP.\n")
  lines.append("| program | V | N | " + " | ".join(
      f"{f} params (x)" for f in cols) + " | layers " + " / ".join(cols) +
               " | qk params lib -> bits_num (x) | mlp params lib -> bits_num |")
  # program, V, N, one column per form, layers, qk, mlp.
  lines.append("|---" * (len(cols) + 6) + "|")
  wins = {f: [] for f in bits_forms}
  for prog in CFG["programs"]:
    for V, N in configs:
      base = ok.get((prog, LIB_FORM, V, N))
      if base is None:
        continue
      cells, layers = [], []
      for f in cols:
        r = ok.get((prog, f, V, N))
        if r is None:
          cells.append("FAIL")
          layers.append("-")
          continue
        cells.append(f"{r['params_after']:,} "
                     f"({ratio(base['params_after'], r['params_after'])})")
        layers.append(str(r["layers"]))
        if f in wins:
          wins[f].append((base["params_after"] / r["params_after"], prog, V,
                          N))
      num = ok.get((prog, "bits_num", V, N))
      qk = mlp = "-"
      if num is not None:
        qk = (f"{base['qk_params']:,} -> {num['qk_params']:,} "
              f"({ratio(base['qk_params'], num['qk_params'])})")
        mlp = f"{base['mlp_params']:,} -> {num['mlp_params']:,}"
      lines.append(f"| {prog} | {V} | {N} | " + " | ".join(cells) +
                   f" | {' / '.join(layers)} | {qk} | {mlp} |")
  lines.append("\n**Findings (computed).**")
  for f, xs in wins.items():
    if not xs:
      continue
    beat = [x for x in xs if x[0] > 1]
    lo, hi = min(xs), max(xs)
    lines.append(f"- {f}: smaller than lib in {len(beat)} of {len(xs)} "
                 f"configs; x from {lo[0]:.2f} ({lo[1]} V={lo[2]} N={lo[3]}) "
                 f"to {hi[0]:.2f} ({hi[1]} V={hi[2]} N={hi[3]}).")
  pairs = [(ok[(p, "bits_cat", V, N)]["params_after"] /
            ok[(p, "bits_num", V, N)]["params_after"], p, V, N)
           for p in CFG["programs"] for V, N in configs
           if (p, "bits_cat", V, N) in ok and (p, "bits_num", V, N) in ok]
  if pairs:
    lo, hi = min(pairs), max(pairs)
    lines.append(f"- bits_cat / bits_num params: {lo[0]:.2f} ({lo[1]} V={lo[2]}"
                 f" N={lo[3]}) to {hi[0]:.2f} ({hi[1]} V={hi[2]} N={hi[3]}); "
                 "each categorical SequenceMap is an (N+1)^2-entry table.")
  for r in rows:
    if not r["ok"]:
      lines.append(f"- FAIL {r['prog']} {r['form']} V={r['V']} N={r['N']}: "
                   f"{r['err']}")
  lines.append(SUMMARY_TAIL)
  with open(path, "w") as f:
    f.write("\n".join(lines))
  return lines


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  # Biggest configs first so they don't finish last alone.
  configs = sorted((tuple(c) for c in CFG["configs"]),
                   key=lambda c: -c[0] * c[1])
  jobs = [(p, f, V, N) for V, N in configs for p in CFG["programs"]
          for f in CFG["forms"]]
  rows = []
  with open(os.path.join(OUT_DIR, "results.jsonl"), "w") as out, \
       open(os.path.join(OUT_DIR, "run.log"), "w") as log:

    def emit(line):
      print(line, flush=True)
      log.write(line + "\n")
      log.flush()

    emit(f"compiling {len(jobs):,} jobs on {CFG['workers']} workers, timeout "
         f"{CFG['timeout_s']} s each -> {OUT_DIR}")
    emit("d_trim / params_trim / heads / mlp_hidden: after shrink(reuse=False);"
         " x_lib = lib params_trim / this params_trim (rows print as they "
         "finish, so x_lib is '?' until the lib row exists); mism = of "
         f"{CFG['samples']} random inputs, compiled != rasp.evaluate; ref_mism ="
         " bit-split RASP != lib RASP")
    emit(HEADER)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(CFG["workers"]) as ex:
      futs = [ex.submit(run_one, *j) for j in jobs]
      for fu in as_completed(futs):
        r = fu.result()
        rows.append(r)
        out.write(json.dumps(r) + "\n")
        out.flush()
        emit(fmt_row(r, rows))
    emit(f"done {len(rows):,} jobs in {time.perf_counter() - t0:.0f} s")
    plot_path = os.path.join(OUT_DIR, PLOT_NAME)
    plot(rows, plot_path)
    emit(f"plot: {plot_path}")
    summary_path = os.path.join(OUT_DIR, "summary.md")
    emit("\n".join(write_summary(rows, summary_path)))
    emit(f"summary: {summary_path}")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--one":
    child(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
  else:
    main()
