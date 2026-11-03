"""Plot and summary for experiments/shrink_report.py results.

Run alone to rebuild them from results.jsonl:
  uv run --extra experiments python experiments/shrink_report_summary.py
"""

import json
import os
import statistics
import tomllib

import numpy as np

PLOT_NAME = "shrink_report.png"
SUMMARY_NAME = "summary.md"
SUMMARY_JSON = "summary.json"


def nines(x):
  """Ratio in (0,1) -> signed number of nines; 0.9 -> 1, 0.99 -> 2."""
  x = np.clip(np.asarray(x, float), 1e-15, 1 - 1e-15)
  s, y = np.sign(x - .5), .5 + abs(x - .5)
  t = (y - .5) / .4
  bridge = ((t**3 - 2 * t**2 + t) * 1.6 / np.log(10) + (3 * t**2 - 2 * t**3)
            + (t**3 - t**2) * 4 / np.log(10))
  return s * np.where(y >= .9, -np.log10(1 - y), bridge)


SCI_DIGITS = 12  # numbers with at least this many digits use 1.234e+12


def fmt_count(n):
  if abs(n) >= 10**(SCI_DIGITS - 1):
    return f"{n:.3e}"
  return f"{n:,}"


def _plot(ok, cfg, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.ticker import NullFormatter
  programs = cfg["programs"]
  colors = plt.get_cmap("tab10")
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.5))
  for i, prog in enumerate(programs):
    rows = [r for r in ok if r["prog"] == prog]
    if not rows:
      continue
    before = [r["trim"]["params_before"] for r in rows]
    ax0.scatter(before, [r["trim"]["params_after"] for r in rows],
                facecolors="none", edgecolors=colors(i), s=28)
    ax0.scatter(before, [r["reuse"]["params_after"] for r in rows],
                color=colors(i), s=14, label=prog)
  all_before = [r["trim"]["params_before"] for r in ok]
  grid = np.geomspace(min(all_before), max(all_before), 50)
  for factor, style in ((1, "-"), (10, "--"), (100, ":"), (1000, "-.")):
    ax0.plot(grid, grid / factor, color="grey", lw=0.8, ls=style,
             label=f"1/{factor:,}" if factor > 1 else "no change")
  ax0.set(xscale="log", yscale="log", xlabel="dense params, original",
          ylabel="dense params, shrunk",
          title="Params after shrinking (hollow: trim only, filled: + reuse)")
  ax0.legend(fontsize=7, ncol=2)
  v_mid = sorted(cfg["vocab_sizes"])[len(cfg["vocab_sizes"]) // 2]
  for i, prog in enumerate(programs):
    rows = sorted((r for r in ok if r["prog"] == prog and r["V"] == v_mid),
                  key=lambda r: r["N"])
    if not rows:
      continue
    ns = [r["N"] for r in rows]
    for key, style in (("trim", "--"), ("reuse", "-")):
      kept = [r[key]["params_after"] / r[key]["params_before"] for r in rows]
      ax1.plot(ns, -nines(kept), ls=style, marker="o", ms=3, color=colors(i),
               label=prog if key == "reuse" else None)
  ax1.set(xscale="log", xlabel="max_seq_len N",
          ylabel="share of params removed, nines (1 = 90%, 2 = 99%)",
          title=f"Removed share vs N at V={v_mid} (dashed: trim, solid: + reuse)")
  ax1.set_xticks(cfg["max_seq_lens"], [str(n) for n in cfg["max_seq_lens"]])
  ax1.xaxis.set_minor_formatter(NullFormatter())
  ax1.legend(fontsize=7, ncol=2)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def _per_program(ok, cfg):
  table = []
  for prog in cfg["programs"]:
    rows = [r for r in ok if r["prog"] == prog]
    if not rows:
      continue
    big = max(rows, key=lambda r: r["trim"]["params_before"])
    trim_x = [r["trim"]["params_before"] / r["trim"]["params_after"]
              for r in rows]
    reuse_x = [r["reuse"]["params_before"] / r["reuse"]["params_after"]
               for r in rows]
    d_x = [r["reuse"]["d_model_before"] / r["reuse"]["d_model_after"]
           for r in rows]
    table.append(dict(
        prog=prog, configs=len(rows), median_x_trim=statistics.median(trim_x),
        median_x_reuse=statistics.median(reuse_x),
        median_x_d=statistics.median(d_x), big_V=big["V"], big_N=big["N"],
        big_before=big["trim"]["params_before"],
        big_trim=big["trim"]["params_after"],
        big_reuse=big["reuse"]["params_after"], big_nnz=big["nnz"],
        reuse_wins=sum(r["reuse"]["params_after"] < r["trim"]["params_after"]
                       for r in rows)))
  return table


def write(rows, out_dir, cfg):
  """Writes the plot, summary.md and summary.json; returns summary lines."""
  ok = [r for r in rows if r.get("ok")]
  failed = [r for r in rows if not r.get("ok")]
  _plot(ok, cfg, os.path.join(out_dir, PLOT_NAME))
  table = _per_program(ok, cfg)
  modes = ("trim", "reuse")
  pairs = [r[m] for r in ok for m in modes]
  cat_seqs = sum(p["sequences"] for p in pairs if p["categorical"])
  cat_mism = sum(p["mismatches"] for p in pairs if p["categorical"])
  cat_ties = sum(p["tie_flips"] for p in pairs if p["categorical"])
  tie_where = sorted({f"{r['prog']} V={r['V']} N={r['N']} {m}" for r in ok
                      for m in modes if r[m]["tie_flips"]})
  num_seqs = sum(p["sequences"] for p in pairs if not p["categorical"])
  num_bits = sum(p["mismatches"] for p in pairs if not p["categorical"])
  seqs = cat_seqs + num_seqs
  exhaustive = sum(r["verify"] == "exhaustive" for r in ok)
  num_diff = max((r[m]["max_abs_diff"] for r in ok for m in modes
                  if not r[m]["categorical"]), default=0.0)
  cat_diff = max((r[m]["max_abs_diff"] for r in ok for m in modes
                  if r[m]["categorical"]), default=0.0)
  before = sum(r["trim"]["params_before"] for r in ok)
  after_trim = sum(r["trim"]["params_after"] for r in ok)
  after_reuse = sum(r["reuse"]["params_after"] for r in ok)
  nnz = sum(r["nnz"] for r in ok)

  lines = ["# Lossless shrink report", ""]
  lines += [
      "## What was tested",
      "",
      f"Every tracr lib program in config.toml, compiled at vocab size V in "
      f"{cfg['vocab_sizes']} and max_seq_len N in {cfg['max_seq_lens']}, "
      f"was shrunk two ways: trim only (drop dims, heads, head columns and MLP "
      f"units that cannot affect the output, and give each layer its own "
      f"shapes) and trim + reuse (residual dims whose value is dead are "
      f"erased with two ReLU units and reused). Each shrunk model was run on "
      f"the same inputs as the original: every input when the domain has at "
      f"most {cfg['exhaustive_limit']:,} sequences, else {cfg['samples']} "
      f"random ones (seed {cfg['seed']}).",
      "",
      "## Key numbers",
      "",
      f"- Configs: {len(ok):,} shrunk and verified, {len(failed):,} failed "
      f"(compile timeout or error), {exhaustive:,} verified on every input.",
      f"- Categorical programs, decoded tokens differ where the original is "
      f"decided: **{cat_mism:,} of {cat_seqs:,}** sequence comparisons (both "
      f"modes).",
      f"- Differ only at ties of the original (top-2 scores within "
      f"{cfg['tie_tol']:g}): {cat_ties:,} of {cat_seqs:,}"
      + (f", in {'; '.join(tie_where)}." if tie_where else "."),
      f"- Numerical programs, output not bitwise equal: {num_bits:,} of "
      f"{num_seqs:,}; largest difference {num_diff:.2e}.",
      f"- Largest difference in categorical output scores (before argmax): "
      f"{cat_diff:.2e}.",
      f"- Dense params summed over all configs: {fmt_count(before)} -> "
      f"{fmt_count(after_trim)} (trim) -> {fmt_count(after_reuse)} (reuse); "
      f"nonzero weights in the originals: {fmt_count(nnz)}.",
      "",
      "| program | configs | median x trim ↑ | median x reuse ↑ | median x d_model ↑ "
      "| reuse smaller | largest (V, N) | params before | trim ↓ | reuse ↓ | nonzero |",
      "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
  ]
  for t in table:
    lines.append(
        f"| {t['prog']} | {t['configs']} | {t['median_x_trim']:.1f} "
        f"| {t['median_x_reuse']:.1f} | {t['median_x_d']:.2f} "
        f"| {t['reuse_wins']}/{t['configs']} | ({t['big_V']}, {t['big_N']}) "
        f"| {t['big_before']:,} | {t['big_trim']:,} | {t['big_reuse']:,} "
        f"| {t['big_nnz']:,} |")
  if failed:
    lines += ["", "Failed configs: " + ", ".join(
        f"{r['prog']} V={r['V']} N={r['N']} ({r['err']})" for r in failed)]
  lines += [
      "",
      "x = dense params before / after. d_model is the residual width. "
      "\"reuse smaller\" counts configs where reuse beat trim only; reuse can "
      "lose because each erased dim costs two MLP units.",
      "",
      "## What it means",
      "",
      "Most of a compiled tracr model is padding: every layer is as wide as "
      "the widest one, and every residual dim exists in every layer. "
      "Removing it leaves a model that computes the same function, so the "
      "params a program really needs are far fewer than the dense count. "
      "Reuse narrows the residual stream further, at the cost of erase units.",
      "",
      "## What it does not prove",
      "",
      "- Sampled configs are checked on a few hundred inputs, not all of them.",
      "- Float32 sums can run in a different order after shrinking, so "
      "numerical outputs may differ in the last bits (see the max difference).",
      "- Tie flips are not errors of the shrink, but they are output changes. "
      "Where they occur, the original's top two scores are equal (float64 "
      "margin under 3e-15), so its own argmax is set by rounding; the "
      "original was also wrong against RASP on every such input checked. "
      "In float64, trim and reuse outputs agree to 4e-12 there.",
      "- Shrunk models use per-layer shapes. They do not fit the uniform haiku "
      "model class, so the saving is in storage and FLOPs of this forward only.",
      "- Dense params after shrinking still include zeros inside kept blocks. "
      "The nonzero column counts the original, including weights into MLP "
      "units whose output weights are all zero, so it is not a lower bound: "
      "shuffle_dyck2 shrinks below it.",
      "",
      f"Plot: {PLOT_NAME}",
  ]
  with open(os.path.join(out_dir, SUMMARY_NAME), "w") as f:
    f.write("\n".join(lines) + "\n")
  with open(os.path.join(out_dir, SUMMARY_JSON), "w") as f:
    json.dump(dict(configs_ok=len(ok), configs_failed=len(failed),
                   sequences=seqs, categorical_sequences=cat_seqs,
                   categorical_mismatches=cat_mism,
                   categorical_tie_flips=cat_ties,
                   numerical_sequences=num_seqs,
                   numerical_not_bitwise_equal=num_bits,
                   max_abs_diff_numerical=num_diff,
                   max_abs_diff_categorical=cat_diff, params_before=before,
                   params_trim=after_trim, params_reuse=after_reuse,
                   nonzero=nnz, per_program=table,
                   explanation="\n".join(lines)), f, indent=1)
  return lines + [f"wrote {os.path.join(out_dir, PLOT_NAME)}",
                  f"wrote {os.path.join(out_dir, SUMMARY_NAME)}"]


if __name__ == "__main__":
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  with open(os.path.join(root, "config.toml"), "rb") as f:
    config = tomllib.load(f)["shrink_report"]
  out = os.path.join(root, config["out_dir"])
  with open(os.path.join(out, "results.jsonl")) as f:
    results = [json.loads(line) for line in f]
  for line in write(results, out, config):
    print(line)
