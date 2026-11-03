"""Plot and summary for experiments/pack_report.py results.

Run alone to rebuild them from results.jsonl:
  uv run --extra experiments python experiments/pack_report_summary.py
"""

import json
import os
import statistics
import tomllib

from shrink_report import CFG as SHRINK_CFG
from shrink_report_summary import fmt_count

PLOT_NAME = "pack_report.png"
SUMMARY_NAME = "summary.md"
SUMMARY_JSON = "summary.json"
# Ratios at least this large are written in bold.
BOLD_RATIO = 3.0
# Y ticks of the cost panel (log scale, ratios over separate models).
Y_TICKS = (1, 1.5, 2, 3, 4, 6)

def _bold(x, text):
  return f"**{text}**" if x >= BOLD_RATIO else text


def _plot(ok, cfg, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  from matplotlib.ticker import MaxNLocator, ScalarFormatter
  colors = plt.get_cmap("viridis")
  ns = sorted(cfg["max_seq_lens"])
  vs = sorted(cfg["vocab_sizes"])
  v_mid = vs[len(vs) // 2]
  fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5.5))
  for j, n in enumerate(ns):
    rows = [r for r in ok if r["V"] == v_mid and r["N"] == n]
    if not rows:
      continue
    sw = rows[0]["sweep"]
    ks = [s["K"] for s in sw]
    c = colors(j / max(1, len(ns) - 1))
    ax0.plot(ks, [s["params"] for s in sw], "-o", ms=3, color=c, label=f"N={n}")
    ax0.plot(ks, [s["params_block_diagonal"] for s in sw], "--", color=c)
    ax0.plot(ks, [s["separate_trim"] for s in sw], ":", color=c)
    ax1.plot(ks, [s["d_model"] for s in sw], "-o", ms=3, color=c,
             label=f"N={n}")
    ax1.plot(ks, [sum(s["d_programs"]) for s in sw], "--", color=c)
    ax1.plot(ks, [max(s["d_programs"]) for s in sw], ":", color=c)
  style = [Line2D([], [], color="grey", ls=ls, label=lab) for ls, lab in
           (("-", "packed"), ("--", "block-diagonal"),
            (":", "K separate models"))]
  ax0.set(yscale="log", xlabel="programs packed K", ylabel="dense params",
          title=f"Params vs K at V={v_mid}")
  ax0.legend(handles=ax0.get_legend_handles_labels()[0] + style, fontsize=7)
  style = [Line2D([], [], color="grey", ls=ls, label=lab) for ls, lab in
           (("-", "packed"), ("--", "sum of widths"), (":", "widest program"))]
  ax1.set(yscale="log", xlabel="programs packed K", ylabel="residual width",
          title=f"Residual width vs K at V={v_mid}")
  ax1.legend(handles=ax1.get_legend_handles_labels()[0] + style, fontsize=7)
  for i, v in enumerate(vs):
    rows = sorted((r for r in ok if r["V"] == v), key=lambda r: r["N"])
    if not rows:
      continue
    c = plt.get_cmap("tab10")(i)
    last = [r["sweep"][-1] for r in rows]
    x = [r["N"] for r in rows]
    ax2.plot(x, [s["params"] / s["separate_trim"] for s in last], "-o",
             color=c, label=f"V={v} packed")
    ax2.plot(x, [s["params_block_diagonal"] / s["separate_trim"]
                 for s in last], "--^", mfc="none", color=c,
             label=f"V={v} block-diagonal")
  ax2.axhline(1, color="grey", lw=0.8)
  ax2.set(xscale="log", yscale="log", xlabel="max_seq_len N",
          ylabel="dense params / sum of separate trimmed models",
          title=f"All {len(cfg['programs'])} programs: cost of one model")
  ax2.set_xticks(ns, [str(n) for n in ns])
  ax2.minorticks_off()
  ax2.set_yticks(Y_TICKS)
  ax2.yaxis.set_major_formatter(ScalarFormatter())
  for ax in (ax0, ax1):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax2.legend(fontsize=7, ncol=2)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def write(rows, out_dir, cfg):
  """Writes the plot, summary.md and summary.json; returns summary lines."""
  ok = sorted((r for r in rows if r.get("ok")), key=lambda r: (r["V"], r["N"]))
  failed = [r for r in rows if not r.get("ok")]
  _plot(ok, cfg, os.path.join(out_dir, PLOT_NAME))
  checks = [v for r in ok for v in r["verify"]]
  cat = [v for v in checks if v["categorical"]]
  num = [v for v in checks if not v["categorical"]]
  cat_seqs = sum(v["sequences"] for v in cat)
  cat_mism = sum(v["mismatches"] for v in cat)
  cat_ties = sum(v["tie_flips"] for v in cat)
  tie_where = sorted({f"{v['prog']} V={r['V']} N={r['N']}" for r in ok
                      for v in r["verify"] if v["tie_flips"]})
  num_seqs = sum(v["sequences"] for v in num)
  num_bits = sum(v["mismatches"] for v in num)
  num_diff = max((v["max_abs_diff"] for v in num), default=0.0)
  cat_diff = max((v["max_abs_diff"] for v in cat), default=0.0)
  diff_where = sorted({f"{v['prog']} V={r['V']} N={r['N']}" for r in ok
                       for v in r["verify"] if v["categorical"]
                       and v["max_abs_diff"] > cfg["tie_tol"]})
  rasp_seqs = sum(v["sequences"] for v in checks)
  rasp_orig = sum(v["rasp_mismatches_original"] for v in checks)
  rasp_pack = sum(v["rasp_mismatches"] for v in checks)
  rasp_where = sorted({f"{v['prog']} V={r['V']} N={r['N']}" for r in ok
                       for v in r["verify"] if v["rasp_mismatches"]
                       or v["rasp_mismatches_original"]})
  rasp_err = max((v["rasp_max_abs_err"] for v in num), default=0.0)
  exhaustive = sum(all(v["verify"] == "exhaustive" for v in r["verify"])
                   for r in ok)
  last = [r["sweep"][-1] for r in ok]
  tot = {k: sum(s[k] for s in last) for k in
         ("params", "params_block_diagonal", "separate_trim",
          "separate_reuse")}
  width_vs_max = [s["d_model"] / max(s["d_programs"]) for s in last]
  width_vs_sum = [sum(s["d_programs"]) / s["d_model"] for s in last]
  # Overhead of one model over K separate ones, per K, median over configs.
  k_max = len(cfg["programs"])
  overhead = []
  for k in range(1, k_max + 1):
    at_k = [r["sweep"][k - 1] for r in ok if len(r["sweep"]) >= k]
    overhead.append(dict(
        K=k, packed=statistics.median(s["params"] / s["separate_trim"]
                                      for s in at_k),
        block=statistics.median(s["params_block_diagonal"] / s["separate_trim"]
                                for s in at_k)))
  packs = [s for r in ok for s in r["sweep"]]
  meets = sum(s["slots_dsatur"] == s["slots_lower_bound"] for s in packs)
  ilp_runs = [s for s in packs if not s["ilp_status"].startswith("not run")]
  ilp_opt = sum("Optimal" in s["ilp_status"] for s in ilp_runs)
  ilp_better = sum(s["solver"] == "ilp" for s in packs)
  ilp_skipped = sum(s["ilp_status"].startswith("skipped") for s in ilp_runs)
  ilp_limit = sum("Time limit" in s["ilp_status"] for s in ilp_runs)
  gaps = [s["slots_final"] - s["slots_lower_bound"] for s in packs]
  x_block = tot["params_block_diagonal"] / tot["params"]
  x_sep = tot["params"] / tot["separate_trim"]

  lines = ["# Packing report", ""]
  lines += [
      "## What was tested",
      "",
      f"The {k_max} tracr lib programs in config.toml were compiled at vocab "
      f"size V in {cfg['vocab_sizes']} and max_seq_len N in "
      f"{cfg['max_seq_lens']}. For each (V, N), the first K programs "
      f"(K = 1..{k_max}, config order) were packed into one model: the token "
      "id selects the program, MLP units of the other programs are switched "
      "off by a large negative weight on the active program's `one` dim, and "
      "programs share residual dims when no attention value read can see "
      "another program's value. The layout is a DSatur colouring of the "
      "conflict graph, checked against a clique lower bound; when they "
      "differ, an integer program (HiGHS) searches for fewer dims. Every task "
      f"of the full {k_max}-program pack was run on the same inputs as its "
      "program alone: every input on small domains, sampled ones on large "
      "domains (settings of [shrink_report]).",
      "",
      "Baselines: block-diagonal stacking (the same blocks, each program in "
      "its own dims) and K separate trimmed models (`shrink`, no reuse), "
      "whose params add up. Params are dense: every weight of every kept "
      "block, zeros included.",
      "",
      "## Key numbers",
      "",
      f"- Configs: {len(ok):,} packed and verified, {len(failed):,} failed, "
      f"{exhaustive:,} verified on every input.",
      f"- Categorical tasks, decoded tokens differ where the original is "
      f"decided: **{cat_mism:,} of {cat_seqs:,}** sequences.",
      f"- Differ only at ties of the original (top-2 scores within "
      f"{cfg['tie_tol']:g}): {cat_ties:,} of {cat_seqs:,}"
      + (f", in {'; '.join(tie_where)}." if tie_where else "."),
      f"- Numerical tasks, output not bitwise equal: {num_bits:,} of "
      f"{num_seqs:,}; largest difference {num_diff:.2e}. Largest difference "
      f"in categorical scores: {cat_diff:.2e}"
      + (f", above {cfg['tie_tol']:g} only in {'; '.join(diff_where)}."
         if diff_where else "."),
      f"- Against the RASP program itself (rasp.fast_eval; positions where "
      f"RASP gives None skipped; numerical outputs within "
      f"{SHRINK_CFG['rasp_atol']:g} + {SHRINK_CFG['rasp_rtol']:g} * |RASP|), "
      f"sequences whose decoded output differs: originals "
      f"**{rasp_orig:,}**, packed tasks **{rasp_pack:,}**, each of "
      f"{rasp_seqs:,}"
      + (f", in {'; '.join(rasp_where)}." if rasp_where else ".")
      + f" Largest numerical error vs RASP: {rasp_err:.2e}.",
      f"- All {k_max} programs, summed over configs: packed "
      f"{fmt_count(tot['params'])}, block-diagonal "
      f"{fmt_count(tot['params_block_diagonal'])} "
      f"({_bold(x_block, f'{x_block:.2f}x')} the packed model), separate "
      f"trimmed models {fmt_count(tot['separate_trim'])} (packed is "
      f"{x_sep:.2f}x that), separate reused models "
      f"{fmt_count(tot['separate_reuse'])}.",
      f"- Residual width of the full pack / widest program alone: median "
      f"{statistics.median(width_vs_max):.2f}, max {max(width_vs_max):.2f}. "
      f"Sum of widths / packed width: median "
      f"{statistics.median(width_vs_sum):.2f}.",
      "- Cost of one model over K separate trimmed models (dense params "
      "ratio, median over configs), K: packed / block-diagonal: "
      + ", ".join(f"{o['K']}: {o['packed']:.2f} / {o['block']:.2f}"
                  for o in overhead) + ".",
      f"- Layout: DSatur meets the clique bound in {meets:,} of {len(packs):,} "
      f"packs. The integer program ran {len(ilp_runs):,} times: proved "
      f"optimal {ilp_opt:,}, beat DSatur {ilp_better:,}, hit the "
      f"{cfg['ilp_time_limit_s']:g} s limit {ilp_limit:,}, skipped as too "
      f"large {ilp_skipped:,}. Shared slots above the bound: max "
      f"{max(gaps)}, median {statistics.median(gaps):g}.",
      "",
      f"Full pack ({k_max} programs) per config:",
      "",
      "| V | N | width ↓ | widest | sum | bound | solver / status | packed ↓ "
      "| block-diagonal ↓ | x block ↑ | separate trim | separate reuse "
      "| max gate ↓ | max weight ↓ | vs RASP ↓ | pack s | peak RSS MB |",
      "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:"
      "|---:|---:|---:|",
  ]
  for r in ok:
    s = r["sweep"][-1]
    bound = s["slots_lower_bound"] + s["gate_slots"] + s["position_slots"]
    xb = s["params_block_diagonal"] / s["params"]
    status = s["ilp_status"].split("(")[-1].rstrip(")") if s["ilp_status"] \
        else ""
    lines.append(
        f"| {r['V']} | {r['N']} | {s['d_model']:,} | {max(s['d_programs']):,} "
        f"| {sum(s['d_programs']):,} | {bound:,} | {s['solver']}: {status} "
        f"| {fmt_count(s['params'])} | {fmt_count(s['params_block_diagonal'])} "
        f"| {_bold(xb, f'{xb:.2f}')} | {fmt_count(s['separate_trim'])} "
        f"| {fmt_count(s['separate_reuse'])} | {s['max_gate']:.2e} "
        f"| {s['max_weight']:.2e} "
        f"| {sum(v['rasp_mismatches_original'] + v['rasp_mismatches'] for v in r['verify']):,} "
        f"| {s['pack_s']:.1f} | {r['peak_rss_mb']:,.0f} |")
  if failed:
    lines += ["", "Failed configs: " + ", ".join(
        f"V={r['V']} N={r['N']} ({r['err']})" for r in failed)]
  lines += [
      "",
      "width = residual dims of the packed model; widest / sum = trimmed "
      "widths of the programs alone; bound = clique lower bound on width "
      "for this conflict rule; x block = block-diagonal / packed params; "
      "vs RASP = sequences whose decoded output differs from RASP, summed "
      "over the originals and the packed tasks.",
      "",
      "## What it means",
      "",
      "Only one program runs per input, so their values never meet and can "
      "share residual dims. The packed stream is "
      f"{statistics.median(width_vs_max):.2f}x as wide as the widest program "
      "(median), while the programs' own widths add up to "
      f"{statistics.median(width_vs_sum):.2f}x the packed width. A program "
      "narrower than the pack so far mostly fits into dims that are already "
      "there.",
      "",
      "Every weight matrix spans the residual width. Block-diagonal stacking "
      "puts all programs' heads and units on the sum of their widths; "
      "packing puts them on about the widest one. Against K separate trimmed "
      "models, the median cost of one model rises from "
      f"{overhead[0]['packed']:.2f} at K=1 to {overhead[-1]['packed']:.2f} "
      f"at K={k_max} for packing, and to {overhead[-1]['block']:.2f} for "
      "block-diagonal. So packing grows faster than K: each small program "
      "pays for the widest program's width. That is why the packed model "
      f"still costs {x_sep:.2f}x the separate models summed. The gain is "
      f"over one stacked model: {x_block:.2f}x fewer params summed over "
      "configs. The steps in the left panels are where the widest programs "
      "join; x block is smallest where the widest program is most of the "
      "summed width.",
      "",
      "Categorical score differences above the tie tolerance are float32 "
      "rounding at attention ties of the original. A different matrix width "
      "changes the summation order, and near-tied keys then share attention "
      "weight differently. Decoded outputs that the original decides do not "
      "change.",
      "",
      "## What it does not prove",
      "",
      f"- Only the full {k_max}-program pack is checked per config; the "
      "smaller packs of the K sweep are built by the same code and only "
      "measured. Unit tests check 2-10 program packs on every input up to "
      "length 4.",
      "- Sampled configs are checked on a few hundred inputs, not all of them.",
      "- The clique bound and the integer program are optimal for this "
      "conflict rule only: no erasing, all programs starting at layer 0, "
      "one dim per value. Other rules can give narrower models.",
      "- The K sweep uses one program order; other orders give other curves.",
      "- Gates come from sound interval bounds, not from the inputs seen. "
      f"The largest gate is {max(r['sweep'][-1]['max_gate'] for r in ok):.3g}"
      " and the largest weight after rebalancing "
      f"{max(r['sweep'][-1]['max_weight'] for r in ok):.3g}. They hold for "
      "any input, but they are not shown to be the smallest gates that do, "
      "and the model is not shown to be a good starting point for training.",
      "- Dense params include zeros inside kept blocks.",
      "",
      f"Plot: {PLOT_NAME}",
  ]
  with open(os.path.join(out_dir, SUMMARY_NAME), "w") as f:
    f.write("\n".join(lines) + "\n")
  with open(os.path.join(out_dir, SUMMARY_JSON), "w") as f:
    json.dump(dict(configs_ok=len(ok), configs_failed=len(failed),
                   categorical_sequences=cat_seqs,
                   categorical_mismatches=cat_mism,
                   categorical_tie_flips=cat_ties,
                   numerical_sequences=num_seqs,
                   numerical_not_bitwise_equal=num_bits,
                   max_abs_diff_numerical=num_diff,
                   max_abs_diff_categorical=cat_diff,
                   rasp_sequences=rasp_seqs,
                   rasp_mismatches_original=rasp_orig,
                   rasp_mismatches_packed=rasp_pack,
                   rasp_max_abs_err_numerical=rasp_err,
                   rasp_atol=SHRINK_CFG["rasp_atol"],
                   rasp_rtol=SHRINK_CFG["rasp_rtol"], totals=tot,
                   width_vs_widest_median=statistics.median(width_vs_max),
                   overhead_vs_separate=overhead,
                   dsatur_meets_bound=meets, packs=len(packs),
                   ilp_runs=len(ilp_runs), ilp_optimal=ilp_opt,
                   ilp_better=ilp_better, explanation="\n".join(lines)),
              f, indent=1)
  return lines + [f"wrote {os.path.join(out_dir, PLOT_NAME)}",
                  f"wrote {os.path.join(out_dir, SUMMARY_NAME)}"]


if __name__ == "__main__":
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  path = os.environ.get("PACK_REPORT_CONFIG", os.path.join(root, "config.toml"))
  with open(path, "rb") as f:
    config = tomllib.load(f)["pack_report"]
  out = os.path.join(root, config["out_dir"])
  with open(os.path.join(out, "results.jsonl")) as f:
    results = [json.loads(line) for line in f]
  for line in write(results, out, config):
    print(line)
