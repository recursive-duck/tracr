"""Inputs per second of rasp.evaluate vs rasp.fast_eval.evaluate_batch.

Settings live in config.toml [fast_eval_bench]. Each timing runs in this one
process on one core: the numbers are per-core speeds. The inputs timed with
rasp.evaluate are also checked against fast_eval. Writes run.log and
results.json to out_dir.

Usage:
  uv run python experiments/fast_eval_bench.py
"""

import json
import os
import random
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "config.toml"), "rb") as _f:
  CFG = tomllib.load(_f)["fast_eval_bench"]
OUT_DIR = os.path.join(ROOT, CFG["out_dir"])

HEADER = (f"{'program':<8}{'N':>5}{'batch':>7}{'evaluate/s↑':>13}"
          f"{'fast/s↑':>12}{'speedup↑':>10}{'eval_n':>8}{'fast_n':>9}"
          f"{'equal':>7}")


def build(prog, n):
  from tracr.compiler import lib
  from tracr.rasp import rasp
  if prog == "sort":
    return lib.make_sort(rasp.tokens, rasp.tokens, max_seq_len=n, min_key=1)
  return lib.make_hist()


def timed(fn):
  """Runs fn until min_time_s and min_reps; returns (reps, seconds)."""
  reps, t0 = 0, time.perf_counter()
  while True:
    fn(reps)
    reps += 1
    elapsed = time.perf_counter() - t0
    if elapsed >= CFG["min_time_s"] and reps >= CFG["min_reps"]:
      return reps, elapsed


def main():
  from tracr.rasp import fast_eval
  from tracr.rasp import rasp
  os.makedirs(OUT_DIR, exist_ok=True)
  rng = random.Random(CFG["seed"])
  vocab = list(range(CFG["vocab_size"]))
  rows = []
  with open(os.path.join(OUT_DIR, "run.log"), "w") as log:

    def emit(line):
      print(line, flush=True)
      log.write(line + "\n")
      log.flush()

    emit(f"timing {CFG['programs']} at N in {CFG['seq_lens']}, vocab "
         f"{CFG['vocab_size']}, at least {CFG['min_time_s']} s and "
         f"{CFG['min_reps']} reps per timing, one core")
    emit(HEADER)
    for prog in CFG["programs"]:
      for n in CFG["seq_lens"]:
        program = build(prog, n)
        batch = max(1, min(CFG["max_batch"], CFG["batch_elements"] // n**2))
        seqs = [[rng.choice(vocab) for _ in range(n)] for _ in range(batch)]
        ref = {}

        def slow(i, program=program, seqs=seqs, ref=ref):
          ref[i % len(seqs)] = rasp.evaluate(program, seqs[i % len(seqs)])

        eval_n, eval_s = timed(slow)
        got = {}

        def fast(_, program=program, seqs=seqs, got=got):
          got["out"] = fast_eval.evaluate_batch(program, seqs)

        fast_reps, fast_s = timed(fast)
        equal = all(got["out"][i] == v for i, v in ref.items())
        row = dict(prog=prog, N=n, batch=batch, eval_inputs=eval_n,
                   eval_s=eval_s, eval_per_s=eval_n / eval_s,
                   fast_inputs=fast_reps * batch, fast_s=fast_s,
                   fast_per_s=fast_reps * batch / fast_s, equal=equal)
        row["speedup"] = row["fast_per_s"] / row["eval_per_s"]
        rows.append(row)
        emit(f"{prog:<8}{n:>5}{batch:>7,}{row['eval_per_s']:>13,.1f}"
             f"{row['fast_per_s']:>12,.1f}{row['speedup']:>9,.0f}x"
             f"{eval_n:>8,}{row['fast_inputs']:>9,}{str(equal):>7}")
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
      json.dump(dict(config=CFG, rows=rows), f, indent=1)
    emit(f"wrote {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
  main()
