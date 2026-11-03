# TODO extended

## Backlog

## In Progress

## Done

- ilp-time-limit: Cut packing time with a 5 s integer-program limit; it never beat DSatur in 64 runs @pack-layout `658936b` [perf] #ilp #runtime #packing

  Pack report `3b8c4ad`: the integer program is all of the pack time, 157-356 s per config in 5 of 16 configs, 0 narrower layouts in 32 runs (0 in 32 before).
  Lower ilp_time_limit_s 60 -> 5 in config.toml and the module default; re-run the report and compare pack_s before -> after.
  After: summed pack time 2,108 -> 294 s, summed config time 3,711 -> 1,574 s, wall 657 -> 501 s; widths identical in all 160 packs; optimality proofs 13 -> 8 of 32.

- gate-bounds: Shrink packing gates from 1.9e8 toward the 100 actually needed, outputs unchanged @ilp-packing `3b8c4ad` [perf] #gating #bounds #packing

  Probe at V=4, N=4, every input, 206 units: gate used max 1.945e+08 (median 2,587); needed (largest positive pre-activation under another task) max 100 (median 2).
  The largest weight in the 10 original models is 100. Worst units are layer-0 units of hist and length.
  LayerNorm would not help: it rescales the gate and the activation it must beat by the same input-dependent factor, and breaks exactness.
  After: probe gate max 201 (median 5), needed at the unit's own layer max 100 (median 2), gate / (needed + 1) at most 2.5. Pack report max gate 1.6e+09 -> 8.45e+03.

  - gate-tight-bound: Charge each unit only for other programs' values alive at its layer, one-hot dims bounded by 1 `3b8c4ad` [perf] #bounds #soundness #intervals

    Before: interval bound pushed through every layer, summing |w2| over all units, final-time bound of every program sharing the slot, own program included.
    Now: per other task j, sign-aware intervals of j's values born before the unit's layer plus positions; categorical dims clamped to [0, 1].

  - gate-rebalance: Scale each unit's w1 by 1/c and w2 by c so its largest weights are balanced `3b8c4ad` [perf] #relu #homogeneity #weights

    ReLU(x / c) c = ReLU(x) and c is a power of 2, so outputs are bitwise equal (tested). Largest weight 201 -> 100 at V=4 N=4, 128 at N=32.


- ilp-packing: Pack many programs into one model with task gating, layout chosen by an integer program @lossless-shrink `fcf107b` [experiment] #superposition #ilp #packing

  Gate every block on a task token, so one program's values are zero while another runs and both can share residual dims.
  Goal: dense params grow about K instead of K^2 for K programs, with outputs identical to each program alone.
  Result: width grows about like the widest program (median 1.10x it), params do not: packed 1.84x separate at K=10 vs 4.87x for block-diagonal (medians).
  Probe at V=8, N=8, 10 programs (scratchpad): block-diagonal D 385, 792,715 params; gated MLPs D 129 (max single program 126), 219,171; 10 separate models 125,233.
  Without gates (MLP reads must see zeros) D 205. Erasing every dead dim cuts D to 108 but adds units: 357,156.

  - pack-model: Task-tagged token ids, gated MLPs, shared residual slots; each task matches its program `fcf107b` [feature] #packing #gating #exactness

    A block of another program is silent: attention value reads see zeros, MLP units get -M per other task's `one` dim, M from a sound bound.
    Positional table shared (identical across programs, never value-read). Per-head logit divisors. No erasing, all programs start at layer 0.
    Report: 0 decided mismatches in 33,440 categorical comparisons; 29 tie flips (sort_freq N=32, same as shrink reuse); numerical max diff 2.38e-07.

  - pack-layout: Slots by DSatur colouring, certified by a clique bound, integer program when they differ `fcf107b` [perf] #ilp #colouring #residual

    Conflict: same program, or a value-read of one after the other's birth. Cliques per (program, time) give the bound and the ILP rows.
    DSatur met the bound in 128/160 packs, at most 3 slots above it. The ILP ran 32 times: optimal 13, beat DSatur 0, time limit 15, skipped 4.

  - pack-report: Width and params vs number of programs: packed vs block-diagonal vs separate `fcf107b` [experiment] #scaling #superposition #report

    16 configs, 0 failed. Summed: packed 62,076,515, block-diagonal 132,620,160 (2.14x), separate trimmed 38,013,232 (packed 1.63x).
    x block 3.04-3.92 at N<=8 with V<=8, 1.64-2.54 at N=32. Max gate up to 1.6e+09, see @gate-bounds.

- lossless-shrink: Shrink compiled models with identical outputs: fewer dims, heads and units, per-layer shapes `04fb553` [perf] #compression #exactness #residual

  Scaling sweep (2026-11-03, scratchpad): 0.05-4.9% of dense params are nonzero, and every layer is padded to the widest one.
  Peak simultaneously-live residual dims are 28-81% of d_model at V=8, N=8.
  `tracr/compiler/shrinking.py`: `shrink(model, reuse)` and `compare(model, shrunk, inputs, batch_size, tie_tol)`. Report in `experiments/shrink_report/`.

  - shrink-trim: Drop never-read dims, padding heads, unused key columns and dead MLP units `04fb553` [perf] #compression #assemble #padding

    Iterate to a fixed point, since removing a head or unit can make more dims unread. Keep the original logit divisor sqrt(key_size).
    Dense params over 160 configs: 3,559,320,080 -> 38,013,232. shuffle_dyck2 V=32 N=32: 893,494,508 -> 132,320.

  - shrink-reuse: Reuse residual dims after their last read, erasing old values with 2 MLP units each `04fb553` [perf] #compression #liveness #register-allocation

    Greedy interval colouring by birth time is optimal for intervals. Units ReLU(x) and ReLU(-x) write -x exactly, for any sign.
    Further 38,013,232 -> 37,601,684 overall; smaller than trim in 81/160 configs. Erase units cost more than the dims saved for length, hist, sort_unique, sort.

  - shrink-verify: Shrunk and original models agree on every input of small domains, sampled on large ones `04fb553` [experiment] #verification #exactness #testing

    Tests: 48 cases, every input up to length 4, causal and non-causal, both modes. Report: 0 decided mismatches in 66,880 categorical comparisons.
    29 sort_freq N=32 reuse sequences flip at the original's own ties (float64 margin <= 2.5e-15); trim and reuse agree to 3.97e-12 in float64 there.

  - shrink-report: Dense params before -> after for 10 lib programs over vocab and length `04fb553` [experiment] #compression #scaling #report

    160 configs, V and N in {4,8,16,32}, 523 s on 8 workers. Median reduction 1.9x (length, hist) to 436x (shuffle_dyck2, reuse).

- repo-analysis: Understand tracr and verify suspected issues by measurement `9ce2b8c` [experiment] #analysis #compiler #baseline

  Baseline: 5,756 passed, 1 xfailed, 12 subtests passed in 41.04 s on 8 workers. README reverse example matches.
  Causal compile matched the causal RASP evaluator on 30/30 inputs for frac_prevs, pair_balance and length.
  Probes are in the session scratchpad, not in the repo. No source file was changed.

- fix-key-size: Compile crashes when value dims exceed query/key dims, e.g. vocab 6 with max_seq_len 3 `8cfa45b` [bugfix] #compiler #assemble #crash

  `key_size` now also covers `w_ov.matrix.shape[1]` for every head.
  Vocab 6, 7, 8: ValueError -> 0 mismatches on 258, 399, 584 sequences. Vocab 20: ValueError -> 0 mismatches on 1,000 random sequences.

- fix-pad-mask: PAD is never masked, so indices-keyed selectors change output when padded `49d45f8` [bugfix] #compiler #transformer #padding

  `assemble_craft_model` takes an optional `pad_token`; `craft_model_to_transformer` passes the encoder's PAD id.
  Over 174 padded/unpadded pairs: indices-keyed width 174 -> 0 mismatches, indices-keyed GEQ mean 122 -> 0.
  Length, reverse and frac_prevs stay at 0. Decoded output at PAD positions is still undefined.

- fix-aggregate-values: Numerical Aggregate value set misses k/n, so a Map on a mean silently miscompiles `37ab6aa` [bugfix] #compiler #soundness #basis-inference

  The value set is now every reduced k/n, plus 0. The ramp uses L = max(caller's L, 4 / min gap), and near-duplicates under 1e-9 are merged.
  `x>0.5`: 25/62 -> 0/62 at length 5, 171/300 -> 0/300 at 10, 206/300 -> 0/300 at 20.
  `x*10`: 35/62 -> 0/62 at length 5, 238/300 -> 0/300 at 10, 270/300 -> 0/300 at 20.
  MLP hidden size: 11 -> 21 at length 5, 21 -> 65 at 10, 41 -> 257 at 20.
  See @fix-int-guard and @float32-composite-means for what this does not cover.

- fix-graph-traversal: Graph extraction and depth DFS lack visited sets, cost grows 4x per diamond `44e54db` [perf] #compiler #complexity #graph

  Each expression is visited once, sources are deduped, and depth is a longest-path pass in topological order.
  Depth at n=12: 20.018 -> 0.000227 s. Full compile at n=12: 20.785 -> 1.474 s. n=128 compiles in 2.723 s.
  Identical output on 8 lib programs: 328 arrays compared, 0 differ.

- fix-copy-id: copy() keeps cached unique_id, so a copy can share a label with different encoding `497a089` [bugfix] #rasp #identity #labels

  `copy()` now drops the cached ID, so a copy always gets a new label.
  Two programs mixing categorical and numerical uses of one expression crashed at compile before; both now match RASP on all 81 inputs.
  A silent wrong compile from the old behaviour was not reproduced.

- fix-validator-coverage: Validator passes unsupported exprs and dedups by name instead of label `5627996` [bugfix] #compiler #validation #errors

  Dedupes by label at enqueue time. Full, LengthType, ConstantSOp and non-Select selectors are now reported, so they fail early with NotImplementedError.
  Checked 49 compilable programs: 0 rejected. UNSUPPORTED_TEST_CASES: 9/9 flagged.

- project-hygiene: Add .gitignore, python_requires, pinned deps and CI `ac64d3d` [productivity] #packaging #ci #repo

  `pyproject.toml` replaces `setup.py`, with `requires-python >=3.11` because chex 0.1.92 needs it. Adds `uv.lock`, `.gitignore` and a test workflow.
  Built wheels hold the same 61 files as before. A fresh install outside the repo imports every subpackage and runs the README example.
  See @ci-first-run for what is still unverified.

- fix-int-guard: Aggregate binary guard accepts values like 0.5, so a Map on their mean miscompiles `d66a157` [bugfix] #compiler #soundness #basis-inference

  The guard in `basis_inference.py` now compares values (`x == 0 or x == 1`) instead of `int(x)`, and the dead non-binary fallback is gone.
  `Map(x>0.3, mean of t*0.5)` at max_seq_len 4: 9/30 wrong -> NotImplementedError at compile.
  New test: 0.5, 1.9, -0.5 failed on the old code and are rejected now; 2 was already rejected. No existing test relied on non-binary means.

- float32-composite-means: Map after a difference of two means loses accuracy from about max_seq_len 14 `1a8d3d4` [bugfix] #compiler #float32 #numerical-mlp

  Root cause: precision only. Old code in float64 at max_seq_len 14 gives `x*10` max error 1.21e-11 vs 4.48e-03 in float32.
  Fix: each threshold uses L_k = max(L, 4 / gap_k) for its own gap, so only close pairs get large activations.
  `x*10`, 60 inputs: max_seq_len 10: 4.76e-04 -> 3.43e-05. max_seq_len 14: 4.48e-03 -> 1.67e-04, wrong (>1e-3) 55 -> 0. `x<0` stays 0/60. Hidden sizes unchanged.
  max_seq_len 20 (after @fix-onehot-alloc made it compile): 6.82e-03 -> 1.77e-04, wrong 55 -> 0 of 60, 12,973 hidden.
  The inferred set is still a loose over-approximation, because the two means are not independent.

- ci-first-run: Confirm the GitHub Actions workflow passes on its first real run `54f9254` [productivity] #ci #github #verification

  Run 41592589029 on push of `54f9254`: Python 3.11 in 2m54s, 3.12 in 2m10s, 3.13 in 1m56s, all green.
  Only warnings: actions/checkout targets the deprecated Node 20, and ubuntu-latest moves to Ubuntu 26.

- fix-onehot-alloc: One-hot vectors built from a full identity matrix make large MLP compiles O(n^3) `fa0ca17` [perf] #compiler #craft #complexity

  `vector_from_basis_direction` took `np.eye(n)[i]`; `combine_in_parallel` calls it once per hidden unit.
  Profile at max_seq_len 14: 11.5 of 14.0 s in `np.zeros`. Now a direct one-hot: profiled compile 14.028 -> 0.235 s.
  Compile wall time: 15.7 -> 1.11 s at max_seq_len 14; 1.19 s at 16; 2.4 s at 20, which did not finish in minutes before.
  The cause was the algorithm, not the core count: compile is single-process, and the test suite already runs on 8 workers.

- ci-node24: CI actions target the deprecated Node 20 runtime, so every job logs a warning `a78d2c5` [bugfix] #ci #github #actions

  Run 41593768525 warned for `actions/checkout@11d5960` (v4) and `astral-sh/setup-uv@d0cc045` (v6), both `using: node20`.
  Pinned to checkout v7.0.1 `3d3c42e` and setup-uv v10.3.3 `68559e9`, both `using: node24`; the inputs we use still exist.
  Run 41594629705: 3.11, 3.12, 3.13 green, Node 20 annotations 3 -> 0. Only the Ubuntu 26 label notice is left.
  setup-uv v9 stops pruning the cache, so cache usage may grow.
