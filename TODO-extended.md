# TODO extended

## Backlog

- ci-first-run: Confirm the GitHub Actions workflow passes on its first real run [productivity] #ci #github #verification

  `.github/workflows/test.yml` was only parsed as YAML and run as an equivalent local locked pytest on Python 3.11.
  Python 3.12 and 3.13 in the matrix have not been run. Nothing has been pushed.

## In Progress

## Done

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
  max_seq_len 20 was skipped: compile alone ran past the 1-minute limit, so its error is unmeasured.
  The inferred set is still a loose over-approximation, because the two means are not independent.
