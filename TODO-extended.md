# TODO extended

## Backlog

- fix-aggregate-values: Numerical Aggregate value set misses k/n, so a Map on a mean silently miscompiles [bugfix] #compiler #soundness #basis-inference

  `tracr/compiler/basis_inference.py:87` adds `value / length`, giving {0, 1/n}. True means of binary values are k/n.
  A later numerical Map discretises against that incomplete set and snaps 2/3 to 1/2.
  Measured over all vocab {0,1} inputs of length 1-5: `frac>0.5` 25/62 wrong, `frac*10` 35/62 wrong, bare `frac` 0/62 wrong.
  Enumerating all k/n grows the set: max_seq_len 5 values 6 -> 11, hidden 11 -> 21; 10: 11 -> 33, 21 -> 65; 20: 21 -> 129, 41 -> 257.
  Hazard, computed but not run: at max_seq_len 10 the half-gap between values is 0.0056, smaller than the default ramp width 1/100 = 0.01, so `mlp_exactness` probably has to scale with it.

- fix-key-size: Compile crashes when value dims exceed query/key dims, e.g. vocab 6 with max_seq_len 3 [bugfix] #compiler #assemble #crash

  `tracr/compiler/assemble.py:133` sizes `key_size` from `w_qk` only; `:303` then writes `w_ov` into that width.
  Measured with an indices-EQ Aggregate at max_seq_len 3: vocab 5 compiles, vocab 6 raises a broadcast ValueError.
  The fix should include `w_ov.matrix.shape[1]` in the max.

- fix-pad-mask: PAD is never masked, so indices-keyed selectors change output when padded [bugfix] #compiler #transformer #padding

  `tracr/compiler/assemble.py:254` never passes `pad_token`, so `input_mask` is all ones.
  Tokens-keyed selectors are safe only because PAD has no basis direction in the tokens value set.
  Indices-keyed width on [1,2]: 2 -> 4 with two PADs. Indices-keyed GEQ mean on [1,1]: 1.0 -> 0.5.
  The pad id must be available when the model is built, but it is only known in `craft_model_to_transformer`.

- fix-graph-traversal: Graph extraction and depth DFS lack visited sets, cost grows 4x per diamond [perf] #compiler #complexity #graph

  `tracr/compiler/rasp_to_graph.py:56` enqueues every child with no visited check and `:61` appends a source per path.
  `tracr/compiler/craft_graph_to_model.py:52` walks every path from each of those duplicate roots.
  Depth step on a diamond chain: n=4 <0.001 s, n=8 0.082 s, n=12 20.410 s with 26 nodes and 4,096 source entries.
  Real lib programs are small today: shuffle_dyck2 has 32 nodes and 29 source entries for 2 unique sources.
  A topological-order longest-path pass plus deduped sources makes this linear in edges.

- fix-validator-coverage: Validator passes unsupported exprs and dedups by name instead of label [bugfix] #compiler #validation #errors

  `tracr/compiler/validating.py:179` compares `.name`, so an unsupported Aggregate nested in another Aggregate returns zero issues.
  Six unsupported forms also validate clean: Full, LengthType, Aggregate(And), SelectorWidth(And), ConstantSelector, ConstantSOp.
  All still fail at compile time, but with mixed errors, including a raw AttributeError for SelectorWidth(And).
  `validating.py:40` also reuses the "tokens" message for the indices check.

- fix-copy-id: copy() keeps cached unique_id, so a copy can share a label with different encoding [bugfix] #rasp #identity #labels

  `tracr/rasp/rasp.py:137` is a cached_property and `:143` is a plain `copy.copy`, despite the "new ID" docstring.
  After reading `x.label`, `rasp.numerical(x)` has the same label `map_141` but NUMERICAL encoding.
  The label collision is measured. A wrong compile caused by it has not been demonstrated.

- project-hygiene: Add .gitignore, python_requires, pinned deps and CI [productivity] #packaging #ci #repo

  There is no `.gitignore`, so running tests dirties the tree with `__pycache__`.
  `setup.py` has no `python_requires` and no pins. `validating.py:32` evaluates `X | None` at definition time, which needs Python 3.10+.
  Tests pass on Python 3.11.6, jax 0.10.2, haiku 0.0.17, numpy 2.4.6.

## In Progress

## Done

- repo-analysis: Understand tracr and verify suspected issues by measurement `9ce2b8c` [experiment] #analysis #compiler #baseline

  Baseline: 5,756 passed, 1 xfailed, 12 subtests passed in 41.04 s on 8 workers. README reverse example matches.
  Causal compile matched the causal RASP evaluator on 30/30 inputs for frac_prevs, pair_balance and length.
  Probes are in the session scratchpad, not in the repo. No source file was changed.
