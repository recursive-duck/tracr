# TODO

## Backlog


## In Progress

- ilp-time-limit: Cut packing time with a 5 s integer-program limit; it never beat DSatur in 56 runs @pack-layout [perf] #ilp #runtime #packing

## Done

- gate-bounds: Shrink packing gates from 1.9e8 toward the 100 actually needed, outputs unchanged @ilp-packing `3b8c4ad` [perf] #gating #bounds #packing
  - gate-tight-bound: Charge each unit only for other programs' values alive at its layer, one-hot dims bounded by 1 `3b8c4ad` [perf] #bounds #soundness #intervals
  - gate-rebalance: Scale each unit's w1 by 1/c and w2 by c so its largest weights are balanced `3b8c4ad` [perf] #relu #homogeneity #weights

- ilp-packing: Pack many programs into one model with task gating, layout chosen by an integer program @lossless-shrink `fcf107b` [experiment] #superposition #ilp #packing
  - pack-model: Task-tagged token ids, gated MLPs, shared residual slots; each task matches its program `fcf107b` [feature] #packing #gating #exactness
  - pack-layout: Slots by DSatur colouring, certified by a clique bound, integer program when they differ `fcf107b` [perf] #ilp #colouring #residual
  - pack-report: Width and params vs number of programs: packed vs block-diagonal vs separate `fcf107b` [experiment] #scaling #superposition #report

- lossless-shrink: Shrink compiled models with identical outputs: fewer dims, heads and units, per-layer shapes `04fb553` [perf] #compression #exactness #residual
  - shrink-trim: Drop never-read dims, padding heads, unused key columns and dead MLP units `04fb553` [perf] #compression #assemble #padding
  - shrink-reuse: Reuse residual dims after their last read, erasing old values with 2 MLP units each `04fb553` [perf] #compression #liveness #register-allocation
  - shrink-verify: Shrunk and original models agree on every input of small domains, sampled on large ones `04fb553` [experiment] #verification #exactness #testing
  - shrink-report: Dense params before -> after for 10 lib programs over vocab and length `04fb553` [experiment] #compression #scaling #report
- repo-analysis: Understand tracr and verify suspected issues by measurement `9ce2b8c` [experiment] #analysis #compiler #baseline
- fix-key-size: Compile crashes when value dims exceed query/key dims, e.g. vocab 6 with max_seq_len 3 `8cfa45b` [bugfix] #compiler #assemble #crash
- fix-pad-mask: PAD is never masked, so indices-keyed selectors change output when padded `49d45f8` [bugfix] #compiler #transformer #padding
- fix-aggregate-values: Numerical Aggregate value set misses k/n, so a Map on a mean silently miscompiles `37ab6aa` [bugfix] #compiler #soundness #basis-inference
- fix-graph-traversal: Graph extraction and depth DFS lack visited sets, cost grows 4x per diamond `44e54db` [perf] #compiler #complexity #graph
- fix-copy-id: copy() keeps cached unique_id, so a copy can share a label with different encoding `497a089` [bugfix] #rasp #identity #labels
- fix-validator-coverage: Validator passes unsupported exprs and dedups by name instead of label `5627996` [bugfix] #compiler #validation #errors
- project-hygiene: Add .gitignore, python_requires, pinned deps and CI `ac64d3d` [productivity] #packaging #ci #repo
- fix-int-guard: Aggregate binary guard accepts values like 0.5, so a Map on their mean miscompiles `d66a157` [bugfix] #compiler #soundness #basis-inference
- float32-composite-means: Map after a difference of two means loses accuracy from about max_seq_len 14 `1a8d3d4` [bugfix] #compiler #float32 #numerical-mlp
- ci-first-run: Confirm the GitHub Actions workflow passes on its first real run `54f9254` [productivity] #ci #github #verification
- fix-onehot-alloc: One-hot vectors built from a full identity matrix make large MLP compiles O(n^3) `fa0ca17` [perf] #compiler #craft #complexity
- ci-node24: CI actions target the deprecated Node 20 runtime, so every job logs a warning `a78d2c5` [bugfix] #ci #github #actions
