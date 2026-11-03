# TODO

## Backlog

- nary-sum: n-ary linear sum primitive so a sum tree of numerical SOps is one MLP, not log2(n) layers @bitsplit-compare [perf] #compiler #mlp #layers
- llm-shapes: RMSNorm-safe codes, RoPE positions, uniform head size, bf16 audit @scale-to-llm [feature] #architecture #precision #llm
- id-codes: ID tokens embed and unembed as fixed bit codes shared with the DB keys; no ID embedding table @scale-to-llm [feature] #ids #encoding #tools
- hybrid-train: Frozen compiled modules inside trainable layers; check training leaves them exact @llm-shapes [experiment] #training #modules #exactness
- bit-count: Exact counting that outputs bits, without N decode units or 1/N^2 resolution @scale-to-llm [experiment] #counting #precision #research
- compiler-symbolic: Symbolic value sets and vectorized matrix building; Python loops are O(m) to O(m^2) per op @scale-to-llm [perf] #compiler #runtime #scaling
- pack-rasp-rerun: Rerun the full pack report with the vs-RASP column; only the smoke grid ran @scale-check [experiment] #verification #packing #reports
- const-sop: Numerical constant SOp on the existing `one` direction; drops the `ones` Map MLP layer @exact-copy [perf] #compiler #mlp #layers
- marker-window: Nearest-preceding-marker select without one-hot indices, so copy scales to 1M context @exact-copy [feature] #attention #positions #scaling
- bit-mlp-width: Binary MLPs keep one hidden unit per input value, so params still grow ~V^1.1; make them log-width @bit-encoding [perf] #mlp #encoding #scaling
- bit-probe-4096: Add binary-only V=N=4,096 rows to the bit_encoding probe; one-hot agg_eq ran 22 min without finishing @bit-encoding [experiment] #encoding #probe #scaling

## In Progress

- scale-to-llm: Compile at vocab 100-250k and context 1M: bit codes, bit-split compares, bit arithmetic, scalable checking [experiment] #scaling #encoding #llm

## Done

- bit-encoding: Binary encoding next to categorical/numerical; EQ heads as AND of bit equalities, unembed from bits @scale-to-llm `2873d13` `b3da568` `5990e8c` [feature] #encoding #residual #compiler

- exact-copy: Compiled head that copies a multi-token ID from a tool result verbatim @bit-encoding `b9af7c2` [feature] #copy #ids #attention

- scale-check: Vectorized RASP evaluator; shrink/pack reports check against RASP, not only the compiled original @scale-to-llm `321d668` `30ee62a` `88e5442` [feature] #verification #evaluator #reports

- bit-arith: Adders and comparators as small MLP circuits on bits instead of m1*m2 lookup tables @scale-to-llm `c6d16ed` `10b8008` [feature] #mlp #arithmetic #bits

- bitsplit-compare: Less-than as log2(m) AND heads over bit prefixes; make_sort on bit-split counts @scale-to-llm `f7f7011` [perf] #attention #sort #bits
- split-pair-compare: Sort by two counts instead of one comparison on a 1,024-value pair key; AND of Selects compiles as one head @ilp-packing `bc31d24` `1402531` [perf] #sort #attention #residual
- fix-logit-scale: Attention divides logits by sqrt(key_size) uncorrected, so sort at V=N=32 is wrong on 81% of inputs `5d345a6` [bugfix] #compiler #attention #soundness
- merge-shared: Merge identical computations across packed programs? No-go: repeats save at most 6% of units, no width @ilp-packing [experiment] #dedup #packing #probe
- ilp-time-limit: Cut packing time with a 5 s integer-program limit; it never beat DSatur in 64 runs @pack-layout `658936b` [perf] #ilp #runtime #packing
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
