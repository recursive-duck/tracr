# TODO

## Backlog

- fix-aggregate-values: Numerical Aggregate value set misses k/n, so a Map on a mean silently miscompiles [bugfix] #compiler #soundness #basis-inference
- fix-key-size: Compile crashes when value dims exceed query/key dims, e.g. vocab 6 with max_seq_len 3 [bugfix] #compiler #assemble #crash
- fix-pad-mask: PAD is never masked, so indices-keyed selectors change output when padded [bugfix] #compiler #transformer #padding
- fix-graph-traversal: Graph extraction and depth DFS lack visited sets, cost grows 4x per diamond [perf] #compiler #complexity #graph
- fix-validator-coverage: Validator passes unsupported exprs and dedups by name instead of label [bugfix] #compiler #validation #errors
- fix-copy-id: copy() keeps cached unique_id, so a copy can share a label with different encoding [bugfix] #rasp #identity #labels
- project-hygiene: Add .gitignore, python_requires, pinned deps and CI [productivity] #packaging #ci #repo

## In Progress

## Done

- repo-analysis: Understand tracr and verify suspected issues by measurement `9ce2b8c` [experiment] #analysis #compiler #baseline
