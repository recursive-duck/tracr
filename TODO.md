# TODO

## Backlog

- fix-int-guard: Aggregate binary guard accepts values like 0.5, so a Map on their mean miscompiles [bugfix] #compiler #soundness #basis-inference
- float32-composite-means: Map after a difference of two means loses accuracy from about max_seq_len 14 [bugfix] #compiler #float32 #numerical-mlp
- ci-first-run: Confirm the GitHub Actions workflow passes on its first real run [productivity] #ci #github #verification

## In Progress

## Done

- repo-analysis: Understand tracr and verify suspected issues by measurement `9ce2b8c` [experiment] #analysis #compiler #baseline
- fix-key-size: Compile crashes when value dims exceed query/key dims, e.g. vocab 6 with max_seq_len 3 `8cfa45b` [bugfix] #compiler #assemble #crash
- fix-pad-mask: PAD is never masked, so indices-keyed selectors change output when padded `49d45f8` [bugfix] #compiler #transformer #padding
- fix-aggregate-values: Numerical Aggregate value set misses k/n, so a Map on a mean silently miscompiles `37ab6aa` [bugfix] #compiler #soundness #basis-inference
- fix-graph-traversal: Graph extraction and depth DFS lack visited sets, cost grows 4x per diamond `44e54db` [perf] #compiler #complexity #graph
- fix-copy-id: copy() keeps cached unique_id, so a copy can share a label with different encoding `497a089` [bugfix] #rasp #identity #labels
- fix-validator-coverage: Validator passes unsupported exprs and dedups by name instead of label `5627996` [bugfix] #compiler #validation #errors
- project-hygiene: Add .gitignore, python_requires, pinned deps and CI `ac64d3d` [productivity] #packaging #ci #repo
