# Progress

All entries below are from 2026-11-03 01:52. Each compares against the baseline of 5,756 passed, 1 xfailed.
After all seven commits together: 5,823 passed, 1 xfailed, 892 subtests passed in 58.42 s.

## Crash on large vocabularies (`8cfa45b`)

- **What:** attention `key_size` now covers the value size of every head.
- **Why:** compiling crashed whenever a vocabulary was larger than the key width, for example vocab 6 at max_seq_len 3.
- **How:** added `w_ov.matrix.shape[1]` to the `key_size` maximum in `assemble.py`.
- **Result:** vocab 6, 7, 8 and 20 went from ValueError to 0 mismatches against the RASP interpreter.

## Padding changed answers (`49d45f8`)

- **What:** compiled models now ignore PAD positions in attention.
- **Why:** selectors keyed on indices counted PAD as real input. Width on [1,2] gave 4 with two PADs instead of 2.
- **How:** `craft_model_to_transformer` passes the encoder's PAD id through `assemble_craft_model` to the model.
- **Result:** mismatches over 174 padded pairs fell 174 -> 0 and 122 -> 0 on the two broken programs.

## Wrong answers after averaging (`37ab6aa`)

- **What:** a numerical Aggregate now lists every possible mean, and numerical MLPs sharpen their steps to tell close values apart.
- **Why:** the compiler only knew means like 1/3, not 2/3, so a later Map silently rounded 2/3 to 1/2.
- **How:** the value set in `basis_inference.py` enumerates every reduced k/n, and `numerical_mlp.py` raises L to at least 4 / min gap.
- **Result:** `x>0.5` went 25/62 -> 0/62 wrong at length 5, and 206/300 -> 0/300 at length 20. `x*10` went 35/62 -> 0/62 and 270/300 -> 0/300.
- **Cost:** MLP hidden size grows 41 -> 257 at length 20.
- **Limit:** a Map after a difference of two means loses accuracy from about length 14, because float32 runs out of precision.

## Compile time on shared subprograms (`44e54db`)

- **What:** graph extraction and depth computation now visit each expression once.
- **Why:** cost grew 4x for every reused subexpression, so a 26-node program took 20 s, and a 34-node one did not finish in 120 s.
- **How:** added a visited set in `rasp_to_graph.py`, and replaced the DFS in `craft_graph_to_model.py` with a topological pass.
- **Result:** depth at n=12 went 20.018 -> 0.000227 s, and full compile 20.785 -> 1.474 s. Output on 8 lib programs is identical across 328 arrays.

## Copies sharing an ID (`497a089`)

- **What:** `copy()` now always produces a new ID.
- **Why:** after a label was read, `rasp.numerical(x)` shared that label with `x`, and the compiler merged the two nodes.
- **How:** `copy()` drops the cached `unique_id`.
- **Result:** two programs that crashed at compile now match RASP on all 81 inputs.

## Validator coverage (`5627996`)

- **What:** the validator now reports everything the compiler cannot handle.
- **Why:** it compared shared default names, so it skipped children. Six unsupported forms also got through and failed later with mixed errors.
- **How:** dedupes by label, and checks S-Op and selector types against the supported set.
- **Result:** those forms now all raise NotImplementedError. 49 of 49 compilable programs still pass, and 9 of 9 unsupported ones are flagged.

## Packaging and CI (`ac64d3d`)

- **What:** `pyproject.toml` replaces `setup.py`, with bounded dependencies, `uv.lock`, `.gitignore` and a GitHub Actions test workflow.
- **Why:** there were no pins, no Python floor, no CI, and running tests dirtied the tree.
- **How:** set `requires-python >=3.11` from uv resolution, since chex 0.1.92 needs it. Direct dependencies lock at the tested versions.
- **Result:** built wheels hold the same 61 files, and a fresh install runs the README example. The workflow has not yet run on GitHub.

## Averaging non-binary values (`d66a157`, 2026-11-03)

- **What:** a numerical Aggregate of values other than 0 and 1 is now rejected at compile time.
- **Why:** the guard checked `int(x)`, so 0.5 or 1.9 passed as binary and got an incomplete value set. A Map on their mean was wrong on 9/30 inputs at max_seq_len 4.
- **How:** compare values directly in `basis_inference.py`, and remove the fallback that only those values reached.
- **Result:** 9/30 wrong -> NotImplementedError. Suite 5,823 -> 5,827 passed.

## Float32 error after combined means (`1a8d3d4`, 2026-11-03)

- **What:** each step in a numerical MLP now gets a steepness sized to its own gap, not the smallest gap overall.
- **Why:** one huge steepness made every hidden activation large, and float32 loses precision when it subtracts two large numbers. The same model in float64 was exact (error 1.21e-11).
- **How:** `_threshold_large_numbers` in `numerical_mlp.py` returns L_k = max(L, 4 / gap_k) for each threshold.
- **Result:** `pair_balance` then `x*10`, 60 inputs: max error 4.48e-03 -> 1.67e-04 at length 14, with wrong answers 55 -> 0; 4.76e-04 -> 3.43e-05 at length 10. Hidden sizes unchanged. Suite 5,827 -> 5,828 passed.
- **Length 20** (measurable after the next fix): 6.82e-03 -> 1.77e-04, wrong answers 55 -> 0.

## CI first run (`54f9254`, 2026-11-03)

- **Result:** the GitHub Actions workflow passed on Python 3.11, 3.12 and 3.13 (run 41592589029, 2-3 min per job).

## Slow compile of large MLPs (`fa0ca17`, 2026-11-03)

- **What:** building a one-hot vector no longer allocates a full identity matrix.
- **Why:** each call made an n x n matrix to keep one row, once per hidden unit, so compile cost grew as n^3. 82% of a 14 s compile went there.
- **How:** `vector_from_basis_direction` in `bases.py` fills a zero vector and sets one entry.
- **Result:** compile 15.7 -> 1.11 s at length 14 (3,241 hidden). Length 20 (12,973 hidden) went from not finishing in minutes to 2.4 s.
