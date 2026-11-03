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

## CI on Node 24 (`a78d2c5`, 2026-11-03)

- **What:** the test workflow now pins `actions/checkout` v7.0.1 and `astral-sh/setup-uv` v10.3.3.
- **Why:** the old pins (v4 and v6) declared the Node 20 runtime, which GitHub deprecated, so every job logged a warning.
- **How:** both new pins are full commit SHAs whose `action.yml` declares node24. The inputs we pass are unchanged.
- **Result:** run 41594629705 passed on 3.11, 3.12 and 3.13, with Node 20 warnings 3 -> 0.

## Lossless model shrinking (`04fb553`, 2026-11-03 09:30)

- **What:** `tracr/compiler/shrinking.py` turns a compiled model into a smaller one that computes the same outputs. `shrink(model, reuse=False)` removes padding; `reuse=True` also recycles residual dimensions.
- **Why:** compiled models are mostly padding. Every layer is as wide as the widest one and every residual dimension exists everywhere, so only 0.05-4.9% of weights are nonzero. The size people quote for a compiled program is mostly not the program.
- **How:** a fixed-point pass drops residual dims nobody reads or writes, heads with no effect, unused query/key/value columns and dead MLP units, and gives each layer its own shapes. With reuse, a dim whose last read has passed is zeroed by two ReLU units (ReLU(x) and ReLU(-x) write exactly -x) and its slot is handed to a later value, assigned greedily by birth time. `compare()` checks a shrunk model against the original and counts flips where the original itself is tied separately.
- **Result:** 10 lib programs × V, N in {4, 8, 16, 32}, 160 configs: dense params 3,559,320,080 -> 38,013,232 (trim) -> 37,601,684 (reuse). shuffle_dyck2 at V=32, N=32: 893,494,508 -> 111,173 (8,037x). **0 decided mismatches in 66,880 categorical comparisons.** 29 comparisons flip at exact ties of the original (sort_freq, N=32), where the original is also wrong against RASP. Suite 5,829 -> 5,877 passed (48 new shrinking tests).
- **Limits:** shrunk models use per-layer shapes, so they run in their own forward, not the haiku model class. Numerical outputs can differ in the last float32 bits (max 2.38e-07). Large configs are checked on 256 random inputs, not all.
- **Report:** `experiments/shrink_report/summary.md`, plot `shrink_report.png`; settings in `config.toml` [shrink_report].

## Many programs in one model (`fcf107b`, 2026-11-03 10:25)

- **What:** `tracr/compiler/packing.py` packs K compiled programs into one model. The token id picks the program, and every task gives the same outputs as its program run alone.
- **Why:** stacking K models block-diagonally makes every weight matrix span the sum of all their widths. Only one program runs per input, so their values never meet and could share residual dims instead.
- **How:** each program keeps its own `one` dim, which is 1 only while it runs. Every MLP unit gets a large negative weight on the other programs' `one` dims, so it stays off. Values may share a slot unless an attention head could read one while the other is there. Slots come from a DSatur colouring of that conflict graph, checked against a clique lower bound. An integer program (HiGHS) runs when they differ.
- **Result:** 10 lib programs, V and N in {4, 8, 16, 32}, 16 configs: **0 decided mismatches in 33,440 categorical comparisons.** The 29 tie flips (sort_freq, N=32) are the same as in shrinking with reuse. The packed width is a median 1.10x the widest program, against 2.49x for the sum of widths. Summed dense params: packed 62,076,515, block-diagonal 132,620,160 (2.14x), separate trimmed models 38,013,232 (packed is 1.63x those). The integer program never beat DSatur (proved it optimal 13 times out of 32). Suite 5,877 -> 5,885 passed (8 new packing tests).
- **Limits:** params grow faster than K, because each small program pays for the widest program's width. Gates reach 1.6e+09, though a probe at V=4, N=4 shows 100 is enough; see gate-bounds in TODO.md. Only the full 10-program pack is verified in each config.
- **Report:** `experiments/pack_report/summary.md`, plot `pack_report.png`; settings in `config.toml` [pack_report].

## Smaller packing gates (`3b8c4ad`, 2026-11-03 10:45)

- **What:** the weight that switches an MLP unit off under another task (its gate) is now sized from a tight bound, and each unit's input and output weights are balanced.
- **Why:** gates reached 1.6e+09, while the largest input a gate ever had to cancel was 100. Weights that large are fragile and hide the model's real scale.
- **How:** for each other task j, the gate covers only j's values that exist before the unit's layer, plus positions, using lower/upper intervals pushed through the layers with signs kept. One-hot dims are kept in [0, 1]. Then each unit's input weights are divided by a power of 2 and its output weights multiplied by it. ReLU(x / c) c = ReLU(x), and powers of 2 scale floats exactly.
- **Result:** **max gate 1.6e+09 -> 8.45e+03 at N=32, 1.945e+08 -> 201 at V=4 N=4 (about 190,000x and 970,000x smaller).** Largest weight 100 (N <= 16) and 128 (N=32). Probe at V=4, N=4: each gate is at most 2.5x the largest input it must cancel on any input. In all 16 configs, widths, mismatches, tie flips, non-bitwise numerical outputs and max differences are identical to before. Suite 5,885 -> 5,889 passed (gates shut on every input of every other task, with and without the one-hot clamp; the one-hot clamp gives smaller gates; rebalancing is bitwise equal).
- **Limits:** the one-hot clamp trusts that tracr keeps categorical dims in [0, 1]; the soundness test checks this on every input up to length 4, not in general. Gates are still 2x a bound, not the minimum that works.
- **Report:** `experiments/pack_report/summary.md` (new max weight column), probe script in the session scratchpad.

## Faster packing: 5 s integer-program limit (`658936b`, 2026-11-03 11:05)

- **What:** the integer program that searches for a narrower slot layout now stops after 5 s instead of 60 s (`config.toml` [pack_report] and the `ILP_TIME_LIMIT_S` default).
- **Why:** it was nearly all of the packing time, up to 356 s per config, and in two full reports it never found a narrower layout than DSatur (0 of 64 runs). A GPU would not help: branch and bound is serial CPU work, and the parts a GPU could speed up (the forward passes) are a small share.
- **How:** one constant. DSatur and the clique lower bound are unchanged, so the reported gap to the best possible width is still exact.
- **Result:** **summed pack time 2,108 -> 294 s (7.2x), summed config time 3,711 -> 1,574 s (2.4x)**, wall 657 -> 501 s. Widths are identical in all 160 packs (16 configs x 10 K); 0 decided mismatches before and after. The N=32 configs are now limited by compiling the programs, not packing.
- **Limits:** the integer program proves DSatur optimal less often (13 -> 8 of 32 runs). A narrower layout that needs more than 5 s of search would be missed; none was found with 60 s either.
- **Report:** `experiments/pack_report/summary.md`.

## Attention logits divided by sqrt(key_size) (`5d345a6`, 2026-11-03 11:20)

- **What:** the assembler multiplies every query matrix by sqrt(key_size), so compiled attention logits equal 100 x the craft score again.
- **Why:** found while checking a smaller sort against RASP: compiled lib `sort` at V=N=32 gave the wrong answer on 242 of 300 random inputs, `sort_freq` on 232 (0 at N <= 16). The attention layer divides logits by sqrt(key_size) and nothing undid it. The compiler's sharpness factor of 100 fell to 100 / 32 = 3.1 at key_size 1,024, so the start-of-sequence position kept about half the attention and a count of 0 decoded as 1.
- **How:** one scale in `assemble.py`. Shrinking and packing copy the same divisor from the model config, so they need no change.
- **Result:** **wrong answers 242/300 -> 0/300 (sort) and 232/300 -> 0/300 (sort_freq) at V=N=32.** New test: logits equal 100 x the score at vocab 3 and 40 (fails before the fix). The packing rebalance test now compares MLP weights, because the scaled query weights (100 x sqrt(key_size)) are now the largest. Suite 5,895 -> 5,897 passed.
- **Limits:** the shrink and pack reports compare shrunk or packed models against the compiled original, not against RASP, which is why they could not see this. Query weights now reach 100 x sqrt(key_size).

## AND of Selects compiles as one head (`bc31d24`, 2026-11-03 11:25)

- **What:** `SelectorWidth` and `Aggregate` accept `SelectorAnd` of Selects; before, the validator rejected them.
- **Why:** sort breaks ties by comparing a (key, position) pair, which has 1,024 values at V=N=32. A "less than" head over m values needs at least m - 1 query/key dims, so that pair alone set the width of every pack. Counting "equal key and earlier position" needs only the two small comparisons at once.
- **How:** the head scores how many predicates hold, each Select reading its own query and key values. A selected pair scores n. The start-of-sequence position scores n (selector width) or n - 0.5 (aggregate), so a pair missing any predicate (at most n - 1) never wins. `categorical_attn` and `selector_width` take this n as `max_logit`.
- **Result:** 0 mismatches against RASP on all 120 inputs (V=3, N=4) for selector width and numerical/categorical aggregates with 2 and 3 predicates. Two shared test cases added; the validator still rejects OR, NOT and an AND containing them. Suite 5,889 -> 5,895 passed.
- **Limits:** only AND; an OR cannot be one head of this kind in general.

## Sort by two counts (`1402531`, 2026-11-03 12:10)

- **What:** `lib.make_sort` (and so `make_sort_freq`) finds each element's target position as #(smaller keys) + #(equal keys at earlier positions), instead of comparing a (key, position) pair key.
- **Why:** the pair key x + min_key * i / max_seq_len has |keys| x max_seq_len values (1,024 at V=N=32). It is read only by a "less than" head, which needs at least m - 1 query/key dims for m values, so it alone set the width of every pack.
- **How:** two selector widths, the second over an AND of `Select(keys, keys, EQ)` and `Select(indices, indices, LT)` (`bc31d24`), added by a categorical SequenceMap capped at max_seq_len. Sums the value sets allow but inputs never reach map to max_seq_len, which matches no index, so trimming drops their MLP units (capping at max_seq_len - 1 instead kept them: 560,934 vs 334,572 params at V=N=32). `min_key` is kept for compatibility but unused.
- **Result:** same outputs: old and new definitions agree under RASP, and the compiled model matches RASP, on all 1,364 inputs at V=4, N=5. At V=N=32 (trimmed): **sort 1,155 -> 196 dims, 5,110,875 -> 334,572 params (15.3x); sort_freq 1,254 -> 263 dims, 6,043,026 -> 519,951 params (11.6x).** **Full 10-program pack at V=N=32: width 1,266 -> 275, params 16,431,414 -> 2,040,225 (8.1x).** **Packed params summed over 16 configs 62,076,515 -> 10,033,942 (6.2x)**; block-diagonal / packed 2.14x -> 4.23x; packed / separate trimmed 1.63x -> 1.56x. Tie flips in both reports 29 -> 0 (they came from the logit-scale bug, `5d345a6`). Suite 5,897 passed.
- **Limits:** the reports still compare shrunk or packed models with the compiled original, not with RASP. The largest weight after packing rebalance rose from 128 to 1.64e+03 because query matrices now carry 100 x sqrt(key_size). The packed model still costs 1.56x the separate trimmed models.
- **Report:** `experiments/pack_report/summary.md`, `experiments/shrink_report/summary.md`.
