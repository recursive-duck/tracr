# Packing report

## What was tested

The 10 tracr lib programs in config.toml were compiled at vocab size V in [4, 8, 16, 32] and max_seq_len N in [4, 8, 16, 32]. For each (V, N), the first K programs (K = 1..10, config order) were packed into one model: the token id selects the program, MLP units of the other programs are switched off by a large negative weight on the active program's `one` dim, and programs share residual dims when no attention value read can see another program's value. The layout is a DSatur colouring of the conflict graph, checked against a clique lower bound; when they differ, an integer program (HiGHS) searches for fewer dims. Every task of the full 10-program pack was run on the same inputs as its program alone: every input on small domains, sampled ones on large domains (settings of [shrink_report]).

Baselines: block-diagonal stacking (the same blocks, each program in its own dims) and K separate trimmed models (`shrink`, no reuse), whose params add up. Params are dense: every weight of every kept block, zeros included.

## Key numbers

- Configs: 16 packed and verified, 0 failed, 1 verified on every input.
- Categorical tasks, decoded tokens differ where the original is decided: **0 of 33,440** sequences.
- Differ only at ties of the original (top-2 scores within 1e-05): 0 of 33,440.
- Numerical tasks, output not bitwise equal: 356 of 8,360; largest difference 1.19e-07. Largest difference in categorical scores: 7.63e-06.
- All 10 programs, summed over configs: packed 10,033,942, block-diagonal 42,406,272 (**4.23x** the packed model), separate trimmed models 6,415,696 (packed is 1.56x that), separate reused models 5,490,997.
- Residual width of the full pack / widest program alone: median 1.18, max 2.03. Sum of widths / packed width: median 4.25.
- Cost of one model over K separate trimmed models (dense params ratio, median over configs), K: packed / block-diagonal: 1: 1.00 / 1.00, 2: 1.17 / 1.59, 3: 1.31 / 2.24, 4: 1.33 / 3.15, 5: 1.41 / 3.93, 6: 1.43 / 4.59, 7: 1.48 / 4.91, 8: 1.56 / 5.40, 9: 1.53 / 6.21, 10: 1.59 / 6.97.
- Layout: DSatur meets the clique bound in 130 of 160 packs. The integer program ran 30 times: proved optimal 20, beat DSatur 0, hit the 5 s limit 10, skipped as too large 0. Shared slots above the bound: max 3, median 0.

Full pack (10 programs) per config:

| V | N | width ↓ | widest | sum | bound | solver / status | packed ↓ | block-diagonal ↓ | x block ↑ | separate trim | separate reuse | max gate ↓ | max weight ↓ | pack s | peak RSS MB |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 57 | 48 | 225 | 57 | dsatur: not run: DSatur meets the clique bound | 46,227 | 182,475 | **3.95** | 26,027 | 22,132 | 2.01e+02 | 6.63e+02 | 0.0 | 1,187 |
| 4 | 8 | 75 | 63 | 325 | 74 | dsatur: HiGHS Status 7: Optimal | 99,225 | 429,975 | **4.33** | 62,019 | 54,608 | 5.77e+02 | 8.72e+02 | 0.3 | 1,125 |
| 4 | 16 | 123 | 111 | 525 | 122 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 324,105 | 1,383,375 | **4.27** | 208,595 | 183,496 | 2.18e+03 | 1.18e+03 | 6.4 | 1,416 |
| 4 | 32 | 219 | 207 | 925 | 218 | dsatur: HiGHS Status 7: Optimal | 1,404,009 | 5,930,175 | **4.22** | 950,643 | 815,288 | 8.45e+03 | 1.64e+03 | 10.1 | 7,549 |
| 8 | 4 | 73 | 52 | 273 | 70 | dsatur: HiGHS Status 7: Optimal | 69,715 | 260,715 | **3.74** | 35,367 | 30,238 | 2.01e+02 | 7.48e+02 | 0.2 | 995 |
| 8 | 8 | 83 | 71 | 373 | 82 | dsatur: HiGHS Status 7: Optimal | 121,761 | 547,191 | **4.49** | 76,655 | 66,892 | 5.77e+02 | 8.72e+02 | 0.6 | 1,122 |
| 8 | 16 | 131 | 119 | 573 | 130 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 364,049 | 1,592,367 | **4.37** | 236,127 | 205,892 | 2.18e+03 | 1.18e+03 | 10.8 | 1,400 |
| 8 | 32 | 227 | 215 | 973 | 226 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 1,487,985 | 6,378,015 | **4.29** | 1,013,183 | 862,516 | 8.45e+03 | 1.64e+03 | 17.0 | 7,647 |
| 16 | 4 | 113 | 63 | 369 | 110 | dsatur: HiGHS Status 7: Optimal | 140,459 | 458,667 | **3.27** | 59,423 | 55,109 | 2.01e+02 | 9.38e+02 | 0.5 | 1,159 |
| 16 | 8 | 121 | 87 | 469 | 118 | dsatur: HiGHS Status 7: Optimal | 212,355 | 823,095 | **3.88** | 111,303 | 98,654 | 5.77e+02 | 1.02e+03 | 0.5 | 1,043 |
| 16 | 16 | 147 | 135 | 669 | 146 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 450,849 | 2,051,823 | **4.55** | 296,567 | 255,004 | 2.18e+03 | 1.18e+03 | 5.6 | 1,397 |
| 16 | 32 | 243 | 231 | 1,069 | 242 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 1,662,849 | 7,315,167 | **4.40** | 1,143,639 | 961,292 | 8.45e+03 | 1.64e+03 | 15.9 | 8,056 |
| 32 | 4 | 193 | 95 | 561 | 190 | dsatur: HiGHS Status 7: Optimal | 351,067 | 1,020,459 | 2.91 | 129,039 | 126,613 | 2.01e+02 | 1.23e+03 | 2.4 | 1,403 |
| 32 | 8 | 201 | 119 | 661 | 198 | dsatur: HiGHS Status 7: Optimal | 468,531 | 1,540,791 | **3.29** | 202,103 | 192,469 | 5.77e+02 | 1.30e+03 | 2.1 | 1,460 |
| 32 | 16 | 217 | 167 | 861 | 214 | dsatur: HiGHS Status 7: Optimal | 790,531 | 3,136,623 | **3.97** | 438,951 | 384,670 | 2.18e+03 | 1.41e+03 | 2.3 | 1,599 |
| 32 | 32 | 275 | 263 | 1,261 | 274 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 2,040,225 | 9,355,359 | **4.59** | 1,426,055 | 1,176,124 | 8.45e+03 | 1.64e+03 | 11.3 | 9,616 |

width = residual dims of the packed model; widest / sum = trimmed widths of the programs alone; bound = clique lower bound on width for this conflict rule; x block = block-diagonal / packed params.

## What it means

Only one program runs per input, so their values never meet and can share residual dims. The packed stream is 1.18x as wide as the widest program (median), while the programs' own widths add up to 4.25x the packed width. A program narrower than the pack so far mostly fits into dims that are already there.

Every weight matrix spans the residual width. Block-diagonal stacking puts all programs' heads and units on the sum of their widths; packing puts them on about the widest one. Against K separate trimmed models, the median cost of one model rises from 1.00 at K=1 to 1.59 at K=10 for packing, and to 6.97 for block-diagonal. So packing grows faster than K: each small program pays for the widest program's width. That is why the packed model still costs 1.56x the separate models summed. The gain is over one stacked model: 4.23x fewer params summed over configs. The steps in the left panels are where the widest programs join; x block is smallest where the widest program is most of the summed width.

Categorical score differences above the tie tolerance are float32 rounding at attention ties of the original. A different matrix width changes the summation order, and near-tied keys then share attention weight differently. Decoded outputs that the original decides do not change.

## What it does not prove

- Only the full 10-program pack is checked per config; the smaller packs of the K sweep are built by the same code and only measured. Unit tests check 2-10 program packs on every input up to length 4.
- Sampled configs are checked on a few hundred inputs, not all of them.
- The clique bound and the integer program are optimal for this conflict rule only: no erasing, all programs starting at layer 0, one dim per value. Other rules can give narrower models.
- The K sweep uses one program order; other orders give other curves.
- Gates come from sound interval bounds, not from the inputs seen. The largest gate is 8.45e+03 and the largest weight after rebalancing 1.64e+03. They hold for any input, but they are not shown to be the smallest gates that do, and the model is not shown to be a good starting point for training.
- Dense params include zeros inside kept blocks.

Plot: pack_report.png
