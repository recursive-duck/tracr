# Packing report

## What was tested

The 10 tracr lib programs in config.toml were compiled at vocab size V in [4, 8, 16, 32] and max_seq_len N in [4, 8, 16, 32]. For each (V, N), the first K programs (K = 1..10, config order) were packed into one model: the token id selects the program, MLP units of the other programs are switched off by a large negative weight on the active program's `one` dim, and programs share residual dims when no attention value read can see another program's value. The layout is a DSatur colouring of the conflict graph, checked against a clique lower bound; when they differ, an integer program (HiGHS) searches for fewer dims. Every task of the full 10-program pack was run on the same inputs as its program alone: every input on small domains, sampled ones on large domains (settings of [shrink_report]).

Baselines: block-diagonal stacking (the same blocks, each program in its own dims) and K separate trimmed models (`shrink`, no reuse), whose params add up. Params are dense: every weight of every kept block, zeros included.

## Key numbers

- Configs: 16 packed and verified, 0 failed, 1 verified on every input.
- Categorical tasks, decoded tokens differ where the original is decided: **0 of 33,440** sequences.
- Differ only at ties of the original (top-2 scores within 1e-05): 29 of 33,440, in sort_freq V=16 N=32; sort_freq V=32 N=32; sort_freq V=4 N=32; sort_freq V=8 N=32.
- Numerical tasks, output not bitwise equal: 586 of 8,360; largest difference 2.38e-07. Largest difference in categorical scores: 2.13e-03, above 1e-05 only in sort_freq V=16 N=32; sort_freq V=32 N=32; sort_freq V=4 N=32; sort_freq V=8 N=32.
- All 10 programs, summed over configs: packed 62,076,515, block-diagonal 132,620,160 (2.14x the packed model), separate trimmed models 38,013,232 (packed is 1.63x that), separate reused models 37,601,684.
- Residual width of the full pack / widest program alone: median 1.10, max 1.54. Sum of widths / packed width: median 2.49.
- Cost of one model over K separate trimmed models (dense params ratio, median over configs), K: packed / block-diagonal: 1: 1.00 / 1.00, 2: 1.17 / 1.59, 3: 1.31 / 2.24, 4: 1.33 / 3.15, 5: 1.62 / 3.21, 6: 1.57 / 3.43, 7: 1.60 / 3.56, 8: 1.65 / 3.75, 9: 1.77 / 4.44, 10: 1.84 / 4.87.
- Layout: DSatur meets the clique bound in 128 of 160 packs. The integer program ran 32 times: proved optimal 13, beat DSatur 0, hit the 60 s limit 15, skipped as too large 4. Shared slots above the bound: max 3, median 0.

Full pack (10 programs) per config:

| V | N | width ↓ | widest | sum | bound | solver / status | packed ↓ | block-diagonal ↓ | x block ↑ | separate trim | separate reuse | max gate ↓ | max weight ↓ | pack s | peak RSS MB |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 62 | 50 | 243 | 61 | dsatur: HiGHS Status 7: Optimal | 51,274 | 200,961 | **3.92** | 29,301 | 25,821 | 2.01e+02 | 1.00e+02 | 0.2 | 1,170 |
| 4 | 8 | 130 | 118 | 395 | 129 | dsatur: HiGHS Status 7: Optimal | 187,590 | 569,985 | **3.04** | 96,549 | 91,177 | 5.77e+02 | 1.00e+02 | 2.9 | 1,093 |
| 4 | 16 | 362 | 350 | 795 | 361 | dsatur: HiGHS Status 7: Optimal | 1,142,110 | 2,508,225 | 2.20 | 601,989 | 588,993 | 2.18e+03 | 1.00e+02 | 53.9 | 1,868 |
| 4 | 32 | 1,210 | 1,198 | 1,979 | 1,209 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 10,283,790 | 16,819,521 | 1.64 | 6,117,957 | 6,063,601 | 8.45e+03 | 1.28e+02 | 147.7 | 9,574 |
| 8 | 4 | 91 | 59 | 307 | 88 | dsatur: HiGHS Status 7: Optimal | 92,729 | 312,833 | **3.37** | 44,273 | 39,891 | 2.01e+02 | 1.00e+02 | 0.3 | 1,040 |
| 8 | 8 | 138 | 126 | 475 | 137 | dsatur: HiGHS Status 7: Optimal | 234,462 | 807,025 | **3.44** | 132,209 | 125,233 | 5.77e+02 | 1.00e+02 | 2.9 | 1,135 |
| 8 | 16 | 370 | 358 | 907 | 369 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 1,309,430 | 3,209,873 | 2.45 | 708,977 | 693,193 | 2.18e+03 | 1.00e+02 | 157.6 | 1,910 |
| 8 | 32 | 1,218 | 1,206 | 2,155 | 1,217 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 11,131,302 | 19,694,545 | 1.77 | 6,487,409 | 6,426,361 | 8.45e+03 | 1.28e+02 | 139.2 | 9,527 |
| 16 | 4 | 155 | 107 | 435 | 152 | dsatur: HiGHS Status 7: Optimal | 217,465 | 610,305 | 2.81 | 89,193 | 85,051 | 2.01e+02 | 1.00e+02 | 0.9 | 1,165 |
| 16 | 8 | 224 | 179 | 635 | 221 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 495,264 | 1,403,985 | 2.83 | 241,161 | 233,179 | 5.77e+02 | 1.00e+02 | 114.4 | 1,336 |
| 16 | 16 | 386 | 374 | 1,131 | 385 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 1,662,502 | 4,871,217 | 2.93 | 1,042,761 | 1,020,729 | 2.18e+03 | 1.00e+02 | 93.5 | 2,063 |
| 16 | 32 | 1,234 | 1,222 | 2,507 | 1,233 | dsatur: skipped: 21,382,094 nonzeros > 20,000,000 | 12,857,046 | 26,120,433 | 2.03 | 7,657,929 | 7,582,825 | 8.45e+03 | 1.28e+02 | 6.7 | 8,163 |
| 32 | 4 | 283 | 203 | 691 | 280 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 614,393 | 1,500,161 | 2.44 | 238,937 | 237,375 | 2.01e+02 | 1.00e+02 | 88.4 | 1,633 |
| 32 | 8 | 416 | 339 | 955 | 413 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 1,345,760 | 3,089,425 | 2.30 | 609,593 | 602,219 | 5.77e+02 | 1.00e+02 | 63.2 | 2,049 |
| 32 | 16 | 688 | 611 | 1,579 | 685 | dsatur: HiGHS Status 13: model_status is Time limit reached; primal_status is None | 4,019,984 | 9,226,097 | 2.30 | 2,189,561 | 2,166,507 | 2.18e+03 | 1.00e+02 | 88.2 | 3,693 |
| 32 | 32 | 1,266 | 1,254 | 3,211 | 1,265 | dsatur: skipped: 31,278,094 nonzeros > 20,000,000 | 16,431,414 | 41,675,569 | 2.54 | 11,725,433 | 11,619,529 | 8.45e+03 | 1.28e+02 | 5.7 | 9,014 |

width = residual dims of the packed model; widest / sum = trimmed widths of the programs alone; bound = clique lower bound on width for this conflict rule; x block = block-diagonal / packed params.

## What it means

Only one program runs per input, so their values never meet and can share residual dims. The packed stream is 1.10x as wide as the widest program (median), while the programs' own widths add up to 2.49x the packed width. A program narrower than the pack so far mostly fits into dims that are already there.

Every weight matrix spans the residual width. Block-diagonal stacking puts all programs' heads and units on the sum of their widths; packing puts them on about the widest one. Against K separate trimmed models, the median cost of one model rises from 1.00 at K=1 to 1.84 at K=10 for packing, and to 4.87 for block-diagonal. So packing grows faster than K: each small program pays for the widest program's width. That is why the packed model still costs 1.63x the separate models summed. The gain is over one stacked model: 2.14x fewer params summed over configs. The steps in the left panels are where the widest programs join; x block is smallest where the widest program is most of the summed width.

Categorical score differences above the tie tolerance are float32 rounding at attention ties of the original. A different matrix width changes the summation order, and near-tied keys then share attention weight differently. Decoded outputs that the original decides do not change.

## What it does not prove

- Only the full 10-program pack is checked per config; the smaller packs of the K sweep are built by the same code and only measured. Unit tests check 2-10 program packs on every input up to length 4.
- Sampled configs are checked on a few hundred inputs, not all of them.
- The clique bound and the integer program are optimal for this conflict rule only: no erasing, all programs starting at layer 0, one dim per value. Other rules can give narrower models.
- The K sweep uses one program order; other orders give other curves.
- Gates come from sound interval bounds, not from the inputs seen. The largest gate is 8.45e+03 and the largest weight after rebalancing 128. They hold for any input, but they are not shown to be the smallest gates that do, and the model is not shown to be a good starting point for training.
- Dense params include zeros inside kept blocks.

Plot: pack_report.png
