# Lossless shrink report

## What was tested

Every tracr lib program in config.toml, compiled at vocab size V in [4, 8, 16, 32] and max_seq_len N in [4, 8, 16, 32], was shrunk two ways: trim only (drop dims, heads, head columns and MLP units that cannot affect the output, and give each layer its own shapes) and trim + reuse (residual dims whose value is dead are erased with two ReLU units and reused). Each shrunk model was run on the same inputs as the original: every input when the domain has at most 2,000 sequences, else 256 random ones (seed 0).

## Key numbers

- Configs: 160 shrunk and verified, 0 failed (compile timeout or error), 10 verified on every input.
- Categorical programs, decoded tokens differ where the original is decided: **0 of 66,880** sequence comparisons (both modes).
- Differ only at ties of the original (top-2 scores within 1e-05): 0 of 66,880.
- Numerical programs, output not bitwise equal: 712 of 16,720; largest difference 1.19e-07.
- Largest difference in categorical output scores (before argmax): 7.63e-06.
- Dense params summed over all configs: 3,372,997,994 -> 6,415,696 (trim) -> 5,490,997 (reuse); nonzero weights in the originals: 3,062,016.

| program | configs | median x trim ↑ | median x reuse ↑ | median x d_model ↑ | reuse smaller | largest (V, N) | params before | trim ↓ | reuse ↓ | nonzero |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| length | 16 | 1.9 | 1.9 | 1.46 | 0/16 | (32, 32) | 34,205 | 18,156 | 18,156 | 1,421 |
| reverse | 16 | 6.7 | 7.3 | 1.97 | 13/16 | (32, 32) | 2,505,135 | 295,815 | 232,263 | 5,903 |
| hist | 16 | 1.9 | 1.9 | 1.46 | 0/16 | (32, 32) | 34,205 | 18,156 | 18,156 | 429 |
| sort_unique | 16 | 1.9 | 1.9 | 1.03 | 0/16 | (32, 32) | 81,067 | 51,483 | 51,483 | 1,024 |
| sort | 16 | 4.3 | 4.6 | 1.23 | 16/16 | (32, 32) | 1,642,655 | 334,572 | 300,940 | 6,236 |
| sort_freq | 16 | 5.8 | 6.7 | 1.59 | 15/16 | (32, 32) | 3,651,204 | 519,951 | 392,590 | 6,666 |
| frac_prevs | 16 | 4.5 | 4.4 | 1.80 | 5/16 | (32, 32) | 24,149 | 5,069 | 5,076 | 666 |
| pair_balance | 16 | 4.2 | 4.2 | 1.67 | 16/16 | (32, 32) | 45,848 | 8,815 | 8,697 | 1,240 |
| shuffle_dyck2 | 16 | 360.3 | 436.2 | 1.68 | 16/16 | (32, 32) | 893,494,508 | 132,320 | 111,173 | 684,649 |
| count_less_freq | 16 | 2.3 | 2.3 | 1.65 | 15/16 | (32, 32) | 83,367 | 41,718 | 37,590 | 826 |

x = dense params before / after. d_model is the residual width. "reuse smaller" counts configs where reuse beat trim only; reuse can lose because each erased dim costs two MLP units.

## What it means

Most of a compiled tracr model is padding: every layer is as wide as the widest one, and every residual dim exists in every layer. Removing it leaves a model that computes the same function, so the params a program really needs are far fewer than the dense count. Reuse narrows the residual stream further, at the cost of erase units.

## What it does not prove

- Sampled configs are checked on a few hundred inputs, not all of them.
- Float32 sums can run in a different order after shrinking, so numerical outputs may differ in the last bits (see the max difference).
- Tie flips are not errors of the shrink, but they are output changes. Where they occur, the original's top two scores are equal (float64 margin under 3e-15), so its own argmax is set by rounding; the original was also wrong against RASP on every such input checked. In float64, trim and reuse outputs agree to 4e-12 there.
- Shrunk models use per-layer shapes. They do not fit the uniform haiku model class, so the saving is in storage and FLOPs of this forward only.
- Dense params after shrinking still include zeros inside kept blocks. The nonzero column counts the original, including weights into MLP units whose output weights are all zero, so it is not a lower bound: shuffle_dyck2 shrinks below it.

Plot: shrink_report.png
