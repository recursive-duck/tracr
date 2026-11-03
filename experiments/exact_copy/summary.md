# Exact copy of multi-token IDs

## What was tested

tracr/compiler/lib_copy.make_exact_copy: at each of the k positions after CALL_s the compiled model must output the matching digit of the source ID SRC_s d_1..d_k earlier in the sequence. Layout: digits 0..base-1, SRC_s = base + 2s, CALL_s = base + 2s + 1, filler = base + 2S; markers at least k + 1 apart. Head 1 (one attention layer) finds the marker 1..k positions back: one head per bit of its code and of its index mod (k + 1), each averaging a constant 1 over window AND marker (exactly 0 or 1, never None), decoded with lib_arith.from_bits. Head 2 copies tokens by exact equality of the bit-coded (rasp.binary) address slot * (k + 1) + offset.

Sweeps: k over [1, 2, 4, 8, 16, 32] at S = 4; S over [1, 2, 4, 8, 16, 32] at k = 8; base 10. Each config: 64 random valid sequences (1..S sources in random slot order, 1..2 calls, gaps of 0..2 random digits/filler, random tokens at copy positions) vs the pure-Python reference; the first 4 also vs rasp.evaluate. Baseline: the same program with one-hot (categorical) copy key/query.

Terms: units = MLP hidden units with nonzero input weights; heads = attention heads with nonzero query weights; qk_dims = residual dims of the copy key + query; d_trim / p_trim = residual width and dense params after shrinking.shrink(reuse=False); N = max_seq_len.

## Key numbers

- Configs: 22 compiled and checked, 0 failed.
- Wrong copied tokens (compiled vs reference): 0 of 19,502 over 1,408 sequences. RASP vs reference: 0 of 88 sequences.
- Binary, k 1 -> 32 (N 23 -> 209): layers 6 -> 8, heads 6 -> 11, units 110 -> 3,344, qk_dims 6 -> 16, p_trim 64,523 -> 4,646,622, compile 2.2 -> 8.4 s.
  - categorical -> binary key/query at k=32, S=4: qk_dims 258 -> 16 (**16.1x**), d_trim 704 -> 474, units 3,344 -> 3,344, p_trim 5,841,088 -> 4,646,622 (1.3x), compile 10.4 -> 8.4 s.
- Binary, S 1 -> 32 (N 32 -> 373): layers 7 -> 7, heads 7 -> 12, units 272 -> 2,679, qk_dims 8 -> 18, p_trim 117,306 -> 10,334,817, compile 2.4 -> 7.2 s.
  - categorical -> binary key/query at k=8, S=32: qk_dims 514 -> 18 (**28.6x**), d_trim 1,204 -> 783, units 2,679 -> 2,679, p_trim 11,959,332 -> 10,334,817 (1.2x), compile 11.4 -> 7.2 s.

## Growth (log-log least squares: y ~ x^slope)

| sweep / key-query | x | layers | heads | units | qk_dims | d_trim | p_trim | compile_s |
|---|---|---|---|---|---|---|---|---|
| k/binary | 1..32 | [6, 6, 7, 7, 8, 8] | [6, 7, 8, 9, 10, 11] | 0.98 | 0.28 | 0.41 | 1.23 | 0.38 |
| k/categorical | 1..32 | [6, 6, 7, 7, 8, 8] | [6, 7, 8, 9, 10, 11] | 0.98 | 0.94 | 0.55 | 1.39 | 0.40 |
| S/binary | 1..32 | [7, 7, 7, 7, 7, 7] | [7, 8, 9, 10, 11, 12] | 0.66 | 0.23 | 0.51 | 1.30 | 0.26 |
| S/categorical | 1..32 | [7, 7, 7, 7, 7, 7] | [7, 8, 9, 10, 11, 12] | 0.66 | 0.97 | 0.63 | 1.38 | 0.54 |

## What the numbers mean

The copy itself is exact by construction: the key and query are the same integer address only when slot and offset agree, and the binary head scores a full match L above every partial match. Zero wrong tokens over all configs says the soft attention and the 0/1 marker heads stayed exact at these sizes. Bit-coded key/query need ceil(log2(S (k + 1) + 2)) dims each instead of one per address.

## What this does NOT prove

- Nothing about long contexts: max_seq_len here is at most 373. The marker window Select reads one-hot indices, so its width and d_model grow linearly with max_seq_len; 1M context needs a different window (e.g. relative/bit positions).
- Nothing about a trained model or natural text: inputs follow the strict layout (markers k + 1 apart, one source per slot, every called slot present). Broken layouts are rejected by check_layout, not handled.
- Not an autoregressive decoder: the compiled model is not causal and reads all copy positions at once; teacher forcing is only simulated by random tokens at copy positions.
- Random checks, not exhaustive, at these sizes (exhaustive k <= 3, S <= 3 base 2 is in lib_copy_test).

Plot: experiments/exact_copy/exact_copy.png. Total wall time 109 s.
