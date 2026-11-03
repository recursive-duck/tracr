# Bit arithmetic vs lookup table

## What was tested

a + b for a = tokens (vocab 0..2^w-1) and b = indices (max_seq_len 2^w), compiled as one lookup-table SequenceMap and as bit circuits from tracr/compiler/lib_arith.py, for w = 2..10. Every bit program reads its bits with to_bits Maps (encode) and returns one categorical value through from_bits (decode). Each compiled model ran on 200 random full-length inputs (seed 0 + w; fewer when they would exceed 105,000 tokens, see seqs) against rasp.evaluate and integer a + b.

Terms: units = MLP hidden units (one ReLU each) from the value sets; circuit = units of the adder alone (or of the table); layers = attention+MLP layer pairs; d_trim / p_trim = residual width and dense params after shrinking.shrink(reuse=False).

## Key numbers

- Configs: 42 compiled and checked, 1 not: lookup w=8: timeout > 300 s
- Mismatching sequences vs RASP + vs integers: 0 of 16,016 comparisons.
- Largest lookup that compiled: w=7, 16,384 units, 16,909,824 trimmed params, 156.9 s.
  - ripple_kpg at w=7: units 16,384 -> 436 (**37.6x**), circuit 16,384 -> 104, params_trim 16,909,824 -> 718,185 (**23.5x**), layers 1 -> 11, compile 156.9 -> 4.2 s.
  - ripple_pair at w=7: units 16,384 -> 460 (**35.6x**), circuit 16,384 -> 128, params_trim 16,909,824 -> 755,739 (**22.4x**), layers 1 -> 11, compile 156.9 -> 6.8 s.
  - ripple_bits at w=7: units 16,384 -> 460 (**35.6x**), circuit 16,384 -> 128, params_trim 16,909,824 -> 769,887 (**22.0x**), layers 1 -> 17, compile 156.9 -> 4.8 s.
  - kogge_stone at w=7: units 16,384 -> 508 (**32.3x**), circuit 16,384 -> 176, params_trim 16,909,824 -> 840,225 (**20.1x**), layers 1 -> 9, compile 156.9 -> 4.1 s.
- ripple_kpg at w=10: 2,376 units (circuit 152, decode 2,184, encode 40), 15 layers, d_trim 4,340, p_trim 29,525,020, compile 79.3 s.
- ripple_pair at w=10: 2,412 units (circuit 188, decode 2,184, encode 40), 15 layers, d_trim 4,349, p_trim 29,899,375, compile 82.5 s.
- ripple_bits at w=10: 2,412 units (circuit 188, decode 2,184, encode 40), 24 layers, d_trim 4,367, p_trim 30,023,125, compile 163.2 s.
- kogge_stone at w=10: 2,520 units (circuit 296, decode 2,184, encode 40), 11 layers, d_trim 4,388, p_trim 31,115,308, compile 76.9 s.

## Growth fits (least squares over the compiled widths)

| method | widths | units log2-slope | circuit log2-slope | circuit units ~ a*x + b | params_trim log2-slope | compile_s log2-slope | layers |
|---|---|---|---|---|---|---|---|
| lookup | 2..7 | 2.00 | 2.00 | 2705.8*w + -8536.2 (max resid 5979.4) | 2.92 | 1.02 | [1, 1, 1, 1, 1, 1] |
| ripple_kpg | 2..10 | 0.70 | 0.31 | 16.0*w + -8.0 (max resid 0.0) | 1.58 | 0.48 | [5, 6, 8, 9, 10, 11, 13, 14, 15] |
| ripple_pair | 2..10 | 0.68 | 0.32 | 20.0*w + -12.0 (max resid 0.0) | 1.56 | 0.56 | [5, 6, 8, 9, 10, 11, 13, 14, 15] |
| ripple_bits | 2..10 | 0.68 | 0.32 | 20.0*w + -12.0 (max resid 0.0) | 1.55 | 0.59 | [6, 8, 11, 13, 15, 17, 20, 22, 24] |
| kogge_stone | 2..10 | 0.70 | 0.42 | 8.6*w*log2(w) + 6.6 (max resid 4.0) | 1.58 | 0.49 | [5, 6, 8, 8, 9, 9, 10, 10, 11] |

A log2-slope s means the quantity grows like 2^(s w): 2 is the 4^w of a two-operand table, 1 the 2^w of a w-bit output.

Plot: experiments/bit_arith/bit_arith.png. Total wall time 1407 s.
