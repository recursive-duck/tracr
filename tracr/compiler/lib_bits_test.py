"""Tests for compiler.lib_bits."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.compiler import lib_bits
from tracr.rasp import rasp

# Every sequence over VOCAB_SIZE tokens of length 1..MAX_SEQ_LEN: 1,364.
VOCAB_SIZE = 4
MAX_SEQ_LEN = 5
NUM_BITS = lib_bits.num_bits_for(VOCAB_SIZE)
COMPILER_BOS = "BOS"
# Wider keys than the vocab needs (unused high bits) must not change results.
EXTRA_BITS = 1


def _all_inputs():
  return [
      list(x)
      for n in range(1, MAX_SEQ_LEN + 1)
      for x in itertools.product(range(VOCAB_SIZE), repeat=n)
  ]


ALL_INPUTS = _all_inputs()


def _width(comparison):
  return rasp.SelectorWidth(rasp.Select(rasp.tokens, rasp.tokens, comparison))


def _bits_and_reference(name, sum_mode, num_bits=NUM_BITS):
  """(bit-split program, reference program) for a test case name."""
  cmp = rasp.Comparison
  kw = dict(max_seq_len=MAX_SEQ_LEN, sum_mode=sum_mode)
  cases = {
      "less": lambda: (lib_bits.make_count_less(rasp.tokens, num_bits, **kw),
                       _width(cmp.LT)),
      "less_equal": lambda: (lib_bits.make_count_less(
          rasp.tokens, num_bits, or_equal=True, **kw), _width(cmp.LEQ)),
      "greater": lambda: (lib_bits.make_count_greater(
          rasp.tokens, num_bits, **kw), _width(cmp.GT)),
      "greater_equal": lambda: (lib_bits.make_count_greater(
          rasp.tokens, num_bits, or_equal=True, **kw), _width(cmp.GEQ)),
      "sort": lambda: (lib_bits.make_sort_bits(
          rasp.tokens, rasp.tokens, num_bits=num_bits, **kw),
                       lib.make_sort(rasp.tokens, rasp.tokens,
                                     max_seq_len=MAX_SEQ_LEN, min_key=1)),
      "sort_freq": lambda: (lib_bits.make_sort_freq_bits(
          MAX_SEQ_LEN, num_bits, sum_mode=sum_mode),
                            lib.make_sort_freq(MAX_SEQ_LEN)),
  }
  return cases[name]()


CASES = [
    dict(testcase_name=f"{name}_{mode}", name=name, sum_mode=mode)
    for name in ("less", "less_equal", "greater", "greater_equal", "sort",
                 "sort_freq")
    for mode in lib_bits.SUM_MODES
]


def _run_compiled(model, inputs):
  """Decoded outputs (without BOS) of a compiled model, batched by length."""
  forward = jax.jit(model.forward)
  enc = model.input_encoder
  by_len = {}
  for i, x in enumerate(inputs):
    by_len.setdefault(len(x), []).append(i)
  outputs = [None] * len(inputs)
  for idx in by_len.values():
    ids = np.array([enc.encode([COMPILER_BOS] + inputs[i]) for i in idx])
    out = np.asarray(forward(model.params, ids).unembedded_output)
    for row, i in enumerate(idx):
      outputs[i] = model.output_encoder.decode(out[row].tolist())[1:]
  return outputs


class LibBitsTest(parameterized.TestCase):

  def test_bits_are_lsb_first(self):
    bits = lib_bits.make_bits(rasp.tokens, 3)
    self.assertEqual([rasp.evaluate(b, [5, 2]) for b in bits],
                     [[1, 0], [0, 1], [1, 0]])

  def test_bits_reject_out_of_range(self):
    bits = lib_bits.make_bits(rasp.tokens, 2)
    with self.assertRaises(ValueError):
      rasp.evaluate(bits[0], [4])

  @parameterized.parameters((1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (256, 8))
  def test_num_bits_for(self, num_values, expected):
    self.assertEqual(lib_bits.num_bits_for(num_values), expected)

  @parameterized.named_parameters(*CASES)
  def test_rasp_matches_reference_on_all_inputs(self, name, sum_mode):
    for num_bits in (NUM_BITS, NUM_BITS + EXTRA_BITS):
      prog, ref = _bits_and_reference(name, sum_mode, num_bits)
      for x in ALL_INPUTS:
        self.assertEqual(rasp.evaluate(prog, x), rasp.evaluate(ref, x), x)

  @parameterized.named_parameters(*CASES)
  def test_compiled_matches_rasp_on_all_inputs(self, name, sum_mode):
    prog, _ = _bits_and_reference(name, sum_mode)
    model = compiling.compile_rasp_to_model(
        prog, vocab=set(range(VOCAB_SIZE)), max_seq_len=MAX_SEQ_LEN,
        compiler_bos=COMPILER_BOS)
    got = _run_compiled(model, ALL_INPUTS)
    for x, out in zip(ALL_INPUTS, got):
      self.assertEqual(out, rasp.evaluate(prog, x), x)


if __name__ == "__main__":
  absltest.main()
