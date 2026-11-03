"""Tests for rasp.fast_eval: same values and types as rasp.evaluate."""

import itertools
import random

from absl.testing import absltest
from absl.testing import parameterized
from tracr.compiler import lib
from tracr.compiler import test_cases
from tracr.rasp import fast_eval
from tracr.rasp import rasp

# Every input up to this length, when there are at most EXHAUSTIVE_LIMIT.
EXHAUSTIVE_LEN = 4
EXHAUSTIVE_LIMIT = 2000
# Random inputs per length, at lengths 1..RANDOM_MAX_LEN.
RANDOM_PER_LEN = 20
RANDOM_MAX_LEN = 12
SEED = 0
INT_VOCAB = [1, 2, 3, 4, 5]
STR_VOCAB = ["(", ")", "{", "}", "a"]
SORT_MAX_LEN = RANDOM_MAX_LEN


def _typed(x):
  """Value with its type, recursively; floats by repr to see -0.0."""
  if isinstance(x, list):
    return [_typed(v) for v in x]
  if isinstance(x, float):
    return (float, repr(x))
  return (type(x), x)


def _reference(expr, inputs):
  """rasp.evaluate per input, or the first exception type raised."""
  try:
    return [_typed(rasp.evaluate(expr, x)) for x in inputs]
  except Exception as e:  # pylint: disable=broad-except
    return type(e)


def _fast(expr, inputs):
  try:
    return [_typed(row) for row in fast_eval.evaluate_many(expr, inputs)]
  except Exception as e:  # pylint: disable=broad-except
    return type(e)


def _inputs(vocab, max_len):
  vocab = sorted(vocab, key=str)
  seqs = []
  for n in range(1, min(max_len, EXHAUSTIVE_LEN) + 1):
    if len(vocab)**n > EXHAUSTIVE_LIMIT:
      break
    seqs += [list(x) for x in itertools.product(vocab, repeat=n)]
  rng = random.Random(SEED)
  for n in range(1, max_len + 1):
    seqs += [[rng.choice(vocab) for _ in range(n)]
             for _ in range(RANDOM_PER_LEN)]
  return seqs


def _lib_programs():
  floats = rasp.numerical(rasp.Map(lambda x: x / 3, rasp.tokens))
  return [
      ("length", lib.make_length(), INT_VOCAB),
      ("reverse", lib.make_reverse(rasp.tokens), STR_VOCAB),
      ("pair_balance", lib.make_pair_balance(rasp.tokens, "(", ")"),
       STR_VOCAB),
      ("shuffle_dyck", lib.make_shuffle_dyck(["()", "{}"]), STR_VOCAB),
      ("shuffle_dyck2", lib.make_shuffle_dyck2(), STR_VOCAB),
      ("hist", lib.make_hist(), INT_VOCAB),
      ("sort_unique_ints", lib.make_sort_unique(rasp.tokens, rasp.tokens),
       INT_VOCAB),
      # Duplicate string keys: rasp._mean raises on both sides.
      ("sort_unique_strs", lib.make_sort_unique(rasp.tokens, rasp.tokens),
       STR_VOCAB),
      ("sort", lib.make_sort(rasp.tokens, rasp.tokens,
                             max_seq_len=SORT_MAX_LEN, min_key=1), INT_VOCAB),
      ("sort_freq", lib.make_sort_freq(max_seq_len=SORT_MAX_LEN), INT_VOCAB),
      ("frac_prevs", lib.make_frac_prevs(rasp.tokens == "("), STR_VOCAB),
      ("shift_by", lib.shift_by(2, rasp.tokens), STR_VOCAB),
      ("detect_pattern", lib.detect_pattern(rasp.tokens, "a("), STR_VOCAB),
      ("count_less_freq", lib.make_count_less_freq(n=2), INT_VOCAB),
      ("count", lib.make_count(rasp.tokens, "a"), STR_VOCAB),
      ("nary_sequencemap", lib.make_nary_sequencemap(
          lambda x, y, z: x * y - z, rasp.tokens, rasp.indices, rasp.tokens),
       INT_VOCAB),
      # Means over floats go through rasp._mean (compensated float sum).
      ("float_mean", rasp.numerical(rasp.Aggregate(
          rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ),
          floats, default=0)), INT_VOCAB),
      ("selector_or_not_width", rasp.SelectorWidth(rasp.selector_not(
          rasp.selector_or(
              rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.GT),
              rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.NEQ),
              simplify=False), simplify=False)), INT_VOCAB),
      ("full_linear", rasp.LinearSequenceMap(
          rasp.Full(0.5), rasp.Map(float, rasp.tokens), 2.0, -0.25),
       INT_VOCAB),
      # A Select over a None-padded SOp: Comparison raises on None keys.
      ("select_on_none", rasp.SelectorWidth(rasp.Select(
          lib.shift_by(1, rasp.tokens), rasp.tokens, rasp.Comparison.EQ)),
       STR_VOCAB),
      ("selector", rasp.Select(rasp.tokens, rasp.indices,
                               lambda k, q: k > q), INT_VOCAB),
  ]


_CASES = [
    dict(testcase_name=c["testcase_name"] + f"_{group}", program=c["program"],
         vocab=c["vocab"], max_seq_len=c["max_seq_len"])
    for group, cases in (("main", test_cases.TEST_CASES),
                         ("causal", test_cases.CAUSAL_TEST_CASES),
                         ("unsupported", test_cases.UNSUPPORTED_TEST_CASES))
    for c in cases
]


class FastEvalTest(parameterized.TestCase):

  @parameterized.named_parameters(*_CASES)
  def test_test_cases_match_evaluate(self, program, vocab, max_seq_len):
    seqs = _inputs(vocab, max_seq_len)
    self.assertEqual(_fast(program, seqs), _reference(program, seqs))

  @parameterized.named_parameters(
      *[dict(testcase_name=n, program=p, vocab=v)
        for n, p, v in _lib_programs()])
  def test_lib_programs_match_evaluate(self, program, vocab):
    seqs = _inputs(vocab, RANDOM_MAX_LEN)
    self.assertEqual(_fast(program, seqs), _reference(program, seqs))

  def test_constants_and_mixed_types(self):
    # 1, 1.0 and True are equal in Python but must keep their own types.
    const = rasp.ConstantSOp([1, 1.0, True, None])
    program = rasp.Aggregate(
        rasp.ConstantSelector([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                               [0, 0, 0, 1]]), rasp.Map(lambda x: x, const))
    seqs = [["a", "b", "c", "d"], ["d", "c", "b", "a"]]
    self.assertEqual(_fast(program, seqs), _reference(program, seqs))
    width = rasp.SelectorWidth(rasp.ConstantSelector([[1, 1], [0, 1]]))
    self.assertEqual(_fast(width, [[0, 1]]), _reference(width, [[0, 1]]))

  def test_mean_of_ints_and_bools(self):
    bools = rasp.numerical(rasp.Map(lambda x: x > 2, rasp.tokens))
    mixed = rasp.SequenceMap(lambda b, t: t if t % 2 else b, bools,
                             rasp.tokens)
    for sop in (bools, mixed):
      program = rasp.Aggregate(
          rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ), sop,
          default=0)
      seqs = _inputs(INT_VOCAB, RANDOM_MAX_LEN)
      self.assertEqual(_fast(program, seqs), _reference(program, seqs))

  def test_unchecked_constant_truncates_like_zip(self):
    short = rasp.ConstantSOp([1, 2], check_length=False)
    program = rasp.SequenceMap(lambda x, y: x + y, short, rasp.indices)
    seqs = [[0, 0, 0], [1, 1, 1]]
    self.assertEqual(_fast(program, seqs), _reference(program, seqs))

  def test_comparison_on_types_that_never_meet(self):
    # 1 < "a" raises, but no input holds both, so rasp.evaluate does not.
    program = rasp.SelectorWidth(
        rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LT))
    seqs = [[1, 2, 2], ["a", "b", "a"]]
    self.assertEqual(_fast(program, seqs), _reference(program, seqs))
    self.assertIsInstance(_reference(program, [[1, "a"]]), type)
    self.assertEqual(_fast(program, [[1, "a"]]),
                     _reference(program, [[1, "a"]]))

  def test_raises_like_evaluate(self):
    with self.assertRaises(ValueError):
      fast_eval.evaluate_batch(rasp.ConstantSOp([1, 2]), [[0, 0, 0]])
    with self.assertRaises(ValueError):
      fast_eval.evaluate_batch(rasp.tokens, [[0], [0, 0]])

    class Unknown(rasp.SOp):

      @property
      def children(self):
        return []

    with self.assertRaises(NotImplementedError):
      fast_eval.evaluate_batch(Unknown(), [[0]])

  def test_empty_inputs(self):
    self.assertEqual(fast_eval.evaluate_many(lib.make_hist(), []), [])
    self.assertEqual(fast_eval.evaluate_many(lib.make_hist(), [[], [1]]),
                     [[], [1]])


if __name__ == "__main__":
  absltest.main()
