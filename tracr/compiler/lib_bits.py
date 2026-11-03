"""RASP programs that compare integer keys bit by bit.

One attention head computing SelectorWidth (or Aggregate) over
Select(keys, keys, LT) gives every selected key the same score, which needs at
least m - 1 query/key dims for m key values. Split by the highest differing
bit j, the comparison needs only 2 dims per bit:

  #(k < q) = sum_j #(k agrees with q on the bits above j, k_j = 0, q_j = 1)

and each term is one head over an AND of one-bit Selects (2-valued SOps). With
L-bit keys that is L heads of at most 2 * L query dims each, instead of one
head with about 2^L dims. The L counts are then summed by MLPs.

Bits are listed least significant first throughout.
"""

from typing import List, Sequence

from tracr.rasp import rasp

# How the per-level counts are summed.
# "numerical": each count is a numerical SelectorWidth (1 dim), summed by a
#   tree of LinearSequenceMaps and mapped back to categorical once.
# "categorical": each count is categorical, summed by a tree of SequenceMaps
#   (a table of (max_seq_len + 1)^2 entries each).
SUM_NUMERICAL = "numerical"
SUM_CATEGORICAL = "categorical"
SUM_MODES = (SUM_NUMERICAL, SUM_CATEGORICAL)
DEFAULT_SUM_MODE = SUM_NUMERICAL


def num_bits_for(num_values: int) -> int:
  """Bits needed to hold the integers 0 .. num_values - 1 (at least 1)."""
  if num_values < 1:
    raise ValueError(f"num_values must be positive, got {num_values}.")
  return max(1, (num_values - 1).bit_length())


def _bit(x: rasp.Value, j: int, num_bits: int) -> int:
  if isinstance(x, bool) or int(x) != x or not 0 <= x < 2**num_bits:
    raise ValueError(f"Bit split needs integers in [0, {2**num_bits}); "
                     f"got {x!r}.")
  return (int(x) >> j) & 1


def make_bits(sop: rasp.SOp, num_bits: int) -> List[rasp.SOp]:
  """Returns categorical 0/1 SOps for the bits of `sop`, least significant first.

  Values of `sop` must be integers in [0, 2^num_bits).

  Example usage:
    bits = make_bits(rasp.tokens, 3)
    [b([5, 2]) for b in bits]
    >> [[1, 0], [0, 1], [1, 0]]

  Args:
    sop: Integer-valued SOp.
    num_bits: Number of bits to extract.
  """
  if num_bits < 1:
    raise ValueError(f"num_bits must be positive, got {num_bits}.")
  return [
      rasp.categorical(
          rasp.Map(lambda x, j=j: _bit(x, j, num_bits),
                   sop)).named(f"{sop.label}_bit{j}")
      for j in range(num_bits)
  ]


def make_equal_selector(bits: Sequence[rasp.SOp]) -> rasp.Selector:
  """Selector of key/query pairs whose bits all agree (one head)."""
  if not bits:
    raise ValueError("Need at least one bit.")
  sel = rasp.Select(bits[0], bits[0], rasp.Comparison.EQ)
  for b in bits[1:]:
    sel = rasp.selector_and(
        sel, rasp.Select(b, b, rasp.Comparison.EQ), simplify=False)
  return sel


def make_level_selectors(bits: Sequence[rasp.SOp],
                         greater: bool = False) -> List[rasp.Selector]:
  """One selector per bit j: keys equal to the query above j, below it at j.

  With greater=True the key is above the query at bit j instead. The
  selectors are disjoint, and their union is Select(x, x, LT) (or GT) on the
  integer x the bits encode.

  Args:
    bits: Bit SOps of the compared integer, least significant first.
    greater: Count keys greater than the query instead of smaller.
  """
  comparison = rasp.Comparison.GT if greater else rasp.Comparison.LT
  sels = []
  for j, bit in enumerate(bits):
    sel = rasp.Select(bit, bit, comparison)
    for above in bits[j + 1:]:
      sel = rasp.selector_and(
          sel, rasp.Select(above, above, rasp.Comparison.EQ), simplify=False)
    sels.append(sel)
  return sels


def _sum_counts(selectors: Sequence[rasp.Selector], cap: int, sum_mode: str,
                name: str) -> rasp.SOp:
  """min(sum of the selectors' widths, cap) as a categorical SOp."""
  if sum_mode not in SUM_MODES:
    raise ValueError(f"sum_mode must be one of {SUM_MODES}, got {sum_mode!r}.")
  numerical = sum_mode == SUM_NUMERICAL
  terms = []
  for i, sel in enumerate(selectors):
    width = rasp.SelectorWidth(sel)
    width = rasp.numerical(width) if numerical else rasp.categorical(width)
    terms.append(width.named(f"{name}_part{i}"))
  # Balanced tree: depth ceil(log2(len(terms))) MLPs instead of len - 1.
  level = 0
  while len(terms) > 1:
    pairs = []
    for i in range(0, len(terms) - 1, 2):
      a, b = terms[i], terms[i + 1]
      if numerical:
        s = rasp.numerical(rasp.LinearSequenceMap(a, b, 1, 1))
      else:
        s = rasp.categorical(
            rasp.SequenceMap(lambda x, y: min(x + y, cap), a, b))
      pairs.append(s.named(f"{name}_sum{level}_{i // 2}"))
    if len(terms) % 2:
      pairs.append(terms[-1])
    terms = pairs
    level += 1
  total = terms[0]
  if numerical:
    return rasp.categorical(
        rasp.Map(lambda x: min(int(round(x)), cap), total)).named(name)
  # Each SequenceMap already caps; a single width is at most max_seq_len.
  return total.named(name)


def make_count_compare(keys: rasp.SOp, num_bits: int, *, max_seq_len: int,
                       greater: bool = False, or_equal: bool = False,
                       sum_mode: str = DEFAULT_SUM_MODE) -> rasp.SOp:
  """Number of positions whose key is <, <=, > or >= this position's key.

  Same values as SelectorWidth(Select(keys, keys, cmp)), capped at
  max_seq_len (never reached). Keys must be integers in [0, 2^num_bits).

  Args:
    keys: Integer-valued SOp.
    num_bits: Bits per key.
    max_seq_len: Maximum sequence length; caps the count.
    greater: Count keys greater than the query key instead of smaller.
    or_equal: Also count keys equal to the query key (one more head).
    sum_mode: SUM_NUMERICAL or SUM_CATEGORICAL, see the module constants.
  """
  bits = make_bits(keys, num_bits)
  sels = make_level_selectors(bits, greater=greater)
  if or_equal:
    sels.append(make_equal_selector(bits))
  op = (">" if greater else "<") + ("=" if or_equal else "")
  return _sum_counts(sels, max_seq_len, sum_mode, f"count{op}_{keys.label}")


def make_count_less(keys: rasp.SOp, num_bits: int, *, max_seq_len: int,
                    or_equal: bool = False,
                    sum_mode: str = DEFAULT_SUM_MODE) -> rasp.SOp:
  """SelectorWidth(Select(keys, keys, LT or LEQ)) from bit comparisons.

  Example usage:
    count_less = make_count_less(rasp.tokens, 2, max_seq_len=5)
    count_less([3, 1, 2, 1])
    >> [3, 0, 2, 0]
  """
  return make_count_compare(keys, num_bits, max_seq_len=max_seq_len,
                            or_equal=or_equal, sum_mode=sum_mode)


def make_count_greater(keys: rasp.SOp, num_bits: int, *, max_seq_len: int,
                       or_equal: bool = False,
                       sum_mode: str = DEFAULT_SUM_MODE) -> rasp.SOp:
  """SelectorWidth(Select(keys, keys, GT or GEQ)) from bit comparisons."""
  return make_count_compare(keys, num_bits, max_seq_len=max_seq_len,
                            greater=True, or_equal=or_equal,
                            sum_mode=sum_mode)


def make_sort_bits(vals: rasp.SOp, keys: rasp.SOp, *, max_seq_len: int,
                   num_bits: int,
                   sum_mode: str = DEFAULT_SUM_MODE) -> rasp.SOp:
  """Returns vals sorted by integer keys, ties in input order.

  Same output as lib.make_sort. A position's target is the number of
  (key, index) pairs below its own in lexicographic order: the index bits go
  below the key bits, so it is one bit-split count over num_bits +
  num_bits_for(max_seq_len) bits. Equal keys and earlier indices are compared
  bit by bit too.

  Example usage:
    sort = make_sort_bits(rasp.tokens, rasp.tokens, max_seq_len=5, num_bits=3)
    sort([2, 4, 1, 2])
    >> [1, 2, 2, 4]

  Args:
    vals: Values to sort.
    keys: Integer keys in [0, 2^num_bits).
    max_seq_len: Maximum sequence length; caps the target position.
    num_bits: Bits per key.
    sum_mode: SUM_NUMERICAL or SUM_CATEGORICAL, see the module constants.
  """
  index_bits = make_bits(rasp.indices, num_bits_for(max_seq_len))
  key_bits = make_bits(keys, num_bits)
  sels = make_level_selectors(index_bits + key_bits)
  # The sum is at most length - 1; sums the value sets allow but inputs never
  # reach map to max_seq_len, which matches no index (as in lib.make_sort).
  target_pos = _sum_counts(sels, max_seq_len, sum_mode, "target_pos")
  sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
  return rasp.Aggregate(sel_new, vals).named("sort")


def make_hist_bits(num_bits: int) -> rasp.SOp:
  """Number of times each token occurs; tokens compared bit by bit (one head).

  Tokens must be integers in [0, 2^num_bits).
  """
  bits = make_bits(rasp.tokens, num_bits)
  return rasp.SelectorWidth(make_equal_selector(bits)).named("hist")


def make_sort_freq_bits(max_seq_len: int, num_bits: int,
                        sum_mode: str = DEFAULT_SUM_MODE) -> rasp.SOp:
  """Tokens sorted by descending frequency, ties in input order.

  Same output as lib.make_sort_freq. The key is max_seq_len - hist, an
  integer in [0, max_seq_len].

  Args:
    max_seq_len: Maximum sequence length.
    num_bits: Bits per token; tokens must be integers in [0, 2^num_bits).
    sum_mode: SUM_NUMERICAL or SUM_CATEGORICAL, see the module constants.
  """
  hist = make_hist_bits(num_bits)
  rank = rasp.categorical(rasp.Map(lambda h: max_seq_len - h,
                                   hist)).named("freq_rank")
  return make_sort_bits(rasp.tokens, rank, max_seq_len=max_seq_len,
                        num_bits=num_bits_for(max_seq_len + 1),
                        sum_mode=sum_mode).named("sort_freq")
