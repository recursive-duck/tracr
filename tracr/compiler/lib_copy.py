"""Exact copy of multi-token IDs: a call repeats an earlier ID verbatim.

Layout (all tokens are ints; see CopyLayout):
  digits        0 .. base - 1          the tokens an ID is made of
  SRC_s         base + 2 s             "the s-th ID of a tool result follows"
  CALL_s        base + 2 s + 1         "copy the ID of slot s here"
  filler        base + 2 num_slots     any other token, e.g. the placeholder

  ... SRC_s d_1 .. d_k ... CALL_s c_1 .. c_k ...

A source ID is SRC_s followed by k = id_len digits. The k positions after
CALL_s are copy positions: they may hold anything (the filler as a
placeholder, or the digits so far under teacher forcing). At copy position i
the program outputs d_i of the slot-s source. Everywhere else its RASP value
is None (undefined).

Rules a sequence must follow (check_layout): two markers are at least k + 1
apart, each source is followed by k digits, a slot has at most one source,
and every called slot has a source. Sources may sit anywhere, also after the
call (the compiled model is not causal).

How it compiles:
  1. Each position finds the marker 1..k positions before it, if any: one
     attention head per bit of the marker's code (1 + 2 s + kind, 0 = none)
     and per bit of the marker's index mod (k + 1). Each head is an AND of a
     window Select on indices and a Select on marker tokens, and averages a
     constant 1, so it outputs exactly 0 or 1 (default 0, never None). The
     bits are decoded with lib_arith.from_bits.
  2. offset = (index - marker index) mod (k + 1), from the two residues.
  3. key   = address(slot, offset) at source digits, NO_KEY elsewhere;
     query = address(slot, offset) at copy positions, NO_QUERY elsewhere.
     Both use the binary encoding, so the copy head matches by exact
     equality of ceil(log2(num_slots * (k + 1) + 2)) bit codes.
  4. output = Aggregate(Select(key, query, EQ), tokens).

Attention layers: 2 (marker bits, copy). The window Select reads one-hot
indices, so its query/key width grows with max_seq_len.
"""

import dataclasses
import random
from typing import Callable, Dict, List, Optional, Sequence

from tracr.compiler import lib_arith
from tracr.compiler import lib_bits
from tracr.rasp import rasp

SRC = 0
CALL = 1
NO_KEY = -1
NO_QUERY = -2
NUM_KINDS = 2


@dataclasses.dataclass(frozen=True)
class CopyLayout:
  """Token codes of the copy layout.

  Attributes:
    id_len: k, digits per ID.
    num_slots: number of distinct ID slots s = 0 .. num_slots - 1.
    base: number of digit tokens 0 .. base - 1.
  """
  id_len: int
  num_slots: int
  base: int

  def __post_init__(self):
    for name in ("id_len", "num_slots", "base"):
      if getattr(self, name) < 1:
        raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")

  @property
  def period(self) -> int:
    return self.id_len + 1

  def src(self, slot: int) -> int:
    return self.base + NUM_KINDS * slot + SRC

  def call(self, slot: int) -> int:
    return self.base + NUM_KINDS * slot + CALL

  @property
  def filler(self) -> int:
    return self.base + NUM_KINDS * self.num_slots

  @property
  def vocab(self) -> set:
    return set(range(self.filler + 1))

  def is_marker(self, token: int) -> bool:
    return self.base <= token < self.filler

  def marker_code(self, token: int) -> int:
    """1 + NUM_KINDS * slot + kind for a marker, 0 for any other token."""
    return token - self.base + 1 if self.is_marker(token) else 0

  @property
  def code_bits(self) -> int:
    return lib_bits.num_bits_for(NUM_KINDS * self.num_slots + 1)

  @property
  def residue_bits(self) -> int:
    return lib_bits.num_bits_for(self.period)

  def address(self, code: int, offset: int, kind: int,
              missing: int) -> int:
    """slot * period + offset if `code` is a marker of `kind`, else missing."""
    if code < 1 or code > NUM_KINDS * self.num_slots or offset < 1:
      return missing
    slot, marker_kind = divmod(code - 1, NUM_KINDS)
    if marker_kind != kind:
      return missing
    return slot * self.period + offset


def _bit(x: int, j: int) -> int:
  return (x >> j) & 1


def _marker_bit(layout: CopyLayout, ones: rasp.SOp,
                index_pred: Callable[[int], bool],
                token_pred: Callable[[int], bool], name: str) -> rasp.SOp:
  """1 if the marker 1..k positions back passes both predicates, else 0.

  One head: window AND marker on the key side. Markers are at least k + 1
  apart, so at most one key is selected and the mean of `ones` is 0 or 1.
  """
  k = layout.id_len
  window = rasp.Select(
      rasp.indices, rasp.indices,
      lambda key, query: 0 < query - key <= k and index_pred(key))
  marker = rasp.Select(
      rasp.tokens, rasp.tokens,
      lambda key, query: layout.is_marker(key) and token_pred(key))
  sel = rasp.selector_and(window, marker, simplify=False)
  mean = rasp.numerical(rasp.Aggregate(sel, ones, default=0))
  return rasp.categorical(rasp.Map(lambda x: int(x > 0.5), mean.named(
      f"{name}_mean"))).named(name)


def make_marker_lookup(layout: CopyLayout) -> Dict[str, rasp.SOp]:
  """S-Ops `code` (marker code, 0 = none) and `offset` per position."""
  ones = rasp.numerical(rasp.Map(lambda _: 1, rasp.tokens)).named("ones")
  everything = lambda _: True
  code_bits = [
      _marker_bit(layout, ones, everything,
                  lambda t, j=j: _bit(layout.marker_code(t), j) == 1,
                  f"code_bit{j}") for j in range(layout.code_bits)
  ]
  residue_bits = [
      _marker_bit(layout, ones,
                  lambda i, j=j: _bit(i % layout.period, j) == 1,
                  everything, f"residue_bit{j}")
      for j in range(layout.residue_bits)
  ]
  code = lib_arith.from_bits(code_bits).named("marker_code")
  residue = lib_arith.from_bits(residue_bits).named("marker_residue")
  own_residue = rasp.categorical(
      rasp.Map(lambda i: i % layout.period, rasp.indices)).named("own_residue")
  offset = rasp.categorical(rasp.SequenceMap(
      lambda p, m: (p - m) % layout.period, own_residue,
      residue)).named("offset")
  return dict(code=code, offset=offset, code_bits=code_bits,
              residue_bits=residue_bits)


def make_exact_copy(
    layout: CopyLayout,
    encoding: Callable[[rasp.SOp], rasp.SOp] = rasp.binary) -> rasp.SOp:
  """Token of the called source ID at each copy position (None elsewhere).

  Example usage:
    layout = CopyLayout(id_len=2, num_slots=2, base=10)
    f = layout.filler
    seq = [f, layout.src(1), 4, 2, layout.src(0), 7, 7,
           layout.call(1), f, f]
    make_exact_copy(layout)(seq)
    >> [None] * 8 + [4, 2]

  Args:
    layout: Token codes and ID length.
    encoding: rasp.binary (default) or rasp.categorical for key and query.
  """
  found = make_marker_lookup(layout)
  key = encoding(rasp.SequenceMap(
      lambda c, o: layout.address(c, o, SRC, NO_KEY), found["code"],
      found["offset"])).named("copy_key")
  query = encoding(rasp.SequenceMap(
      lambda c, o: layout.address(c, o, CALL, NO_QUERY), found["code"],
      found["offset"])).named("copy_query")
  return rasp.categorical(rasp.Aggregate(
      rasp.Select(key, query, rasp.Comparison.EQ),
      rasp.tokens)).named("exact_copy")


def check_layout(layout: CopyLayout, seq: Sequence[int]) -> None:
  """Raises ValueError if `seq` breaks the layout rules (module docstring)."""
  k = layout.id_len
  if any(not 0 <= t <= layout.filler for t in seq):
    raise ValueError(f"Token outside 0..{layout.filler}: {list(seq)}")
  markers = [p for p, t in enumerate(seq) if layout.is_marker(t)]
  for a, b in zip(markers, markers[1:]):
    if b - a <= k:
      raise ValueError(f"Markers at {a} and {b} are closer than {k + 1}.")
  sources = set()
  calls = set()
  for p in markers:
    slot, kind = divmod(seq[p] - layout.base, NUM_KINDS)
    if kind == CALL:
      calls.add(slot)
      continue
    if slot in sources:
      raise ValueError(f"Slot {slot} has two sources.")
    sources.add(slot)
    digits = seq[p + 1:p + 1 + k]
    if len(digits) != k or any(d >= layout.base for d in digits):
      raise ValueError(f"Source at {p} is not followed by {k} digits.")
  if calls - sources:
    raise ValueError(f"Called slots without a source: {sorted(calls - sources)}")


def reference_copy(layout: CopyLayout,
                   seq: Sequence[int]) -> List[Optional[int]]:
  """Pure-Python exact copy: source digits at copy positions, else None."""
  check_layout(layout, seq)
  start = {}
  for p, t in enumerate(seq):
    if layout.is_marker(t) and (t - layout.base) % NUM_KINDS == SRC:
      start[(t - layout.base) // NUM_KINDS] = p
  out: List[Optional[int]] = [None] * len(seq)
  for p, t in enumerate(seq):
    if layout.is_marker(t) and (t - layout.base) % NUM_KINDS == CALL:
      src = start[(t - layout.base) // NUM_KINDS]
      for i in range(1, min(layout.id_len, len(seq) - 1 - p) + 1):
        out[p + i] = seq[src + i]
  return out


def make_sequence(layout: CopyLayout, ids: Dict[int, Sequence[int]],
                  calls: Sequence[int], gap: Sequence[int] = (),
                  copy_fill: Optional[Sequence[int]] = None) -> List[int]:
  """filler, sources in `ids` order, then one CALL block per called slot.

  Args:
    layout: Token codes.
    ids: slot -> its k digits, in the order the sources appear.
    calls: slots to call, in order.
    gap: tokens placed between consecutive blocks (no markers).
    copy_fill: k tokens for each call's copy positions (default: filler).
  """
  fill = list(copy_fill) if copy_fill is not None else [layout.filler
                                                        ] * layout.id_len
  seq = [layout.filler]
  blocks = [[layout.src(s)] + list(d) for s, d in ids.items()]
  blocks += [[layout.call(s)] + fill for s in calls]
  for i, block in enumerate(blocks):
    if i:
      seq += list(gap)
    seq += block
  return seq


def max_sequence_len(layout: CopyLayout, max_calls: int, max_gap: int) -> int:
  """Longest make_sequence output: all slots as sources, max_calls calls."""
  blocks = layout.num_slots + max_calls
  return 1 + blocks * layout.period + (blocks - 1) * max_gap


def random_sequence(layout: CopyLayout, rng: random.Random, max_calls: int,
                    max_gap: int) -> List[int]:
  """A random valid sequence for make_sequence.

  1..num_slots sources in random slot order with random digits, 1..max_calls
  calls of those slots, a gap of 0..max_gap random digits/filler between
  blocks, and random non-marker tokens at the copy positions (as under
  teacher forcing, or garbage).
  """
  k = layout.id_len
  n = rng.randint(1, layout.num_slots)
  order = rng.sample(range(layout.num_slots), n)
  ids = {s: [rng.randrange(layout.base) for _ in range(k)] for s in order}
  calls = [rng.choice(order) for _ in range(rng.randint(1, max_calls))]
  plain = list(range(layout.base)) + [layout.filler]
  gap = [rng.choice(plain) for _ in range(rng.randint(0, max_gap))]
  fill = [rng.choice(plain) for _ in range(k)]
  return make_sequence(layout, ids, calls, gap=gap, copy_fill=fill)
