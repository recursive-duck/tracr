# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary arithmetic on lists of 2-valued categorical S-Ops.

A number is a list of bit S-Ops, least significant bit first; each bit takes
values in {0, 1}. Every operation here is built only from Maps and
SequenceMaps whose inputs have 2 or 3 values, so each compiles to an MLP with
at most 9 hidden units. A SequenceMap on two whole numbers 0..2^w-1 instead
compiles to a 4^w-unit lookup table.

Carry chains use a 3-valued status per bit: KILL (0) and GEN (1) decide the
carry out of a span of bits, PROP (2) passes the carry in through. Composing
spans is `hi if hi != PROP else lo`. A span that starts at bit 0 has carry in
0, so its status is never PROP and its value set is {0, 1}: it *is* the carry
bit. Basis inference finds this on its own, so such spans compile to
2-dimensional subspaces and the MLPs combining them shrink from 9 to 6 units.

Networks (how a chain of n statuses is combined):
  RIPPLE: one combine per bit in a chain, n - 1 layers.
  KOGGE_STONE: parallel prefix in ceil(log2 n) layers and about n log2 n
    combines; for a reduction (one total, as in less_than) a balanced tree
    in ceil(log2 n) layers and n - 1 combines.

Ripple-adder fusions (how a full adder is split into 2-input tables):
  BITS: only bit x bit tables (4 units each), 2 layers per bit.
  PAIR: (a_i, b_i) packed into one 4-valued S-Op, then carry and sum are
    8-unit tables of (pair, carry), 1 layer per bit.
  KPG: (a_i, b_i) packed into the 3-valued status, then carry and sum are
    6-unit tables of (status, carry), 1 layer per bit.
"""

from typing import Callable, List, Sequence

from tracr.rasp import rasp

KILL, GEN, PROP = 0, 1, 2

RIPPLE = "ripple"
KOGGE_STONE = "kogge_stone"
NETWORKS = (RIPPLE, KOGGE_STONE)

BITS = "bits"
PAIR = "pair"
KPG = "kpg"
FUSIONS = (BITS, PAIR, KPG)

Bits = List[rasp.SOp]


def _seqmap(f: Callable[[int, int], int], fst: rasp.SOp,
            snd: rasp.SOp) -> rasp.SOp:
  return rasp.categorical(rasp.SequenceMap(f, fst, snd))


def _bit_map(f: Callable[[int], int], bit: rasp.SOp) -> rasp.SOp:
  # simplify=False keeps the Map on the bit; merged into a to_bits Map it
  # would read the whole number instead.
  return rasp.categorical(rasp.Map(f, bit, simplify=False))


def _and(x: int, y: int) -> int:
  return x & y


def _or(x: int, y: int) -> int:
  return x | y


def _xor(x: int, y: int) -> int:
  return x ^ y


def _combine(hi: int, lo: int) -> int:
  """Status of two adjacent spans: `hi` decides unless it propagates."""
  return lo if hi == PROP else hi


def _add_status(a: int, b: int) -> int:
  """Carry status of one bit position of a + b."""
  if a and b:
    return GEN
  return PROP if a != b else KILL


def _less_status(a: int, b: int) -> int:
  """GEN if a < b at this bit, KILL if a > b, PROP if equal."""
  if a == b:
    return PROP
  return GEN if a < b else KILL


def _check_network(network: str) -> None:
  if network not in NETWORKS:
    raise ValueError(f"network must be one of {NETWORKS}, got {network!r}")


def _check_widths(a_bits: Sequence[rasp.SOp],
                  b_bits: Sequence[rasp.SOp]) -> None:
  if not a_bits or len(a_bits) != len(b_bits):
    raise ValueError("Operands need the same nonzero width, got "
                     f"{len(a_bits)} and {len(b_bits)}.")


def scan(xs: Sequence[rasp.SOp], op: Callable[[int, int], int],
         network: str = RIPPLE) -> Bits:
  """Inclusive prefixes y_i = x_i op x_(i-1) op ... op x_0 of an associative op.

  `op(hi, lo)` takes the more significant operand first.

  Args:
    xs: S-Ops, least significant first.
    op: associative function of (hi, lo) values.
    network: RIPPLE (n - 1 layers) or KOGGE_STONE (ceil(log2 n) layers).

  Returns:
    The n prefix S-Ops.
  """
  _check_network(network)
  ys = list(xs)
  if network == RIPPLE:
    for i in range(1, len(ys)):
      ys[i] = _seqmap(op, ys[i], ys[i - 1])
    return ys
  span = 1
  while span < len(ys):
    ys = [y if i < span else _seqmap(op, y, ys[i - span])
          for i, y in enumerate(ys)]
    span *= 2
  return ys


def reduce(xs: Sequence[rasp.SOp], op: Callable[[int, int], int],
           network: str = RIPPLE) -> rasp.SOp:
  """x_(n-1) op ... op x_0 as a chain (RIPPLE) or balanced tree (KOGGE_STONE)."""
  _check_network(network)
  if not xs:
    raise ValueError("Cannot reduce an empty list.")
  if network == RIPPLE:
    return scan(xs, op, RIPPLE)[-1]
  level = list(xs)
  while len(level) > 1:
    pairs = [_seqmap(op, level[i + 1], level[i])
             for i in range(0, len(level) - 1, 2)]
    level = pairs + level[len(pairs) * 2:]
  return level[0]


def _add_kpg(a_bits: Sequence[rasp.SOp], b_bits: Sequence[rasp.SOp],
             network: str) -> Bits:
  """Adder on carry statuses; bit 0 has no carry in, so its status is a&b."""
  status = [_seqmap(_and, a_bits[0], b_bits[0])]
  status += [_seqmap(_add_status, a, b) for a, b in zip(a_bits[1:], b_bits[1:])]
  carries = scan(status, _combine, network)
  sums = [_seqmap(_xor, a_bits[0], b_bits[0])]
  sums += [_seqmap(lambda s, c: int((s == PROP) != bool(c)), s, c)
           for s, c in zip(status[1:], carries[:-1])]
  return sums + [carries[-1]]


def _add_pair(a_bits: Sequence[rasp.SOp], b_bits: Sequence[rasp.SOp]) -> Bits:
  """Ripple adder with one 8-row table per carry and per sum bit."""
  carry = _seqmap(_and, a_bits[0], b_bits[0])
  sums = [_seqmap(_xor, a_bits[0], b_bits[0])]
  for a, b in zip(a_bits[1:], b_bits[1:]):
    pair = _seqmap(lambda x, y: 2 * x + y, a, b)
    sums.append(_seqmap(lambda p, c: (p.bit_count() + c) % 2, pair, carry))
    carry = _seqmap(lambda p, c: int(p.bit_count() + c >= 2), pair, carry)
  return sums + [carry]


def _add_bits(a_bits: Sequence[rasp.SOp], b_bits: Sequence[rasp.SOp]) -> Bits:
  """Ripple adder from bit x bit tables only: c' = g | (p & c), s = p ^ c."""
  gen = [_seqmap(_and, a, b) for a, b in zip(a_bits, b_bits)]
  prop = [_seqmap(_xor, a, b) for a, b in zip(a_bits, b_bits)]
  carry, sums = gen[0], [prop[0]]
  for g, p in zip(gen[1:], prop[1:]):
    sums.append(_seqmap(_xor, p, carry))
    carry = _seqmap(_or, g, _seqmap(_and, p, carry))
  return sums + [carry]


def ripple_add(a_bits: Sequence[rasp.SOp], b_bits: Sequence[rasp.SOp],
               fusion: str = KPG) -> Bits:
  """a + b as w + 1 bits (carry out last) with a ripple-carry chain.

  Args:
    a_bits: w bit S-Ops, least significant first.
    b_bits: w bit S-Ops, least significant first.
    fusion: BITS, PAIR or KPG; see the module docstring.

  Returns:
    w + 1 bit S-Ops, least significant first.
  """
  _check_widths(a_bits, b_bits)
  if fusion == KPG:
    return _add_kpg(a_bits, b_bits, RIPPLE)
  if fusion == PAIR:
    return _add_pair(a_bits, b_bits)
  if fusion == BITS:
    return _add_bits(a_bits, b_bits)
  raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")


def kogge_stone_add(a_bits: Sequence[rasp.SOp],
                    b_bits: Sequence[rasp.SOp]) -> Bits:
  """a + b as w + 1 bits with a Kogge-Stone carry network.

  Depth is ceil(log2 w) + 2 MLP layers (status, prefix levels, sum) for w a
  power of 2, against w for the KPG ripple adder.
  """
  _check_widths(a_bits, b_bits)
  return _add_kpg(a_bits, b_bits, KOGGE_STONE)


def increment(bits: Sequence[rasp.SOp], network: str = RIPPLE) -> Bits:
  """bits + 1 as w + 1 bits; carry i is the AND of bits 0..i."""
  if not bits:
    raise ValueError("Cannot increment an empty number.")
  carries = scan(bits, _and, network)
  sums = [_bit_map(lambda x: 1 - x, bits[0])]
  sums += [_seqmap(_xor, b, c) for b, c in zip(bits[1:], carries[:-1])]
  return sums + [carries[-1]]


def less_than(a_bits: Sequence[rasp.SOp], b_bits: Sequence[rasp.SOp],
              network: str = RIPPLE) -> rasp.SOp:
  """1 if a < b else 0: the most significant differing bit decides."""
  _check_widths(a_bits, b_bits)
  status = [_seqmap(lambda x, y: int(x < y), a_bits[0], b_bits[0])]
  status += [_seqmap(_less_status, a, b)
             for a, b in zip(a_bits[1:], b_bits[1:])]
  return reduce(status, _combine, network)


def from_bits(bits: Sequence[rasp.SOp]) -> rasp.SOp:
  """The integer sum(bit_i * 2^i) as one categorical S-Op.

  A balanced tree of SequenceMaps, hi * 2^k + lo; its top table alone has
  about 2^w units, since the output has 2^w values.
  """
  if not bits:
    raise ValueError("Cannot decode an empty number.")
  if len(bits) == 1:
    return bits[0]
  half = len(bits) // 2
  lo, hi = from_bits(bits[:half]), from_bits(bits[half:])
  return _seqmap(lambda h, l: (h << half) + l, hi, lo)


def to_bits(sop: rasp.SOp, width: int) -> Bits:
  """`width` bit S-Ops of a non-negative integer S-Op, one Map per bit.

  A small helper for tests and probes; each Map reads the whole input.
  """
  return [rasp.categorical(rasp.Map(lambda x, i=i: (x >> i) & 1, sop))
          for i in range(width)]
