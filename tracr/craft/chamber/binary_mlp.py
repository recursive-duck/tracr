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
"""Bit codes for values and MLPs that read or write them.

A binary-encoded variable with m possible values uses L = ceil(log2 m) dims
instead of m one-hot dims. The value with sorted index k is written as the
code c(k) in {-1, +1}^L whose bit i is +1 iff bit i of k is set. Every code
has squared norm L, and c(j) . c(k) = L - 2 * hamming(j, k), so the dot
product is L on the matching code and at most L - 2 on any other one.

An MLP reads a binary input with one hidden unit per input value v,
ReLU(c(v) . x - (L - 1)), which is exactly 1 when x = c(v) and 0 for every
other code (and for the all-zero BOS vector).
"""

import dataclasses
import functools
import itertools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from tracr.craft import bases
from tracr.craft import transformers
from tracr.craft import vectorspace_fns

# Basis direction values of the bit dims are f"{BIT_PREFIX}{i:0{BIT_DIGITS}d}";
# zero padding keeps the sorted basis in bit order.
BIT_PREFIX = "bit"
BIT_DIGITS = 2


def num_bits(num_values: int) -> int:
  """Bits needed to give `num_values` values distinct codes (at least 1)."""
  return max(1, (num_values - 1).bit_length())


def code_matrix(num_values: int, n_bits: int) -> np.ndarray:
  """[num_values, n_bits] matrix whose row k is the +-1 code of index k."""
  idx = np.arange(num_values)[:, None]
  bit = np.arange(n_bits)[None, :]
  return 2.0 * ((idx >> bit) & 1) - 1.0


@dataclasses.dataclass(frozen=True)
class BinaryCode:
  """The bit code of one binary-encoded variable.

  Attributes:
    name: Name of the bit directions (the S-Op label).
    values: All codable values, in code order (index k gets code c(k)).
  """
  name: bases.Name
  values: Tuple[bases.Value, ...]

  @functools.cached_property
  def n_bits(self) -> int:
    return num_bits(len(self.values))

  @functools.cached_property
  def space(self) -> bases.VectorSpaceWithBasis:
    return bases.VectorSpaceWithBasis([
        bases.BasisDirection(self.name, f"{BIT_PREFIX}{i:0{BIT_DIGITS}d}")
        for i in range(self.n_bits)
    ])

  @functools.cached_property
  def index(self) -> Dict[bases.Value, int]:
    return {v: k for k, v in enumerate(self.values)}

  @functools.cached_property
  def matrix(self) -> np.ndarray:
    """[len(values), n_bits] codes, columns in `space.basis` order."""
    return code_matrix(len(self.values), self.n_bits)

  def code(self, value: bases.Value) -> np.ndarray:
    return self.matrix[self.index[value]]

  def vector(self, value: bases.Value) -> bases.VectorInBasis:
    return self.space.make_vector(self.code(value))


def code_from_values(name: bases.Name, values) -> BinaryCode:
  """Code over `values`, in the sorted order categorical bases use."""
  ordered = bases.VectorSpaceWithBasis.from_values(name, values).basis
  return BinaryCode(name, tuple(d.value for d in ordered))


class Operand:
  """How an MLP reads (and writes) one variable in the residual stream."""

  space: bases.VectorSpaceWithBasis

  def values(self) -> Sequence[bases.Value]:
    """The values a hidden unit can match on."""
    raise NotImplementedError()

  def read(self, value: bases.Value) -> Tuple[np.ndarray, float]:
    """Weights over `space` and the dot product they reach on `value`."""
    raise NotImplementedError()

  def write(self, value: Any) -> Optional[np.ndarray]:
    """Row over `space` encoding `value`, or None if it is not encodable."""
    raise NotImplementedError()


class CategoricalOperand(Operand):
  """One-hot variable: a direction per value."""

  def __init__(self, space: bases.VectorSpaceWithBasis):
    self.space = space
    self._index = {d.value: i for i, d in enumerate(space.basis)}

  def values(self):
    return [d.value for d in self.space.basis]

  def read(self, value):
    row = np.zeros(self.space.num_dims)
    row[self._index[value]] = 1.0
    return row, 1.0

  def write(self, value):
    if value not in self._index:
      return None
    row = np.zeros(self.space.num_dims)
    row[self._index[value]] = 1.0
    return row


class BinaryOperand(Operand):
  """Bit-coded variable."""

  def __init__(self, code: BinaryCode):
    self.code = code
    self.space = code.space

  def values(self):
    return list(self.code.values)

  def read(self, value):
    return self.code.code(value), float(self.code.n_bits)

  def write(self, value):
    if value not in self.code.index:
      return None
    return self.code.code(value)


class NumericalOperand(Operand):
  """1-d variable holding the value itself (write-only here)."""

  def __init__(self, space: bases.VectorSpaceWithBasis):
    bases.ensure_dims(space, num_dims=1, name="numerical output space")
    self.space = space

  def values(self):
    raise NotImplementedError("A numerical input is not read by value.")

  def read(self, value):
    raise NotImplementedError("A numerical input is not read by value.")

  def write(self, value):
    return np.array([float(value)])


def lookup_mlp(
    inputs: Sequence[Operand],
    output: Operand,
    operation: Callable[..., Any],
    one_space: bases.VectorSpaceWithBasis,
    hidden_name: bases.Name,
) -> transformers.MLP:
  """MLP computing output = operation(*inputs) by a lookup table.

  There is one hidden unit per combination of input values whose result is
  encodable and nonzero; unit (v_1, ..., v_n) is
    ReLU(sum_i read_i(v_i) . x_i - (sum_i len_i - 1)),
  which is 1 when every input matches its value and 0 otherwise (a one-hot
  input reaches 1 on its match, a code of L bits reaches L and any other code
  at most L - 2). The output layer writes output.write(operation(...)) from
  each unit. A single one-hot input needs no ReLU threshold, so units with the
  same output are merged (as in categorical_mlp.map_categorical_mlp).

  Args:
    inputs: The operands to read; their spaces must be disjoint.
    output: The operand to write.
    operation: Function of the input values. It may return None for "no
      output", which writes nothing.
    one_space: 1-d space that holds 1 at every position.
    hidden_name: Name of the hidden directions.
  """
  bases.ensure_dims(one_space, num_dims=1, name="one_space")
  input_space = bases.join_vector_spaces(*(op.space for op in inputs),
                                         one_space)
  in_index = input_space.index_by_direction
  one_row = in_index[one_space.basis[0]]

  reads = []
  for op in inputs:
    rows = [op.read(v) for v in op.values()]
    cols = np.array([in_index[d] for d in op.space.basis])
    weights = np.stack([r for r, _ in rows]) if rows else np.zeros(
        (0, len(cols)))
    lengths = np.array([n for _, n in rows])
    reads.append((op.values(), cols, weights, lengths))

  merge = len(inputs) == 1 and isinstance(inputs[0], CategoricalOperand)

  # Hidden unit k reads value combo combos[k] and writes out_rows[k].
  fst_cols, out_rows, unit_of_output = [], [], {}
  for combo in itertools.product(*(range(len(r[0])) for r in reads)):
    result = operation(*(r[0][i] for r, i in zip(reads, combo)))
    if result is None:
      continue
    row = output.write(result)
    if row is None or not np.any(row):
      continue
    col = np.zeros(input_space.num_dims)
    length = 0.0
    for (_, cols, weights, lengths), i in zip(reads, combo):
      col[cols] += weights[i]
      length += lengths[i]
    if merge:
      key = row.tobytes()
      if key in unit_of_output:
        fst_cols[unit_of_output[key]] += col
        continue
      unit_of_output[key] = len(fst_cols)
    else:
      col[one_row] = -(length - 1.0)
    fst_cols.append(col)
    out_rows.append(row)

  hidden_space = bases.VectorSpaceWithBasis(
      [bases.BasisDirection(hidden_name, k) for k in range(len(fst_cols))])
  fst_matrix = (np.stack(fst_cols, axis=1) if fst_cols else
                np.zeros((input_space.num_dims, 0)))
  snd_matrix = (np.stack(out_rows) if out_rows else
                np.zeros((0, output.space.num_dims)))
  first_layer = vectorspace_fns.Linear(input_space, hidden_space, fst_matrix)
  second_layer = vectorspace_fns.Linear(hidden_space, output.space, snd_matrix)
  return transformers.MLP(first_layer, second_layer)


def fold_onehot_output(
    mlp: transformers.MLP,
    onehot_space: bases.VectorSpaceWithBasis,
    code: BinaryCode,
) -> transformers.MLP:
  """Rewrites an MLP that writes one-hot `onehot_space` to write bit codes.

  Every write of direction (name, v) becomes a write of the code of v, so an
  exact one-hot output becomes an exact code. Other output dims are kept.
  Directions whose value has no code are dropped.
  """
  snd = mlp.snd
  out_dirs = snd.output_space.basis
  kept = [d for d in out_dirs if d not in onehot_space]
  new_out = bases.join_vector_spaces(
      bases.VectorSpaceWithBasis(kept), code.space)
  new_index = new_out.index_by_direction
  bit_cols = np.array([new_index[d] for d in code.space.basis])
  # [D_old, D_new] linear map from the old output dims to the new ones.
  fold = np.zeros((len(out_dirs), new_out.num_dims))
  for i, d in enumerate(out_dirs):
    if d in onehot_space:
      if d.value in code.index:
        fold[i, bit_cols] = code.code(d.value)
    else:
      fold[i, new_index[d]] = 1.0
  new_snd = vectorspace_fns.Linear(snd.input_space, new_out,
                                   snd.matrix @ fold)
  return transformers.MLP(mlp.fst, new_snd)
