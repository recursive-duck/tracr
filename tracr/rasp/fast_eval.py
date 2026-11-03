"""Batched, vectorized evaluation of RASP programs.

`evaluate_batch(expr, inputs)` gives the same result as
`[rasp.evaluate(expr, x) for x in inputs]` for inputs of one length, value for
value and type for type, but works on whole arrays:

- An SOp value is held as integer codes [batch, length] into a table of its
  distinct Python values (told apart by type, so 1, 1.0 and True stay apart).
- Map / SequenceMap call their function once per distinct value (or distinct
  pair that occurs at one position), then gather.
- Select calls its predicate once per (distinct query, distinct key) pair and
  gathers a [batch, queries, keys] boolean array.
- SelectorWidth sums rows; Aggregate takes the single selected value, the
  default, or the mean exactly as `rasp._mean` does. Means of ints and bools
  are computed in int64 (exact, then one correctly rounded division, as
  Python's int / int). Means over floats call `rasp._mean` itself, memoized
  per selected sequence of values, since Python's float sum is compensated.

Encodings (numerical / categorical) do not change evaluation, as in
`rasp.evaluate`. Subexpressions shared in the graph are evaluated once.

Differences from calling `rasp.evaluate` per input:
- If any input of the batch would raise, the whole batch raises.
- Functions and predicates are assumed pure: they run once per distinct
  value, not once per position.

`evaluate_many` groups inputs of mixed lengths by length.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tracr.rasp import rasp

# Ints whose summed magnitude stays below this are averaged in int64 / float64
# exactly; larger ones fall back to rasp._mean.
_EXACT_INT_LIMIT = 2**53
_CODE_DTYPE = np.int64
# Entries of a predicate table.
_FALSE, _TRUE, _RAISED = 0, 1, 2


def _key(value: Any) -> Any:
  """Dict key that tells apart values Python considers equal (1, 1.0, True)."""
  if isinstance(value, float):
    return (float, value, math.copysign(1.0, value))
  try:
    hash(value)
  except TypeError:
    return ("unhashable", id(value))
  return (type(value), value)


class _Table:
  """Distinct values in first-seen order, with a code for each."""

  def __init__(self):
    self.values: List[Any] = []
    self._codes: Dict[Any, int] = {}

  def add(self, value: Any) -> int:
    k = _key(value)
    code = self._codes.get(k)
    if code is None:
      code = len(self.values)
      self._codes[k] = code
      self.values.append(value)
    return code

  def add_all(self, values: Sequence[Any]) -> np.ndarray:
    return np.fromiter((self.add(v) for v in values), dtype=_CODE_DTYPE,
                       count=len(values))


class Coded:
  """An SOp's value on a batch: codes [batch, length] into `values`."""

  def __init__(self, codes: np.ndarray, values: List[Any]):
    self.codes = codes
    self.values = values

  def tolist(self) -> List[List[Any]]:
    vals = self.values
    return [[vals[c] for c in row] for row in self.codes.tolist()]


def _from_values(rows: Sequence[Sequence[Any]]) -> Coded:
  table = _Table()
  codes = np.array([[table.add(v) for v in row] for row in rows],
                   dtype=_CODE_DTYPE)
  return Coded(codes.reshape(len(rows), -1), table.values)


def _remap(coded: Coded, new_values: Sequence[Any]) -> Coded:
  """Replaces each distinct value by new_values[code] and merges equal ones."""
  table = _Table()
  remap = table.add_all(new_values)
  return Coded(remap[coded.codes], table.values)


_COMPARISON_UFUNCS = {
    rasp.Comparison.EQ: np.equal,
    rasp.Comparison.LT: np.less,
    rasp.Comparison.LEQ: np.less_equal,
    rasp.Comparison.GT: np.greater,
    rasp.Comparison.GEQ: np.greater_equal,
    rasp.Comparison.NEQ: np.not_equal,
}


def _comparison_table(pred, keys: List[Any],
                      queries: List[Any]) -> Optional[np.ndarray]:
  """[queries, keys] table of a built-in Comparison in one ufunc call.

  The ufunc runs on object arrays, so each entry is the same Python
  comparison `key <op> query` that rasp.Comparison makes. Returns None (use
  the generic path) for other predicates and for None values, on which
  Comparison raises.
  """
  if not isinstance(pred, rasp.Comparison):
    return None
  if any(v is None for v in keys) or any(v is None for v in queries):
    return None
  shape = (len(queries), len(keys))
  if pred is rasp.Comparison.TRUE:
    return np.ones(shape, dtype=bool)
  if pred is rasp.Comparison.FALSE:
    return np.zeros(shape, dtype=bool)
  k = np.empty(len(keys), dtype=object)
  k[:] = keys
  q = np.empty(len(queries), dtype=object)
  q[:] = queries
  try:
    # Object comparison loops store bool(key <op> query).
    return _COMPARISON_UFUNCS[pred](k[None, :], q[:, None]).astype(bool)
  except Exception:  # pylint: disable=broad-except
    # e.g. "a" < 1: raise only where such a pair occurs, via the generic path.
    return None


class FastEvaluator:
  """Evaluates RASP expressions on a batch of inputs of one length."""

  def __init__(self, inputs: Sequence[Sequence[rasp.Value]]):
    self.batch = len(inputs)
    lengths = {len(x) for x in inputs}
    if len(lengths) > 1:
      raise ValueError(f"Inputs have several lengths {sorted(lengths)}; use "
                       "evaluate_many.")
    self.length = lengths.pop() if lengths else 0
    self.inputs = inputs
    self._cache: Dict[int, Union[Coded, np.ndarray]] = {}
    self._fns: Dict[type, Callable[[Any], Union[Coded, np.ndarray]]] = {
        rasp.TokensType: self._tokens,
        rasp.IndicesType: self._indices,
        rasp.LengthType: self._length,
        rasp.LinearSequenceMap: self._sequence_map,
        rasp.SequenceMap: self._sequence_map,
        rasp.Map: self._map,
        rasp.Full: self._full,
        rasp.ConstantSOp: self._constant_sop,
        rasp.SelectorWidth: self._selector_width,
        rasp.Aggregate: self._aggregate,
        rasp.Select: self._select,
        rasp.SelectorAnd: self._selector_and,
        rasp.SelectorOr: self._selector_or,
        rasp.SelectorNot: self._selector_not,
        rasp.ConstantSelector: self._constant_selector,
    }

  def evaluate(self, expr: rasp.RASPExpr) -> Union[Coded, np.ndarray]:
    """Coded values for an SOp, a [batch, queries, keys] array for a Selector."""
    key = id(expr)
    hit = self._cache.get(key)
    if hit is not None:
      return hit
    fn = self._fns.get(type(expr))
    if fn is None:
      raise NotImplementedError(f"Evaluation of {expr} is not defined.")
    out = fn(expr)
    # The expression is kept alive by the program, so its id stays unique.
    self._cache[key] = out
    return out

  def _broadcast(self, row: np.ndarray) -> np.ndarray:
    return np.broadcast_to(row, (self.batch,) + row.shape)

  # Primitives.

  def _tokens(self, expr) -> Coded:
    del expr
    return _from_values(self.inputs)

  def _indices(self, expr) -> Coded:
    del expr
    codes = self._broadcast(np.arange(self.length, dtype=_CODE_DTYPE))
    return Coded(codes, list(range(self.length)))

  def _length(self, expr) -> Coded:
    del expr
    codes = np.zeros((self.batch, self.length), dtype=_CODE_DTYPE)
    return Coded(codes, [self.length])

  def _full(self, expr: rasp.Full) -> Coded:
    codes = np.zeros((self.batch, self.length), dtype=_CODE_DTYPE)
    return Coded(codes, [expr.fill])

  def _constant_sop(self, expr: rasp.ConstantSOp) -> Coded:
    if expr.check_length and (self.length != len(expr.value)):
      raise ValueError(f"Constant len {len(expr.value)} doesn't match input "
                       f"len {self.length}.")
    table = _Table()
    row = table.add_all(list(expr.value))
    return Coded(self._broadcast(row), table.values)

  # SOps.

  def _map(self, expr: rasp.Map) -> Coded:
    inner = self.evaluate(expr.inner)
    f = expr.f
    return _remap(inner, [f(x) if x is not None else None
                          for x in inner.values])

  def _sequence_map(self, expr: rasp.SequenceMap) -> Coded:
    fst = self.evaluate(expr.fst)
    snd = self.evaluate(expr.snd)
    n = min(fst.codes.shape[1], snd.codes.shape[1])  # zip truncates
    pair = fst.codes[:, :n] * len(snd.values) + snd.codes[:, :n]
    pairs, inverse = np.unique(pair, return_inverse=True)
    f = expr.f
    out = []
    for p in pairs.tolist():
      x, y = fst.values[p // len(snd.values)], snd.values[p % len(snd.values)]
      out.append(f(x, y) if None not in [x, y] else None)
    table = _Table()
    remap = table.add_all(out)
    return Coded(remap[inverse.reshape(pair.shape)], table.values)

  def _selector_width(self, expr: rasp.SelectorWidth) -> Coded:
    sel = self.evaluate(expr.selector)
    widths = sel.sum(-1)
    uniq, inverse = np.unique(widths, return_inverse=True)
    table = _Table()
    remap = table.add_all(uniq.tolist())
    return Coded(remap[inverse.reshape(widths.shape)], table.values)

  def _aggregate(self, expr: rasp.Aggregate) -> Coded:
    sel = self.evaluate(expr.selector)
    vals = self.evaluate(expr.sop)
    k = min(sel.shape[-1], vals.codes.shape[1])  # _get_selected zips
    sel = sel[..., :k].astype(bool)
    codes = vals.codes[:, :k]
    count = sel.sum(-1)
    table = _Table()
    out = np.empty(count.shape, dtype=_CODE_DTYPE)
    none = count == 0
    if none.any():
      out[none] = table.add(expr.default)
    one = count == 1
    if one.any():
      picked = np.take_along_axis(codes, sel.argmax(-1), axis=1)[one]
      # Only values that are picked enter the table: a later Map must not
      # call its function on values no input produces.
      uniq, inverse = np.unique(picked, return_inverse=True)
      out[one] = table.add_all([vals.values[c] for c in uniq.tolist()])[inverse]
    many = count >= 2
    if many.any():
      out[many] = self._means(sel, codes, vals.values, many, expr.default,
                              table)
    return Coded(out, table.values)

  def _means(self, sel, codes, values, many, default, table) -> np.ndarray:
    """Codes of rasp._mean over the selected values where >= 2 are selected."""
    numeric = np.array([isinstance(v, (int, bool, float)) for v in values],
                       dtype=bool)
    bad = (sel & ~numeric[codes][:, None, :]).any(-1) & many
    is_int = [isinstance(v, int) for v in values]
    magnitude = sum(abs(v) for v, i in zip(values, is_int) if i)
    exact = (not bad.any() and all(i or not n for i, n in zip(is_int, numeric))
             and magnitude * sel.shape[-1] < _EXACT_INT_LIMIT)
    if exact:
      ints = np.array([int(v) if i else 0 for v, i in zip(values, is_int)],
                      dtype=np.int64)[codes]
      sums = np.matmul(sel.astype(np.int64), ints[:, :, None])[..., 0]
      means = sums[many] / sel.sum(-1)[many]
      uniq, inverse = np.unique(means, return_inverse=True)
      return table.add_all(uniq.tolist())[inverse]
    # Floats, huge ints or non-numeric values: rasp._mean itself (it raises
    # for non-numeric ones), once per distinct selected sequence.
    memo: Dict[Tuple[int, ...], int] = {}
    out = []
    for b, q in zip(*np.nonzero(many)):
      picked = tuple(codes[b][sel[b, q]].tolist())
      code = memo.get(picked)
      if code is None:
        mean = rasp._mean([values[c] for c in picked], default)  # pylint: disable=protected-access
        code = memo[picked] = table.add(mean)
      out.append(code)
    return np.array(out, dtype=_CODE_DTYPE)

  # Selectors.

  def _select(self, expr: rasp.Select) -> np.ndarray:
    keys = self.evaluate(expr.keys)
    queries = self.evaluate(expr.queries)
    pred = expr.predicate
    table = _comparison_table(pred, keys.values, queries.values)
    if table is not None:
      return table[queries.codes[:, :, None], keys.codes[:, None, :]]
    table = np.empty((len(queries.values), len(keys.values)), dtype=np.int8)
    errors = {}
    for qi, query in enumerate(queries.values):
      for ki, key in enumerate(keys.values):
        try:
          table[qi, ki] = _TRUE if bool(pred(key, query)) else _FALSE
        except Exception as e:  # pylint: disable=broad-except
          # Raised only if this pair occurs in one input, as rasp.evaluate.
          table[qi, ki] = _RAISED
          errors[(qi, ki)] = e
    out = table[queries.codes[:, :, None], keys.codes[:, None, :]]
    if errors:
      hit = np.argwhere(out == _RAISED)
      if len(hit):
        b, q, k = hit[0]
        raise errors[(int(queries.codes[b, q]), int(keys.codes[b, k]))]
    return out.astype(bool)

  def _constant_selector(self, expr: rasp.ConstantSelector) -> np.ndarray:
    if expr.check_length and (self.length != len(expr.value)):
      raise ValueError(f"Constant len {self.length} doesn't match input len "
                       f"{len(expr.value)}.")
    return self._broadcast(np.asarray(expr.value))

  def _selector_and(self, expr: rasp.SelectorAnd) -> np.ndarray:
    return np.logical_and(self.evaluate(expr.fst), self.evaluate(expr.snd))

  def _selector_or(self, expr: rasp.SelectorOr) -> np.ndarray:
    return np.logical_or(self.evaluate(expr.fst), self.evaluate(expr.snd))

  def _selector_not(self, expr: rasp.SelectorNot) -> np.ndarray:
    return np.logical_not(self.evaluate(expr.inner))


def evaluate_batch(
    expr: rasp.RASPExpr, inputs: Sequence[Sequence[rasp.Value]]
) -> List[Union[Sequence[rasp.Value], rasp.SelectorValue]]:
  """[rasp.evaluate(expr, x) for x in inputs], for inputs of one length."""
  if not inputs or not inputs[0]:
    # Empty sequences: the value tables of indices, length and Full would
    # hold values no position has.
    return [rasp.evaluate(expr, x) for x in inputs]
  out = FastEvaluator(inputs).evaluate(expr)
  if isinstance(out, Coded):
    return out.tolist()
  return [row.tolist() for row in out]


def evaluate_many(
    expr: rasp.RASPExpr, inputs: Sequence[Sequence[rasp.Value]]
) -> List[Union[Sequence[rasp.Value], rasp.SelectorValue]]:
  """Like evaluate_batch, for inputs of mixed lengths (one batch per length)."""
  by_len: Dict[int, List[int]] = {}
  for i, x in enumerate(inputs):
    by_len.setdefault(len(x), []).append(i)
  out: List[Any] = [None] * len(inputs)
  for idx in by_len.values():
    for i, row in zip(idx, evaluate_batch(expr, [inputs[i] for i in idx])):
      out[i] = row
  return out
