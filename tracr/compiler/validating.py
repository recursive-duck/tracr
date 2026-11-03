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
"""RASP Evaluator which applies causal masks to selectors."""

import collections
import dataclasses
from typing import Sequence, Union

from tracr.rasp import rasp

# S-Op types that have both a basis rule (basis_inference.py) and a craft
# component (expr_to_craft_graph.py). Full has a basis rule but no craft
# component, so it is not listed. ConstantSOp and LengthType have neither.
_COMPILABLE_SOP_TYPES = (
    rasp.TokensType,
    rasp.IndicesType,
    rasp.SelectorWidth,
    rasp.Map,
    rasp.SequenceMap,  # Includes LinearSequenceMap.
    rasp.Aggregate,
)


def and_terms(selector: rasp.Selector) -> list[rasp.Select] | None:
  """The Selects of a Select or an AND of Selects; None for other selectors.

  An AND compiles to one attention head whose score is the number of true
  predicates, so a key-query pair is selected only when all of them hold.
  """
  if isinstance(selector, rasp.Select):
    return [selector]
  if isinstance(selector, rasp.SelectorAnd):
    fst, snd = and_terms(selector.fst), and_terms(selector.snd)
    if fst is not None and snd is not None:
      return fst + snd
  return None


@dataclasses.dataclass
class TracrUnsupportedExpr:
  """An uncompilable expression and the reason it's not compilable."""

  expr: rasp.RASPExpr
  reason: str


def _unsupported_sop_reason(sop: rasp.SOp) -> str | None:
  """Returns why `sop` cannot be compiled, or None if it can."""
  if isinstance(sop, _COMPILABLE_SOP_TYPES):
    return None
  if isinstance(sop, rasp.LengthType):
    return (
        "rasp.length is not compilable. Use lib.make_length() instead, which"
        " is built from SelectorWidth."
    )
  if isinstance(sop, rasp.Full):
    return (
        "Full is not compilable: the compiler has no model component for it."
        " It is only usable inside selectors that simplification removes."
    )
  if isinstance(sop, rasp.ConstantSOp):
    return "ConstantSOp is a testing-only construct and is not compilable."
  return f"{type(sop).__name__} is not a supported S-Op."


def _static_validate_expr(expr: rasp.RASPExpr) -> TracrUnsupportedExpr | None:
  """Returns TracrUnsupportedExpr if `expr` is not supported by Tracr."""
  if isinstance(expr, rasp.SOp):
    sop_reason = _unsupported_sop_reason(expr)
    if sop_reason:
      return TracrUnsupportedExpr(expr=expr, reason=sop_reason)

  if isinstance(expr, rasp.Selector) and and_terms(expr) is None:
    reason = (
        f"{type(expr).__name__} is not compilable: only rasp.Select and ANDs"
        " of Selects are supported as selectors."
    )
    if isinstance(expr, (rasp.SelectorOr, rasp.SelectorNot)):
      reason += (
          " SelectorOr and SelectorNot compile only if simplification reduces"
          " them to a single Select."
      )
    return TracrUnsupportedExpr(expr=expr, reason=reason)

  if isinstance(expr, rasp.TokensType) and rasp.is_numerical(expr):
    return TracrUnsupportedExpr(
        expr=expr, reason="tokens should always be categorical."
    )
  if isinstance(expr, rasp.IndicesType) and rasp.is_numerical(expr):
    return TracrUnsupportedExpr(
        expr=expr, reason="indices should always be categorical."
    )

  if isinstance(expr, rasp.Select) and (rasp.is_binary(expr.keys) or
                                        rasp.is_binary(expr.queries)):
    if not (rasp.is_binary(expr.keys) and rasp.is_binary(expr.queries)):
      return TracrUnsupportedExpr(
          expr=expr,
          reason=(
              "A Select on a binary S-Op needs binary keys and queries (both"
              f" share one bit code). Keys: {rasp.get_encoding(expr.keys)},"
              f" queries: {rasp.get_encoding(expr.queries)}."
          ),
      )
    if expr.predicate != rasp.Comparison.EQ:
      return TracrUnsupportedExpr(
          expr=expr,
          reason=(
              "A Select on binary keys and queries only supports"
              f" Comparison.EQ (a code dot product), got {expr.predicate}."
              " Use categorical keys and queries for other predicates."
          ),
      )
  elif isinstance(expr, rasp.Select):
    if not rasp.is_categorical(expr.keys):
      return TracrUnsupportedExpr(
          expr=expr,
          reason="Select keys must be categorical.",
      )
    if not rasp.is_categorical(expr.queries):
      return TracrUnsupportedExpr(
          expr=expr,
          reason="Select queries must be categorical.",
      )

  if isinstance(expr, rasp.SelectorWidth):
    if and_terms(expr.selector) is None:
      return TracrUnsupportedExpr(
          expr=expr,
          reason=(
              "SelectorWidth's selector must be a rasp.Select or an AND of them, got"
              f" {type(expr.selector).__name__}."
          ),
      )

  if isinstance(expr, rasp.Aggregate):
    if and_terms(expr.selector) is None:
      return TracrUnsupportedExpr(
          expr=expr,
          reason=(
              "Aggregate's selector must be a rasp.Select or an AND of them, got"
              f" {type(expr.selector).__name__}."
          ),
      )
    if rasp.get_encoding(expr) != rasp.get_encoding(expr.sop):
      return TracrUnsupportedExpr(
          expr=expr,
          reason=(
              "An aggregate's output encoding must match its input encoding."
              f" Input: {rasp.get_encoding(expr)}  "
              f" Output: {rasp.get_encoding(expr.sop)}  "
          ),
      )

    if not rasp.is_numerical(expr) and expr.default is not None:
      return TracrUnsupportedExpr(
          expr=expr,
          reason=(
              "Categorical or binary aggregate only supports None as default"
              " value."
          ),
      )
    if rasp.is_numerical(expr) and expr.default != 0:
      return TracrUnsupportedExpr(
          expr=expr,
          reason="Numerical aggregate only supports 0 as default value.",
      )

  if isinstance(expr, rasp.SequenceMap):
    if not isinstance(expr, rasp.LinearSequenceMap) and not all(
        rasp.is_categorical(x) or rasp.is_binary(x)
        for x in (expr.fst, expr.snd, expr)
    ):
      return TracrUnsupportedExpr(
          expr=expr,
          reason=(
              "(Non-linear) SequenceMap only supports categorical or binary"
              " inputs/outputs."
          ),
      )

    if isinstance(expr, rasp.LinearSequenceMap) and not all(
        rasp.is_numerical(x) for x in (expr.fst, expr.snd, expr)
    ):
      return TracrUnsupportedExpr(
          expr=expr,
          reason="LinearSequenceMap only supports numerical inputs/outputs.",
      )


class DynamicValidationEvaluator(rasp.DefaultRASPEvaluator):
  """Evaluates RASP program but raises exceptions to anticipate compiler issues.

  Most features not supported by Tracr are specific input/output types for
  some SOp types and can be checked statically. For example, Tracr does not
  support Aggregate operations with different input and output encodings
  (instead, explicit conversion via a Map is required).

  There are some specific aggregate operations that are not supported and have
  to be checked dynamically. For example, Tracr does not support categorical
  Aggregate operations that require non-trival aggregation (eg, averaging
  tokens instead of moving tokens).
  """

  def __init__(self):
    self.unsupported_exprs = []
    super().__init__()

  def evaluate(
      self, expr: rasp.RASPExpr, xs: Sequence[rasp.Value]
  ) -> Union[Sequence[rasp.Value], rasp.SelectorValue]:
    out = super().evaluate(expr, xs)

    if isinstance(expr, rasp.Aggregate):
      # We support compiling programs which use Aggregates to move a single
      # categorical value to another position, ie when the attention pattern is
      # 1 in one place and 0 otherwise. However, if the attention pattern has
      # two or more 1s attending to different tokens that have to be aggregated
      # the compiled model will silently give incorrect outputs. We don't have
      # a way to do this statically so we have to check this at runtime.

      agg_in = expr.sop(xs)
      if (
          # The easiest way to satisfy this is to have a selector of width 1
          not rasp.is_numerical(expr)
          and not set(out).issubset(set(agg_in) | {None})
      ):
        self.unsupported_exprs.append(
            TracrUnsupportedExpr(
                expr=expr,
                reason=(
                    "Categorical aggregate does not support Selectors with"
                    " width > 1 that require aggregation (eg. averaging)."
                ),
            )
        )
      if rasp.is_numerical(expr) and not set(agg_in).issubset({0, 1}):
        self.unsupported_exprs.append(
            TracrUnsupportedExpr(
                expr=expr,
                reason=(
                    "Numerical aggregate only supports binary inputs 0, 1. But"
                    f" got {set(agg_in)}."
                ),
            )
        )

    return out


def static_validate(program: rasp.RASPExpr) -> list[TracrUnsupportedExpr]:
  """Performs static checks to see if `program` can be compiled.

  Args:
    program: RASP program to validate

  Returns:
    list of all unsupported subexpressions detectable statically.
  """
  # Breadth-first search over the RASP expression graph. Nodes are identified by
  # `label` (unique per expression instance), the same key the compiler uses in
  # rasp_to_graph. Names are not unique: every Aggregate is named "aggregate".
  # Each node is marked visited when enqueued, so shared subgraphs are walked
  # once and the cost stays linear in the number of distinct expressions.
  expr_queue = collections.deque([program])
  visited_labels = {program.label}
  unsupported_exprs = []
  # tokens and indices have fixed labels, so differently encoded copies would
  # collapse into one graph node; record every encoding they are used with.
  primitive_encodings = collections.defaultdict(dict)

  def note_primitive(expr):
    if isinstance(expr, (rasp.TokensType, rasp.IndicesType)):
      primitive_encodings[expr.label].setdefault(rasp.get_encoding(expr), expr)

  note_primitive(program)
  while expr_queue:
    expr = expr_queue.popleft()
    unsupported_expr = _static_validate_expr(expr)
    if unsupported_expr:
      unsupported_exprs.append(unsupported_expr)

    for child_expr in expr.children:
      note_primitive(child_expr)
      if child_expr.label not in visited_labels:
        visited_labels.add(child_expr.label)
        expr_queue.append(child_expr)

  for label, by_encoding in primitive_encodings.items():
    if len(by_encoding) > 1:
      encodings = sorted(e.value for e in by_encoding)
      unsupported_exprs.append(TracrUnsupportedExpr(
          expr=next(iter(by_encoding.values())),
          reason=(
              f"{label} is used with several encodings {encodings}; a"
              " program must encode it one way."
          ),
      ))

  return unsupported_exprs


def dynamic_validate(
    program: rasp.RASPExpr, xs: Sequence[rasp.Value] | None = None
) -> list[TracrUnsupportedExpr]:
  """Checks if `program` can be compiled for input `xs`.

  Args:
    program: RASP program to validate
    xs: Input sequence to use for dynamic compiler check. If None, only do
      static checks.

  Returns:
    list of all unsupported expressions according to the dynamic validation
  """
  validation_evaluator = DynamicValidationEvaluator()
  validation_evaluator.evaluate(expr=program, xs=xs)
  return validation_evaluator.unsupported_exprs


def validate(
    program: rasp.RASPExpr, xs: Sequence[rasp.Value] | None = None
) -> list[TracrUnsupportedExpr]:
  """Checks if `program` can be compiled for input `xs`.

  Args:
    program: RASP program to validate
    xs: Input sequence to use for dynamic compiler check. If None, only do
      static checks.

  Returns:
    list of all unsupported expressions
  """
  static_unsupported = static_validate(program)
  if xs is not None:
    dynamic_unsupported = dynamic_validate(program, xs)
    return static_unsupported + dynamic_unsupported
  return static_unsupported
