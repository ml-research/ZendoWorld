from DSL.zendo import AND_OR_RULE, INTERACTION_PRED, PIECE, RULE, UNARY_PRED, and_pred, count_conjunctive_unary_predicates, count_interaction_predicate, count_unary_predicate, has_color_idx, has_orient_idx, has_shape_idx, is_grounded, is_ungrounded, or_pred, valid_pieces
from type_system import *
from program import *
import operator

def count(pred):
    return lambda structure: count_unary_predicate(pred, structure)

def count2(pred1, pred2):
    return lambda structure: count_conjunctive_unary_predicates(pred1, pred2, structure)

def count_inter(pred):
    return lambda structure: count_interaction_predicate(pred, structure)

def make_comparator(op):
    # op: a function like operator.ge, operator.eq, etc.
    def comparator(f1, f2):
        def rule(structure):
            x = f1(structure)
            y = f2(structure)
            return op(x, y) and y != 0  # or whatever logic you like
        return rule
    return comparator

# ---- Semantics map ----
semantics = {
    # counting
    "COUNT": lambda pred: count(pred),
    "COUNT2": lambda pred1: lambda pred2: count2(pred1, pred2),
    "COUNT_INTER": lambda interaction: count_inter(interaction),
    "LENGTH": lambda structure: len(valid_pieces(structure)),

    # comparisons
    ">=": make_comparator(operator.ge),
    "==": make_comparator(operator.eq),
    ">": make_comparator(operator.gt),
    ">=": make_comparator(operator.ge),
    "<": make_comparator(operator.lt),
    "<=": make_comparator(operator.le),
    "MAJORITY_1": lambda pred: lambda structure: count(pred) > len(valid_pieces(structure)) / 2,
    "MAJORITY_2": lambda pred1: lambda pred2: lambda structure: count2(pred1, pred2) > len(valid_pieces(structure)) / 2,
    "EVEN": lambda x: x % 2 == 0,
    "ODD": lambda x: x % 2 == 1,

    # booleans
    "NOT": lambda rule: not rule,
    "AND": lambda r1: lambda r2: r1 and r2,
    "OR": lambda r1: lambda r2: r1 or r2,

    # unary preds (reuse your definitions)
    "IS_RED": has_color_idx(0),
    "IS_BLUE": has_color_idx(1),
    "IS_YELLOW": has_color_idx(2),
    "IS_BLOCK": has_shape_idx(0),
    "IS_WEDGE": has_shape_idx(1),
    "IS_PYRAMID": has_shape_idx(2),
    "IS_UPRIGHT": has_orient_idx(0),
    "IS_UPSIDE_DOWN": has_orient_idx(1),
    "IS_CHEESECAKE": has_orient_idx(3),
    "IS_DOORSTOP": and_pred(has_orient_idx(2), has_shape_idx(1)),
    "IS_VERTICAL": or_pred(has_orient_idx(0), has_orient_idx(1)),
    "IS_FLAT": or_pred(has_orient_idx(2), has_orient_idx(3)),
    "IS_GROUNDED": is_grounded(),
    "IS_UNGROUNDED": is_ungrounded(),
}

# ---- Types ----
COUNT_EXPR = PrimitiveType("count_expr")
CMP = Arrow(COUNT_EXPR, Arrow(COUNT_EXPR, RULE))
primitive_types = {
    # counting
    "COUNT": Arrow(UNARY_PRED, COUNT_EXPR),
    "COUNT2": Arrow(UNARY_PRED, Arrow(UNARY_PRED, COUNT_EXPR)),
    "COUNT_INTER": Arrow(INTERACTION_PRED, COUNT_EXPR),
    "LENGTH": Arrow(List(PIECE), COUNT_EXPR),

    # comparisons
    "==": CMP,
    "!=": CMP,
    ">": CMP,
    ">=": CMP,
    "<": CMP,
    "<=": CMP,
    "MAJORITY_1": Arrow(UNARY_PRED, RULE),
    "MAJORITY_2": Arrow(UNARY_PRED, Arrow(UNARY_PRED, RULE)),
    "EVEN": Arrow(Arrow(List(PIECE), COUNT_EXPR), RULE),
    "ODD": Arrow(Arrow(List(PIECE), COUNT_EXPR), RULE),


    # boolean combinators
    "NOT": Arrow(RULE, RULE),
    "AND": AND_OR_RULE,
    "OR": AND_OR_RULE,

    # unary preds
    "IS_RED": UNARY_PRED,
    "IS_BLUE": UNARY_PRED,
    "IS_YELLOW": UNARY_PRED,
    "IS_BLOCK": UNARY_PRED,
    "IS_WEDGE": UNARY_PRED,
    "IS_PYRAMID": UNARY_PRED,
    "IS_UPRIGHT": UNARY_PRED,
    "IS_UPSIDE_DOWN": UNARY_PRED,
    "IS_CHEESECAKE": UNARY_PRED,
    "IS_DOORSTOP": UNARY_PRED,
    "IS_VERTICAL": UNARY_PRED,
    "IS_FLAT": UNARY_PRED,
    "IS_GROUNDED": UNARY_PRED,
    "IS_UNGROUNDED": UNARY_PRED,
}
