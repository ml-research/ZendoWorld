from type_system import *
from program import *

ID_IDX = 0
COLOR_IDX = 1
SHAPE_IDX = 2
ORIENT_IDX = 3
TOUCH_IDX = slice(4, 10)  # 6 directions
ON_TOP_IDX = 9
POINT_IDX = 10
BBOX_IDX = slice(11, 15)

# Define concrete type for Zendo pieces
PIECE = PrimitiveType("piece")
t0 = PIECE
STRUCTURE = List(PIECE)

UNARY_PRED = Arrow(PIECE, BOOL)
INTERACTION_PRED = Arrow(PIECE, Arrow(STRUCTURE, BOOL))
RULE = Arrow(STRUCTURE, BOOL)
AND_OR_RULE = Arrow(RULE, Arrow(RULE, RULE))
INTERACTION = Arrow(UNARY_PRED, Arrow(UNARY_PRED, INTERACTION_PRED))

# ---- Semantic functions ----
def valid_pieces(structure_tensor):
    return [piece for piece in structure_tensor if piece[ID_IDX].item() != 7]

def count_unary_predicate(pred, structure_tensor):
    try:
        pieces = valid_pieces(structure_tensor)
        if pred is None:
            return len(pieces)
        count = 0
        for piece in pieces:
            result = pred(piece)
            if isinstance(result, bool):
                count += result
        return count
    except Exception as e:
        print(f"count_unary_predicate error: {e}, {pred}")
        return 0

def count_conjunctive_unary_predicates(pred1, pred2, structure_tensor):
    try:
        pieces = valid_pieces(structure_tensor)
        count = 0
        for piece in pieces:
            r1 = pred1(piece)
            r2 = pred2(piece)
            if isinstance(r1, bool) and isinstance(r2, bool):
                if r1 and r2:
                    count += 1
        return count
    except Exception as e:
        print(f"count_conjunctive_unary_predicates error: {e}, {structure_tensor}")
        return 0

def count_interaction_predicate(pred, structure_tensor):
    try:
        pieces = valid_pieces(structure_tensor)
        count = 0
        for piece in pieces:
            result = pred(piece)(structure_tensor)
            if isinstance(result, bool):
                count += result
            else:
                print(f"[count_interaction_predicate] WARNING: Predicate returned non-bool: {result}")
        return count
    except Exception as e:
        print(f"count_interaction_predicate error: {e}")
        return 0

def is_grounded():
    return lambda piece: piece[ON_TOP_IDX].item() == 8

def is_ungrounded():
    return lambda piece: piece[ON_TOP_IDX].item() != 8

def has_color_idx(color_idx):
    def inner(piece):
        try:
            return piece[COLOR_IDX].item() == color_idx
        except Exception as e:
            print(f"has_color_idx error: {e}, piece={piece}, color_idx={color_idx}")
            return False
    return inner

def has_shape_idx(shape_idx):
    def inner(piece):
        try:
            return piece[SHAPE_IDX].item() == shape_idx
        except Exception as e:
            print(f"has_shape_idx error: {e}, piece={piece}, shape_idx={shape_idx}")
            return False
    return inner

def has_orient_idx(orient_idx):
    def inner(piece):
        return piece[ORIENT_IDX].item() == orient_idx
    return inner

def all_three_shapes():
    def inner(structure):
        try:
            shapes_present = set()
            for piece in valid_pieces(structure):
                shape = piece[SHAPE_IDX].item()
                shapes_present.add(shape)
            return all(shape in shapes_present for shape in [0, 1, 2])  # 0=block, 1=wedge, 2=pyramid
        except Exception as e:
            print(f"all_three_shapes error: {e}")
            return False
    return inner

def all_three_colors():
    def inner(structure):
        try:
            colors_present = set()
            for piece in valid_pieces(structure):
                color = piece[COLOR_IDX].item()
                colors_present.add(color)
            return all(color in colors_present for color in [0, 1, 2])  # 0=red, 1=blue, 2=yellow
        except Exception as e:
            print(f"all_three_colors error: {e}")
            return False
    return inner

def or_pred(pred1, pred2):
    return lambda piece: pred1(piece) or pred2(piece)

def and_pred(pred1, pred2):
    return lambda piece: pred1(piece) and pred2(piece)

def at_least_1(n, pred):
    def inner(structure):
        return count_unary_predicate(pred, structure) >= n
    return inner

def at_least_2(n, pred1, pred2):
    def inner(structure):
        return count_conjunctive_unary_predicates(pred1, pred2, structure) >= n
    return inner

def at_least_interaction(n, pred):
    def inner(structure):
        return count_interaction_predicate(pred, structure) >= n
    return inner

def exactly_1(n, pred):
    def inner(structure):
        return count_unary_predicate(pred, structure) == n
    return inner

def exactly_2(n, pred1, pred2):
    def inner(structure):
        return count_conjunctive_unary_predicates(pred1, pred2, structure) == n
    return inner

def exactly_interaction(n, pred):
    def inner(structure):
        return count_interaction_predicate(pred, structure) == n
    return inner

def even():
    def inner(structure):
        return count_unary_predicate(None, structure) % 2 == 0
    return inner

def even_1(pred):
    def inner(structure):
        count = count_unary_predicate(pred, structure)
        return count != 0 and count % 2 == 0
    return inner

def even_2(pred1, pred2):
    def inner(structure):
        count = count_conjunctive_unary_predicates(pred1, pred2, structure)
        return count != 0 and count % 2 == 0
    return inner

def even_interaction(pred):
    def inner(structure):
        count = count_interaction_predicate(pred, structure)
        return count != 0 and count % 2 == 0
    return inner

def odd():
    def inner(structure):
        return count_unary_predicate(None, structure) % 2 == 1
    return inner

def odd_1(pred):
    def inner(structure):
        return count_unary_predicate(pred, structure) % 2 == 1
    return inner

def odd_2(pred1, pred2):
    def inner(structure):
        return count_conjunctive_unary_predicates(pred1, pred2, structure) % 2 == 1
    return inner

def odd_interaction(pred):
    def inner(structure):
        return count_interaction_predicate(pred, structure) % 2 == 1
    return inner

def exclusively(pred):
    def inner(structure):
        pieces = len(valid_pieces(structure))
        return count_unary_predicate(pred, structure) == pieces
    return inner

def zero_1(pred):
    def inner(structure):
        count = count_unary_predicate(pred, structure)
        return count == 0
    return inner

def zero_2(pred1, pred2):
    def inner(structure):
        count = count_conjunctive_unary_predicates(pred1, pred2, structure)
        return count == 0
    return inner

def pointing_predicate(pred1, pred2):
    def inner(piece):
        def evaluate(structure_tensor):
            try:
                if piece[ID_IDX].item() == 7:
                    return False
                if not pred1(piece):
                    return False
                idx = piece[POINT_IDX].item()
                if idx < 0 or idx >= structure_tensor.shape[0]:
                    return False
                if structure_tensor[idx][ID_IDX].item() == 7:
                    return False
                return bool(pred2(structure_tensor[idx]))
            except Exception as e:
                print(f"pointing_predicate error: {e}, piece={piece}, structure_tensor={structure_tensor}")
                return False
        return evaluate
    return inner

def on_top_of_predicate(pred1, pred2):
    def inner(piece):
        def evaluate(structure_tensor):
            try:
                if piece[ID_IDX].item() == 7:
                    return False
                idx = piece[ON_TOP_IDX].item()
                if idx < 0 or idx >= structure_tensor.shape[0]:
                    return False
                if structure_tensor[idx][ID_IDX].item() == 7 or structure_tensor[idx][ID_IDX].item() == 8:
                    return False
                return bool(pred1(piece)) and bool(pred2(structure_tensor[idx]))
            except Exception as e:
                print(f"on_top_of_predicate error: {e}, piece={piece}, structure_tensor={structure_tensor}")
                return False
        return evaluate
    return inner

def touching_predicate(pred1, pred2):
    def inner(piece):
        def evaluate(structure_tensor):
            try:
                if piece[ID_IDX].item() == 7 or not pred1(piece):
                    return False
                for idx in piece[TOUCH_IDX].tolist():
                    if 0 <= idx < structure_tensor.shape[0]:
                        if structure_tensor[idx][ID_IDX].item() != 8 and pred2(structure_tensor[idx]):
                            return True
                return False
            except Exception as e:
                print(f"touching_predicate error: {e}, piece={piece}, structure_tensor={structure_tensor}")
                return False
        return evaluate
    return inner

def and_rule(rule1, rule2):
    def inner(structure_tensor):
        try:
            res1 = rule1(structure_tensor)
            res2 = rule2(structure_tensor)
            return res1 and res2
        except Exception as e:
            print(f"and_rule error: {e}, rule1={rule1}, rule2={rule2}, structure_tensor={structure_tensor}")
            return False
    return inner

def or_rule(rule1, rule2):
    def inner(structure_tensor):
        try:
            res1 = rule1(structure_tensor)
            res2 = rule2(structure_tensor)
            return res1 or res2
        except Exception as e:
            print(f"or_rule error: {e}, rule1={rule1}, rule2={rule2}, structure_tensor={structure_tensor}")
            return False
    return inner

def either_or(n1, n2):
    def inner(structure_tensor):
        try:
            pieces = valid_pieces(structure_tensor)
            count = len(pieces)
            return count == n1 or count == n2
        except Exception as e:
            print(f"either_or error: {e}, n1={n1}, n2={n2}, structure_tensor={structure_tensor}")
            return False
    return inner

def more_than(pred1, pred2):
    def inner(structure_tensor):
        try:
            count1 = count_unary_predicate(pred1, structure_tensor)
            count2 = count_unary_predicate(pred2, structure_tensor)
            return count1 > count2
        except Exception as e:
            print(f"more_than error: {e}, pred1={pred1}, pred2={pred2}, structure_tensor={structure_tensor}")
            return False
    return inner

def more_or_equal_than(pred1, pred2):
    def inner(structure_tensor):
        try:
            count1 = count_unary_predicate(pred1, structure_tensor)
            count2 = count_unary_predicate(pred2, structure_tensor)
            return count1 >= count2
        except Exception as e:
            print(f"more_than error: {e}, pred1={pred1}, pred2={pred2}, structure_tensor={structure_tensor}")
            return False
    return inner

def same_amount(pred1, pred2):
    def inner(structure_tensor):
        try:
            count1 = count_unary_predicate(pred1, structure_tensor)
            count2 = count_unary_predicate(pred2, structure_tensor)
            return count1 == count2 and count1 > 0
        except Exception as e:
            print(f"same_amount error: {e}, pred1={pred1}, pred2={pred2}, structure_tensor={structure_tensor}")
            return False
    return inner

def majority_1(pred):
    def inner(structure):
        pieces = valid_pieces(structure)
        count = count_unary_predicate(pred, structure)
        return count >= len(pieces) / 2
    return inner

def majority_2(pred1, pred2):
    def inner(structure):
        pieces = valid_pieces(structure)
        count = count_conjunctive_unary_predicates(pred1, pred2, structure)
        return count >= len(pieces) / 2
    return inner

def majority_interaction(pred):
    def inner(structure):
        pieces = valid_pieces(structure)
        count = count_interaction_predicate(pred, structure)
        return count >= len(pieces) / 2
    return inner

def length(n):
    def inner(structure_tensor):
        try:
            pieces = valid_pieces(structure_tensor)
            return len(pieces) == n
        except Exception as e:
            print(f"length error: {e}, n={n}, structure_tensor={structure_tensor}")
            return False
    return inner

def all_1(pred):
    def inner(structure):
        pieces = valid_pieces(structure)
        return count_unary_predicate(pred, structure) == len(pieces)
    return inner

def all_2(pred1, pred2):
    def inner(structure):
        pieces = valid_pieces(structure)
        return count_conjunctive_unary_predicates(pred1, pred2, structure) == len(pieces)
    return inner

# ---- DSL Semantics ----
semantics = {
    'LENGTH': lambda n: length(n),
    'SAME_AMOUNT': lambda pred1: lambda pred2: same_amount(pred1, pred2),
    # Updated AT_LEAST and EXACTLY with 1 or 2 attributes
    'AT_LEAST_1': lambda n: lambda pred1: at_least_1(n, pred1),
    'AT_LEAST_2': lambda n: lambda pred1: lambda pred2: at_least_2(n, pred1, pred2),
    'EXACTLY_1': lambda n: lambda pred1: exactly_1(n, pred1),
    'EXACTLY_2': lambda n: lambda pred1: lambda pred2: exactly_2(n, pred1, pred2),
    'MAJORITY_1': lambda pred1: majority_1(pred1),
    'MAJORITY_2': lambda pred1: lambda pred2: majority_2(pred1, pred2),
    'ALL_1': lambda pred1: all_1(pred1),
    'ALL_2': lambda pred1: lambda pred2: all_2(pred1, pred2),
    
    # Interaction versions for AT_LEAST and EXACTLY
    'AT_LEAST_INTERACTION': lambda n: lambda interaction: at_least_interaction(n, interaction),
    'EXACTLY_INTERACTION': lambda n: lambda interaction: exactly_interaction(n, interaction),
    'MAJORITY_INTERACTION': lambda interaction: majority_interaction(interaction),

    'EVEN':  even(),
    'EVEN_1': lambda pred: even_1(pred),
    'EVEN_2': lambda pred1: lambda pred2: even_2(pred1, pred2),
    'ODD':  odd(),
    'ODD_1': lambda pred: odd_1(pred),
    'ODD_2': lambda pred1: lambda pred2: odd_2(pred1, pred2),
    'ZERO_1': lambda pred: zero_1(pred),
    'ZERO_2': lambda pred1: lambda pred2: zero_2(pred1, pred2),

    'EXCLUSIVELY': lambda pred: exclusively(pred),
    'EVEN_INTERACTION': lambda interaction: even_interaction(interaction),
    'ODD_INTERACTION': lambda interaction: odd_interaction(interaction),
    
    'MORE_THAN': lambda pred1: lambda pred2: more_than(pred1, pred2),
    'MORE_OR_EQUAL_THAN': lambda pred1: lambda pred2: more_or_equal_than(pred1, pred2),
    'EITHER_OR': lambda n1: lambda n2: either_or(n1, n2),
    
    # Logical rules: AND, OR
    'AND': lambda pred1: lambda pred2: and_rule(pred1, pred2),
    'OR': lambda pred1: lambda pred2: or_rule(pred1, pred2),
    
    # Interaction predicates
    'TOUCHING': lambda pred1: lambda pred2: touching_predicate(pred1, pred2),
    'POINTING': lambda pred1: lambda pred2: pointing_predicate(pred1, pred2),
    'ON_TOP_OF': lambda pred1: lambda pred2: on_top_of_predicate(pred1, pred2),

    # Unary predicates
    'ALL_THREE_SHAPES': all_three_shapes(),
    'ALL_THREE_COLORS': all_three_colors(),
    
    # Basic unary predicates
    'IS_GROUNDED': is_grounded(),
    'IS_UNGROUNDED': is_ungrounded(),
    'IS_RED': has_color_idx(0),
    'IS_BLUE': has_color_idx(1),
    'IS_YELLOW': has_color_idx(2),
    'IS_BLOCK': has_shape_idx(0),
    'IS_WEDGE': has_shape_idx(1),
    'IS_PYRAMID': has_shape_idx(2),
    'IS_UPRIGHT': has_orient_idx(0),
    'IS_UPSIDE_DOWN': has_orient_idx(1),
    'IS_DOORSTOP': and_pred(has_orient_idx(2), has_shape_idx(1)),
    'IS_CHEESECAKE': has_orient_idx(3),
    'IS_VERTICAL': or_pred(has_orient_idx(0), has_orient_idx(1)),
    'IS_FLAT': or_pred(has_orient_idx(2), has_orient_idx(3)),
}

# ---- DSL Type Signatures ----
primitive_types = {
    'SAME_AMOUNT': Arrow(UNARY_PRED, Arrow(UNARY_PRED, RULE)),
    'LENGTH': Arrow(INT, RULE),
    # Logical combinators
    "AND": AND_OR_RULE,
    "OR": AND_OR_RULE,

    # Basic predicates (no input required, already curried functions)
    "IS_GROUNDED": UNARY_PRED,
    "IS_UNGROUNDED": UNARY_PRED,
    "IS_RED": UNARY_PRED,
    "IS_BLUE": UNARY_PRED,
    "IS_YELLOW": UNARY_PRED,
    "IS_BLOCK": UNARY_PRED,
    "IS_PYRAMID": UNARY_PRED,
    "IS_WEDGE": UNARY_PRED,
    "IS_UPRIGHT": UNARY_PRED,
    "IS_FLAT": UNARY_PRED,
    "IS_UPSIDE_DOWN": UNARY_PRED,
    "IS_CHEESECAKE": UNARY_PRED,
    "IS_DOORSTOP": UNARY_PRED,
    "IS_VERTICAL": UNARY_PRED,

    # Composed unary rules
    "AT_LEAST_1": Arrow(INT, Arrow(UNARY_PRED, RULE)),
    "AT_LEAST_2": Arrow(INT, Arrow(UNARY_PRED, (Arrow(UNARY_PRED, RULE)))),
    "AT_LEAST_INTERACTION": Arrow(INT, Arrow(INTERACTION_PRED, RULE)),

    "EXACTLY_1": Arrow(INT, Arrow(UNARY_PRED, RULE)),
    "EXACTLY_2": Arrow(INT, Arrow(UNARY_PRED, (Arrow(UNARY_PRED, RULE)))),
    "EXACTLY_INTERACTION": Arrow(INT, Arrow(INTERACTION_PRED, RULE)),

    "EVEN": RULE,
    "EVEN_1": Arrow(UNARY_PRED, RULE),
    "EVEN_2": Arrow(UNARY_PRED, Arrow(UNARY_PRED, RULE)),
    "EVEN_INTERACTION":  Arrow(INTERACTION_PRED, RULE),

    "MAJORITY_1": Arrow(UNARY_PRED, RULE),
    "MAJORITY_2": Arrow(UNARY_PRED, Arrow(UNARY_PRED, RULE)),
    "MAJORITY_INTERACTION":  Arrow(INTERACTION_PRED, RULE),

    "ALL_1": Arrow(UNARY_PRED, RULE),
    "ALL_2": Arrow(UNARY_PRED, Arrow(UNARY_PRED, RULE)),

    "ODD": RULE,
    "ODD_1": Arrow(UNARY_PRED, RULE),
    "ODD_2": Arrow(UNARY_PRED, Arrow(UNARY_PRED, RULE)),
    "ODD_INTERACTION": Arrow(INTERACTION_PRED, RULE),

    "EXCLUSIVELY": Arrow(UNARY_PRED, RULE),

    "ZERO_1": Arrow(UNARY_PRED, RULE),
    "ZERO_2": Arrow(UNARY_PRED, Arrow(UNARY_PRED, RULE)),

    "MORE_THAN": Arrow(UNARY_PRED, Arrow(UNARY_PRED, RULE)),
    "MORE_OR_EQUAL_THAN": Arrow(UNARY_PRED, Arrow(UNARY_PRED, RULE)),
    "EITHER_OR": Arrow(INT, Arrow(INT, RULE)),

    # Interaction combinators (pred × pred -> pred)
    "TOUCHING": INTERACTION,
    "POINTING": INTERACTION,
    "ON_TOP_OF": INTERACTION,

    "ALL_THREE_SHAPES": RULE,
    "ALL_THREE_COLORS": RULE,
}

no_repetitions = set()