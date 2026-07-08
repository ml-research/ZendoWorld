import os
import sys
import warnings

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType
from cons_list import index

from itertools import combinations_with_replacement

# Environments are cons lists: list = None | (value, list).
# `probability` is a dict {(G.__hash__(), S): p} where p is the probability that this
# Program is generated from non-terminal S under PCFG G.

# Hashes must be deterministic across processes.
if os.getenv('PYTHONHASHSEED') != '0':
    warnings.warn(
        "PYTHONHASHSEED is not set to 0. Deterministic hashing is not guaranteed.\n"
        "For reproducibility, run Python with: PYTHONHASHSEED=0"
    )

def strip_trailing_var0(prog):
    if isinstance(prog, Function) and len(prog.arguments) > 0:
        last_arg = prog.arguments[-1]
        if isinstance(last_arg, Variable) and repr(last_arg) == "var0":
            prog.arguments = prog.arguments[:-1]
    return prog

class Program:
    """A program: a lambda term with basic primitives."""

    def __eq__(self, other):
        return (
            isinstance(self, Program)
            and isinstance(other, Program)
            and self.type.__eq__(other.type)
            and self.typeless_eq(other)
        )

    def typeless_eq(self, other):
        b = isinstance(self, Program) and isinstance(other, Program)
        b2 = (
            isinstance(self, Variable)
            and isinstance(other, Variable)
            and self.variable == other.variable
        )
        b2 = b2 or (
            isinstance(self, Function)
            and isinstance(other, Function)
            and self.function.typeless_eq(other.function)
            and len(self.arguments) == len(other.arguments)
            and all(
                [
                    x.typeless_eq(y)
                    for x, y in zip(self.arguments, other.arguments)
                ]
            )
        )
        b2 = b2 or (
            isinstance(self, Lambda)
            and isinstance(other, Lambda)
            and self.body.typeless_eq(other.body)
        )
        b2 = b2 or (
            isinstance(self, BasicPrimitive)
            and isinstance(other, BasicPrimitive)
            and self.primitive == other.primitive
        )
        b2 = b2 or (
            isinstance(self, New)
            and isinstance(other, New)
            and (self.body).typeless_eq(other.body)
        )
        return b and b2

    def __gt__(self, other):
        True

    def __lt__(self, other):
        False

    def __ge__(self, other):
        True

    def __le__(self, other):
        False

    def __hash__(self):
        return self.hash

    def is_constant(self):
        return True

    def derive_with_constants(self, constants):
        return self

    def make_all_constant_variations(self, constants_list):
        n_constants = self.count_constants()
        if n_constants == 0:
            return [self]
        all_possibilities = combinations_with_replacement(constants_list, n_constants)
        return [self.derive_with_constants(list(possibility)) for possibility in all_possibilities]

    def count_constants(self):
        return 0

class Variable(Program):
    def __init__(self, variable, type_=UnknownType(), probability={}):
        self.variable = variable
        self.type = type_
        self.hash = variable

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        return "var" + format(self.variable)

    def eval(self, dsl, environment, i):
        if i in self.evaluation:
            return self.evaluation[i]
        try:
            result = index(environment, self.variable)
            self.evaluation[i] = result
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            result = index(environment, self.variable)
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError) as e:
            print(f"Error evaluating variable {self.variable}: {e}")
            return None

    def is_constant(self):
        return False

class Function(Program):
    def __init__(self, function, arguments, type_=UnknownType(), probability={}):
        self.function = function
        self.arguments = arguments
        self.type = type_
        self.hash = hash(tuple([arg.hash for arg in self.arguments] + [self.function.hash]))

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        if len(self.arguments) == 0:
            return format(self.function)
        else:
            s = "(" + format(self.function)
            for arg in self.arguments:
                s += " " + format(arg)
            return s + ")"

    def eval(self, dsl, environment, i):
        try:
            if len(self.arguments) == 0:
                return self.function.eval(dsl, environment, i)
            else:
                evaluated_arguments = []
                for j in range(len(self.arguments)):
                    e = self.arguments[j].eval(dsl, environment, i)
                    evaluated_arguments.append(e)
                result = self.function.eval(dsl, environment, i)
                for evaluated_arg in evaluated_arguments:
                    if not callable(result):
                        raise TypeError(f"Trying to apply non-callable: {result} with arg: {evaluated_arg}")
                    result = result(evaluated_arg)
                self.evaluation[i] = result
                return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            import traceback
            traceback.print_exc()
            print(f"Error evaluating function {self.function} with arguments {self.arguments}")
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            if len(self.arguments) == 0:
                return self.function.eval_naive(dsl, environment)
            else:
                evaluated_arguments = []
                for j in range(len(self.arguments)):
                    e = self.arguments[j].eval_naive(dsl, environment)
                    evaluated_arguments.append(e)
                result = self.function.eval_naive(dsl, environment)
                for evaluated_arg in evaluated_arguments:
                    result = result(evaluated_arg)
                return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError) as e:
            print(f"Error evaluating function {self.function} with arguments {self.arguments}")
            print(f"Error details: {e}")
            return None

    def is_constant(self):
        return all([self.function.is_constant()] + [arg.is_constant() for arg in self.arguments])

    def count_constants(self):
        return self.function.count_constants() + sum([arg.count_constants() for arg in self.arguments])

    def derive_with_constants(self, constants):
        return Function(self.function.derive_with_constants(constants), [argument.derive_with_constants(constants) for argument in self.arguments], self.type, self.probability)

class Lambda(Program):
    def __init__(self, body, type_=UnknownType(), probability={}):
        self.body = body
        self.type = type_
        self.hash = hash(94135 + body.hash)

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        s = "(lambda " + format(self.body) + ")"
        return s

    def eval(self, dsl, environment, i):
        try:
            result = lambda x: self.body.eval(dsl, (x, environment), i)
            self.evaluation[i] = result
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            result = lambda x: self.body.eval_naive(dsl, (x, environment))
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError) as e:
            print(f"Error evaluating lambda: {e}")
            return None

class BasicPrimitive(Program):
    def __init__(self, primitive, type_=UnknownType(), probability={}, constant_evaluation=None):
        self.primitive = primitive
        self.type = type_
        self.is_a_constant = not isinstance(type_, Arrow) and primitive.startswith("constant")
        self.constant_evaluation = constant_evaluation
        self.hash = hash(primitive) + self.type.hash

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        if self.is_a_constant and self.constant_evaluation:
            return format(self.constant_evaluation)
        return format(self.primitive)

    def eval(self, dsl, environment, i):
        if self.is_a_constant and self.constant_evaluation:
            return self.constant_evaluation
        return dsl.semantics[self.primitive]

    def eval_naive(self, dsl, environment):
        if self.is_a_constant and self.constant_evaluation:
            return self.constant_evaluation
        return dsl.semantics[self.primitive]

    def count_constants(self):
        return 1 if self.is_a_constant else 0

    def derive_with_constants(self, constants):
        if self.is_a_constant:
            return BasicPrimitive(self.primitive, self.type, self.probability, constants.pop())
        else:
            return self


class New(Program):
    def __init__(self, body, type_=UnknownType(), probability={}):
        self.body = body
        self.type = type_
        self.hash = hash(783712 + body.hash) + type_.hash

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        return format(self.body)

    def eval(self, dsl, environment, i):
        try:
            result = self.body.eval(dsl, environment, i)
            self.evaluation[i] = result
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            result = self.body.eval_naive(dsl, environment)
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError) as e:
            print(f"Error evaluating function {self.function} with arguments {self.arguments}: {e}")
            return None

    def is_constant(self):
        return self.body.is_constant()

    def count_constants(self):
        return self.body.count_constants()

    def derive_with_constants(self, constants):
        return New(self.body.derive_with_constants(constants), self.type, self.probability)
