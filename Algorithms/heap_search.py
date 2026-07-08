# This file contains code derived from:
# - https://github.com/nathanael-fijalkow/DeepSynth (MIT License)
# Original authors: Nathanaël Fijalkow
from collections import deque
from heapq import heappush, heappop

from program import Program, Function, Variable
from grammar.pcfg import PCFG

def heap_search(G: PCFG):
    H = heap_search_object(G)
    return H.generator()

class heap_search_object:
    def return_unique(self, P):
        """Canonicalize: same program object across heaps so we don't re-evaluate it."""
        hash_P = P.hash
        if hash_P in self.hash_table_global:
            return self.hash_table_global[hash_P]
        else:
            self.hash_table_global[hash_P] = P
            return P

    def __init__(self, G: PCFG):
        self.current = None

        self.G = G
        self.start = G.start
        self.rules = G.rules
        self.symbols = [S for S in self.rules]

        # heaps[S]: programs generable from non-terminal S, keyed by negative probability.
        # A program may appear in multiple heaps but never twice in the same one.
        self.heaps = {S: [] for S in self.symbols}

        # succ[S][P] caches the successor of P starting at S.
        self.succ = {S: {} for S in self.symbols}

        # hash_table_program[S]: hashes of every program ever pushed into heaps[S].
        self.hash_table_program = {S: set() for S in self.symbols}

        # hash_table_global[h] -> the canonical Program object for that hash.
        self.hash_table_global = {}

        self.G.compute_max_probability()

        # Seed each heap with P(max(S1), max(S2), ...) for every rule S -> P(S1, S2, ...).
        for S in reversed(self.rules):
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                program = self.G.max_probability[(S, P)]
                hash_program = program.hash

                assert hash_program not in self.hash_table_program[S]

                self.hash_table_program[S].add(hash_program)
                self.hash_table_global[hash_program] = program

                heappush(
                    self.heaps[S],
                    (-program.probability[(self.G.hash, S)], program),
                )

        # Prime caches by querying each non-terminal from leaves to root.
        for S in reversed(self.rules):
            self.query(S, None)

    def generator(self):
        """Yield programs in decreasing probability order."""
        while True:
            program = self.query(self.start, self.current)
            self.current = program
            yield program

    def query(self, S, program):
        """Return the successor of `program` from non-terminal S."""
        if program:
            hash_program = program.hash
        else:
            hash_program = 123891

        if hash_program in self.succ[S]:
            return self.succ[S][hash_program]

        try:
            _, succ = heappop(self.heaps[S])
        except:
            return

        self.succ[S][hash_program] = succ

        if isinstance(succ, Function):
            F = succ.function

            for i in range(len(succ.arguments)):
                S2 = self.G.rules[S][F][0][i]
                succ_sub_program = self.query(S2, succ.arguments[i])

                if isinstance(succ_sub_program, Program):
                    new_arguments = succ.arguments[:]
                    new_arguments[i] = succ_sub_program

                    new_program = Function(
                        F, new_arguments, type_=succ.type, probability={}
                    )
                    new_program = self.return_unique(new_program)
                    hash_new_program = new_program.hash

                    if hash_new_program not in self.hash_table_program[S]:
                        self.hash_table_program[S].add(hash_new_program)
                        probability = self.G.rules[S][F][1]
                        for arg, S3 in zip(new_arguments, self.G.rules[S][F][0]):
                            probability *= arg.probability[(self.G.hash, S3)]
                        heappush(self.heaps[S], (-probability, new_program))
                        new_program.probability[(self.G.hash, S)] = probability

        # Variables have no successor.
        if isinstance(succ, Variable):
            return succ

        return succ
