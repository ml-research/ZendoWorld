# This file contains code derived from:
# - https://github.com/nathanael-fijalkow/DeepSynth (MIT License)
# Original authors: Nathanaël Fijalkow
from program import *
from grammar.pcfg import *

from collections import deque
from heapq import heappush, heappop
import time 

def bounded_threshold(G: PCFG, threshold):
    """Yield every program with probability above ``threshold``."""
    # Frontier entries are (partial_program, non_terminals_queue, probability).
    frontier = deque()
    initial_non_terminals = deque()
    initial_non_terminals.append(G.start)
    frontier.append((None, initial_non_terminals, 1))

    while len(frontier) != 0:
        partial_program, non_terminals, probability = frontier.pop()
        if len(non_terminals) == 0:
            yield partial_program
        else:
            S = non_terminals.pop()
            for P in G.rules[S]:
                args_P, w = G.rules[S][P]
                new_probability = probability * w
                if new_probability > threshold:
                    new_partial_program = (P, partial_program)
                    new_non_terminals = non_terminals.copy()
                    for arg in args_P:
                        new_non_terminals.append(arg)
                    frontier.append((new_partial_program, new_non_terminals, new_probability))


def threshold_search(G: PCFG, initial_threshold=1e-4, scale_factor=5e2):
    threshold = initial_threshold
    gen = bounded_threshold(G, threshold)

    while True:
        try:
            yield next(gen)
        except StopIteration:
            threshold /= scale_factor
            gen = bounded_threshold(G, threshold)
    
