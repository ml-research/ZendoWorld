rule_conversion_prompt = """Convert the following hypothesis, from a different DSL about Zendo structures into a DSL representation using the provided syntax.
Output only the DSL expression without any additional text. If the rule cannot be represented with the DSL, return something that is similar.

Hypothesis: {hs}

DSL Syntax:
Use S-expressions: (OP arg1 arg2 ...) with prefix notation.
A complete output must be a single DSL expression.

--------------------------------------------------
Unary predicates (no arguments)
--------------------------------------------------
Colors:
- IS_RED
- IS_BLUE
- IS_YELLOW

Shapes:
- IS_BLOCK
- IS_WEDGE
- IS_PYRAMID

Grounding:
- IS_GROUNDED
- IS_UNGROUNDED

Orientation:
- IS_UPRIGHT
- IS_UPSIDE_DOWN
- IS_VERTICAL
- IS_FLAT
- IS_DOORSTOP
- IS_CHEESECAKE

--------------------------------------------------
Interaction predicates
(take 2 unary predicates; return an interaction predicate)
--------------------------------------------------
- (TOUCHING P1 P2)
- (ON_TOP_OF P1 P2)
- (POINTING P1 P2)

--------------------------------------------------
Logical rule combinators
--------------------------------------------------
- (AND R1 R2)
- (OR R1 R2)

--------------------------------------------------
Global rules (no arguments)
--------------------------------------------------
- (ALL_THREE_COLORS)
- (ALL_THREE_SHAPES)
- (EVEN)          ; total number of pieces is even
- (ODD)           ; total number of pieces is odd

--------------------------------------------------
Count rules over unary predicates
(INT n ∈ (1,2,3,4,5,6,7))
--------------------------------------------------
- (AT_LEAST_1 n P)
- (EXACTLY_1 n P)
- (ZERO_1 P)
- (EVEN_1 P)      ; nonzero even count
- (ODD_1 P)       ; odd count
- (EXCLUSIVELY P) ; all pieces satisfy P
- (ALL_1 P)       ; all pieces satisfy P
- (MAJORITY_1 P)  ; at least half of pieces satisfy P

--------------------------------------------------
Count rules over conjunction on the same piece
(INT n ∈ (1,2,3,4,5,6,7))
--------------------------------------------------
- (AT_LEAST_2 n P1 P2)
- (EXACTLY_2 n P1 P2)
- (ZERO_2 P1 P2)
- (EVEN_2 P1 P2)
- (ODD_2 P1 P2)
- (ALL_2 P1 P2)        ; all pieces satisfy (P1 AND P2)
- (MAJORITY_2 P1 P2)   ; at least half satisfy (P1 AND P2)

--------------------------------------------------
Count rules over interactions
(INT n ∈ (1,2,3,4,5,6,7))
--------------------------------------------------
- (AT_LEAST_INTERACTION n (REL P1 P2))
- (EXACTLY_INTERACTION n (REL P1 P2))
- (EVEN_INTERACTION (REL P1 P2))
- (ODD_INTERACTION (REL P1 P2))
- (MAJORITY_INTERACTION (REL P1 P2))

--------------------------------------------------
Other numeric / comparison rules
--------------------------------------------------
- (LENGTH n)                 ; total number of pieces is exactly n
- (EITHER_OR n1 n2)          ; total number of pieces is n1 or n2
- (MORE_THAN P1 P2)          ; count(P1) > count(P2)
- (MORE_OR_EQUAL_THAN P1 P2) ; count(P1) ≥ count(P2)
- (SAME_AMOUNT P1 P2)        ; count(P1) == count(P2) and > 0

--------------------------------------------------
Examples / templates
--------------------------------------------------
Single attribute:
- “at least 3 wedges”:
  (AT_LEAST_1 3 IS_WEDGE)

Two attributes on same piece:
- “exactly 2 red blocks”:
  (EXACTLY_2 2 IS_RED IS_BLOCK)

Interaction count:
- “at least one blue on top of a yellow”:
  (AT_LEAST_INTERACTION 1 (ON_TOP_OF IS_BLUE IS_YELLOW))

Majority:
- “most pieces are red”:
  (MAJORITY_1 IS_RED)

Comparisons:
- “more red pieces than blue pieces”:
  (MORE_THAN IS_RED IS_BLUE)

Combine clauses:
- (AND R1 R2)
- (OR R1 R2)

--------------------------------------------------

Return only the DSL expression below inside a single code block, like this:
```python
(...)
```
"""
