# Prompts for zendo/player.py players

# ---------------------------------------------------------------------------
# GPTQueryZendoPlayer – text-only structure proposal
# ---------------------------------------------------------------------------

query_structure_prompt = """You are a Zendo player. Your job is to generate a new structure example to gain new knowledge about the hidden rule.
You are given a few positive and negative examples. Each structure consists of a list of items with the format:
"item(ID, color, shape, orientation, interaction)".

The pieces can have colors: red, blue, yellow; shapes: block, wedge, pyramid; orientations: upright, upside_down, flat, cheesecake, doorstop.
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.

The current top rule hypotheses are:
{top_rules}

Positive examples:
{positives}

Negative examples:
{negatives}

Please ONLY return a new example and its label within a python block, in this exact format, where label is 1 for valid and 0 for invalid:
The label should be your best guess based on the rules above, but it does not have to be correct.
Here are examples of valid formats:
[["item(0, red, block, upright, grounded)", "item(1, blue, wedge, doorstop, touching(0))"], 1]
[["item(0, yellow, pyramid, upside_down, grounded)", "item(1, blue, wedge, doorstop, on_top_of(0))", "item(2, red, block, upright, pointing(1))"], 0]
Return your answer in this exact format including the python block:
```python
[["item(ID, color, shape, orientation, interaction)", ...], label]
```
"""

# ---------------------------------------------------------------------------
# FullGPTZendoPlayer – propose structure
# ---------------------------------------------------------------------------

propose_structure_images_header = """You are a Zendo player. Your goal is to **gain new information** about the hidden rule by proposing a \
**novel** structure that is **maximally informative** (highly likely to change or confirm current beliefs).

You are given *visual* positive and negative examples. Study the images, but output your proposal as **text**.
The pieces can have
- colors: red, blue, yellow;
- shapes: block, wedge, pyramid;
- orientations: upright, upside_down, flat, cheesecake, doorstop.
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.

DO NOT include explanations, reasoning, comments, or text outside the python block, in this exact format, where label is 1 for valid and 0 for invalid:
Here are examples of valid answers:
```python
[["item(0, red, block, upright, grounded)", "item(1, blue, wedge, doorstop, touching(0))"], 1]
```
```python
[["item(0, yellow, pyramid, upside_down, grounded)", "item(1, blue, wedge, doorstop, on_top_of(0))", "item(2, red, block, upright, pointing(1))"], 0]
```
Return your answer in this exact format including the python block:
```python
[["item(ID, color, shape, orientation, interaction)", ...], label]
```
"""

propose_structure_text_prompt = """You are a Zendo player. Your job is to generate a new structure example to gain new knowledge about the hidden rule.
The pieces can have
- colors: red, blue, yellow;
- shapes: block, wedge, pyramid;
- orientations: upright, upside_down, flat, cheesecake, doorstop.
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.

You are given a few positive and negative examples. Each structure consists of a list of items with the format:
    "item(ID, color, shape, orientation, interaction)".

Positive examples:
{positives}

Negative examples:
{negatives}

Please ONLY return a new example and its label within a python block, in this exact format, where label is 1 for valid and 0 for invalid:
```python
[["item(ID, color, shape, orientation, interaction)", ...], label]
```
"""

# ---------------------------------------------------------------------------
# FullGPTZendoPlayer – guess rule
# ---------------------------------------------------------------------------

_DSL_INSTRUCTIONS = """
**Available values:**
- Colors: red, blue, yellow
- Shapes: block, wedge, pyramid
- Orientations: upright, upside_down, flat, cheesecake, doorstop, vertical
- Interactions: grounded, touching, pointing, on_top_of

**Important constraints:**
- Wedges are never flat (only doorstop or cheesecake).
- Blocks and pyramids can be flat but not cheesecake or doorstop.
- grounded/ungrounded can only be used with single attributes, like: at_least_interaction(Attribute, Interaction, N, Structure)
- touching/pointing/on_top_of can only be used with two attributes, like: at_least_interaction(Attribute1, Attribute2, Interaction, N, Structure)

**Goal:**
Find **one** Prolog-style rule that is **True for all positive examples** and **False for all negative examples**.

**Critical rules for selecting your answer:**
1. **If a single predicate works, use it.**
   Do NOT combine predicates unless a single one cannot fully separate positives from negatives.
2. The rule must be as short and simple as possible.
3. Return **only** the rule — no explanation, no formatting, no extra text.

The Rule must be written in **Prolog-style syntax**, using only the following predicates.
Use **only the following Prolog-compatible predicates** and regard the argument types:

### Count-based:
- `at_least(Attribute, N, Structure)`
- `at_least(Attribute1, Attribute2, N, Structure)`
- `exactly(Attribute, N, Structure)`
- `exactly(Attribute1, Attribute2, N, Structure)`
- `zero(Attribute, Structure)`
- `zero(Attribute1, Attribute2, Structure)`
- `more_than(Attribute1, Attribute2, Structure)`

### Parity:
- `odd_number_of(Structure)`
- `even_number_of(Structure)`
- `odd_number_of(Attribute, Structure)`
- `odd_number_of(Attribute1, Attribute2, Structure)`
- `even_number_of(Attribute, Structure)`
- `even_number_of(Attribute1, Attribute2, Structure)`

### Interaction-based:
- `at_least_interaction(Attribute, Interaction_g, N, Structure)`
- `at_least_interaction(Attribute1, Attribute2, Interaction, N, Structure)`
- `exactly_interaction(Attribute, Interaction_g, N, Structure)`
- `exactly_interaction(Attribute1, Attribute2, Interaction, N, Structure)`
- `odd_number_of_interaction(Attribute, Interaction_g, Structure)`
- `odd_number_of_interaction(Attribute1, Attribute2, Interaction, Structure)`
- `even_number_of_interaction(Attribute, Interaction_g, Structure)`
- `even_number_of_interaction(Attribute1, Attribute2, Interaction, Structure)`

### Other:
- `exclusively(Attribute, Structure)`
- `either_or(N1, N2, Structure)`
- `all_three_shapes(Structure)`
- `all_three_colors(Structure)`

### Logical:
- `and([Rule1, Rule2])` — use only if no single predicate works and not in combination with or([Rule1, Rule2]) AND ONLY ONCE PER RULE
- `or([Rule1, Rule2])` — use only if no single predicate works and not in combination with and([Rule1, Rule2]) AND ONLY ONCE PER RULE

### Attribute Constants:
Attributes must be lowercase and drawn from:
- Colors: `red`, `blue`, `yellow`
- Shapes: `block`, `wedge`, `pyramid`
- Orientations: `upright`, `upside_down`, `flat`, `cheesecake`, `doorstop`, `vertical`

### Interactions:
- Interaction: `touching`, `pointing`, `on_top_of`
- Interaction_g: `grounded`, `ungrounded`
Note: `grounded` and `ungrounded` can only be used in the interaction predicates with just one attribute argument, noted as Interaction_g, while the others can only be used with two attributes, noted as Interaction.

### Explanations:
N, N1, N2 is a placeholder for a natural numbers in [1, 2, 3].
odd_number_of(Structure) = There exists an odd number of elements in the Structure.
odd_number_of(red, upright, Structure) = There exists an odd number of elements in the Structure that are red and upright.
odd_number_of_interaction(red, grounded, Structure) = There exists an odd number of elements in the Structure that are red and grounded.
odd_number_of_interaction(red, upright, touching, Structure) = There exists an odd number of elements in the Structure that are red touching an upright piece.
-> even, at_least, exactly, zero work similarly
more_than(Attribute1, Attribute2, Structure) = There exists more than Attribute1 elements in the Structure that are Attribute2.
either_or(N1, N2, Structure) = There exists either N1 or N2 elements in the Structure.

### Output Format:
Return **only** a single rule in Prolog-style syntax. Do **not** include explanations or extra text.

### Example:
```prolog
even_number_of(upright, Structure)
```
"""

_NL_INSTRUCTIONS = """
**Available values:**
- Colors: red, blue, yellow
- Shapes: block, wedge, pyramid
- Orientations: upright, upside_down, flat, cheesecake, doorstop, vertical
- Interactions: grounded, touching, pointing, on_top_of, groundedness

**Goal:**
Find **one** rule that is **True for all positive examples** and **False for all negative examples**.

**Critical rules for selecting your answer:**
1. You may combine short rules using "and" or "or" if needed, but **if a single predicate works, use it.**
2. The rule must be as short and simple as possible but still accurate.
3. Return **only** the rule — no explanation, no formatting, no extra text.
4. Do not use any conditionals ("if", "when", "only if", etc.) or any text outside the rule itself.

### Output Format:
Return **only** a single rule in natural language. Do **not** include explanations or extra text.
"""

# Header for guess-rule with images, DSL mode
guess_rule_dsl_images_header = (
    "You are a Zendo player. Your job is to find a new logical classification rule for given examples with labels.\n"
    "You are given a few positive and negative examples. Each image consists of pieces in different configurations."
    + _DSL_INSTRUCTIONS
)

# Header for guess-rule with images, natural-language mode
guess_rule_nl_images_header = (
    "You are a Zendo player. Your job is to find a new logical classification rule for given examples with labels.\n"
    "You are given a few positive and negative examples. Each image consists of pieces in different configurations."
    + _NL_INSTRUCTIONS
)

# Template for guess-rule with text descriptions only (DSL mode)
guess_rule_dsl_text_prompt = (
    "You are a Zendo player. Your job is to find a new logical classification rule for given examples with labels.\n"
    "You are given a few positive and negative examples. Each image consists of pieces in different configurations.\n"
    "\nPositive examples:\n{positives}"
    "\n\nNegative examples:\n{negatives}"
    "\n\nPreviously guessed rules:\n{previous_guesses}"
    "\n\nDo NOT return any of the previous guesses.\n"
    + _DSL_INSTRUCTIONS
)

# Template for guess-rule with text descriptions only (DSL mode)
guess_rule_nl_text_prompt = (
    "You are a Zendo player. Your job is to find a new logical classification rule for given examples with labels.\n"
    "You are given a few positive and negative examples. Each image consists of pieces in different configurations.\n"
    "\nPositive examples:\n{positives}"
    "\n\nNegative examples:\n{negatives}"
    "\n\nPreviously guessed rules:\n{previous_guesses}"
    "\n\nDo NOT return any of the previous guesses.\n"
    + _NL_INSTRUCTIONS
)

# ---------------------------------------------------------------------------
# GPTVisionModel – piece detection
# ---------------------------------------------------------------------------

detect_pieces_header = """You are a Zendo Vision Model. Your goal is to detect all pieces in the image, and return a list of string descriptions for each piece.

Study the image, but output your findings as **text**.
For each piece you find, give it a unique ID starting from 0, and describe its color, shape, orientation, and interaction with other pieces. For the interactions, use the IDs of the other pieces.
The pieces can have
- colors: red, blue, yellow;
- shapes: block, wedge, pyramid;
- orientations: upright, upside_down, flat, cheesecake, doorstop.
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.

Rules for interactions:
- grounded means the piece is touching the ground and is the default value.
- touching(ID) means this piece is in contact with the piece with ID on either side except top or bottom.
- pointing(ID) means this piece is pointing to the piece with ID meaning the piece is flat or cheesecake or doorstop and the head of the piece is pointing to the other piece.
- on_top_of(ID) means this piece is resting on top of the piece with ID.

Please ONLY return the descriptions within a python block, in this exact format:
Here are examples of valid formats:
- ["item(0, red, block, upright, grounded)", "item(1, blue, wedge, doorstop, touching(0))"]
- ["item(0, yellow, pyramid, upside_down, grounded)", "item(1, blue, wedge, doorstop, on_top_of(0))", "item(2, red, block, upright, pointing(1))"]
Return your answer in this exact format including the python block:
```python
["item(ID, color, shape, orientation, interaction)", ...]
```
"""
