# Prompts for src/zendoworld/game/player.py players

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

Please ONLY return a new example and its label within a python block, in this exact format, where label is 1 for "rule following" and 0 for "rule breaking":
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

DO NOT include explanations, reasoning, comments, or text outside the python block, in this exact format, where label is 1 for "rule following" and 0 for "rule breaking":
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

Please ONLY return a new example and its label within a python block, in this exact format, where label is 1 for "rule following" and 0 for "rule breaking":
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
    "You are given a few positive and negative examples. Each structure consists of a list of items with the format: "
    "\"item(ID, color, shape, orientation, interaction)\".\n"
    "\nPositive examples:\n{positives}"
    "\n\nNegative examples:\n{negatives}"
    "\n\nPreviously guessed rules:\n{previous_guesses}"
    "\n\nDo NOT return any of the previous guesses.\n"
    + _DSL_INSTRUCTIONS
)

# Template for guess-rule with text descriptions only (DSL mode)
guess_rule_nl_text_prompt = (
    "You are a Zendo player. Your job is to find a new logical classification rule for given examples with labels.\n"
    "You are given a few positive and negative examples. Each structure consists of a list of items with the format: "
    "\"item(ID, color, shape, orientation, interaction)\".\n"
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


# ---------------------------------------------------------------------------
# VLPWeightedZendoPlayer – training-time and test-time prompts
# ---------------------------------------------------------------------------

vlp_weighted_attribute_discovery_prompt = """You are augmenting the vocabulary of a Zendo DSL by scanning task scenes for novel attributes.

Existing vocabulary (do NOT repeat any of these):
- objects: {existing_objects}
- properties: {existing_properties}
- actions: {existing_actions}

Below are positive (rule-satisfying) and negative (rule-violating) scenes. Each scene is a list of items in the form `item(ID, color, shape, orientation, interaction)`.

{scene_block}

List any NEW attributes you see that are not already in the vocabulary above. Use snake_case for all names. Only include attributes you actually observe in the scenes. If nothing new is present, return empty lists.

Return ONLY a JSON object inside a fenced ```json block, in this exact shape:
```json
{{
  "objects": ["..."],
  "properties": ["..."],
  "actions": ["..."]
}}
```
"""


vlp_weighted_macro_extraction_prompt = """You are distilling a Zendo synthesis result into a reusable named macro.

Available primitives (only these names may appear in the macro body):
{primitive_list}

Existing macros (do NOT redefine these names):
{existing_macros}

A heap-search on task {task_id} found this deeply nested program:
PROGRAM: {program}
TYPE:    {program_type}

Abstract this program into a single named macro. The macro body MUST be a Python lambda that uses ONLY the primitive names listed above (no Python builtins). Pick a descriptive snake_case name.

Return ONLY a JSON object inside a fenced ```json block:
```json
{{
  "name": "snake_case_name",
  "body": "lambda x: ... using only the listed primitives ...",
  "return_type": "BOOL",
  "arg_types": ["IMG"],
  "provenance": "short note on what this captures"
}}
```
"""


vlp_weighted_macro_fix_prompt = """Your previous response was rejected.

ERROR: {error}

PREVIOUS RESPONSE:
{previous}

Fix the issue and return ONLY the corrected JSON block. Do not include any other text.
"""


vlp_weighted_predicate_extraction_prompt = """You are proposing a new REUSABLE predicate for the symbolic VLP-Zendo DSL (attribute-based variant) because heap-search could not find a perfect program for this task.

# Reusability is the #1 requirement
You are NOT being asked to encode this task's rule directly. You are asked to add a missing GENERAL building block that, combined with the existing primitives via heap-search, will let synthesis find the rule for this task — AND be useful for other tasks too.

A good predicate is parameterised by typed atoms (OBJECT / PROPERTY / ACTION) and captures a pattern that appears in many Zendo rules. Examples of good general patterns: "every piece shares a given property", "two pieces have the same shape", "every triple has matching subject and object shape", "at least one piece has both attributes A and B", "the action involves a given object on both sides".

A BAD predicate hardcodes specific vocabulary atoms in its body (e.g. `'yellow' in item`), names itself after a specific rule (`all_yellow_pyramids_are_stable`), or essentially IS the task rule. The trainer will REJECT any predicate body that contains a vocabulary atom as a literal string — those must be CURRIED OBJECT / PROPERTY / ACTION arguments instead.

# DSL interface — READ CAREFULLY
Your predicate composes with the existing primitives. Its first argument is the OUTPUT of `get_objects` OR `get_actions` — NEVER a raw scene string.

## get_objects output: `List(List(STRING))`
One inner list per piece. Each inner list contains ALL attributes of that piece (shape, color, orientation, interaction tag) as plain strings, with NO positional convention. Predicates check membership only — never index by position.
Example:
  [['red', 'block', 'upright', 'grounded'],
   ['blue', 'wedge', 'doorstop', 'touching']]
A scene with no pieces is `[[]]`. A piece with only a shape attribute is `['block']`.

## get_actions output: `List(List(List(STRING)))`
One inner triple per real pairwise interaction. Each triple is
  `[src_attrs, [action_name], tgt_attrs]`
where `src_attrs` and `tgt_attrs` are the FULL attribute lists of the participating pieces (lists of strings), and `[action_name]` is a single-element list naming the interaction.
Example:
  [[['red', 'pyramid', 'upright'], ['touching'], ['blue', 'block', 'upright']],
   [['blue', 'wedge', 'doorstop'], ['on_top_of'], ['red', 'block', 'upright']]]
A scene with no interactions is `[[]]`.

## Additional predicate args
After the first scene argument, additional args (if any) must be typed atoms drawn from the library vocabulary:
  - `OBJECT`   → one of the OBJECTS (shape strings)
  - `PROPERTY` → one of the PROPERTIES (color / orientation strings)
  - `ACTION`   → one of the ACTIONS (verb strings)
At runtime these atoms are just plain strings; the body checks membership against the attribute / action lists.

# Test scenes — your body MUST run without raising on EVERY scene of the shape you chose
The trainer compiles your body and runs it against all of these. ANY exception → REJECTED.

## Object-shape test scenes (use if your first arg is `List(List(STRING))`):
  O1 = [['red', 'block', 'upright', 'grounded']]
  O2 = [['red', 'block', 'upright', 'grounded'], ['blue', 'wedge', 'flat', 'touching']]
  O3 = [['yellow', 'pyramid', 'upside_down', 'grounded']] * 3
  O4 = [['blue', 'wedge', 'cheesecake', 'grounded'], ['red', 'block', 'upright', 'grounded']]
  O5 = [['block']]
  O6 = [[]]

## Action-shape test scenes (use if your first arg is `List(List(List(STRING)))`):
  A1 = [[['red', 'pyramid', 'upright'], ['touching'], ['blue', 'block', 'upright']],
        [['blue', 'block', 'upright'], ['touching'], ['blue', 'block', 'upright']]]
  A2 = [[['blue', 'wedge', 'doorstop'], ['on_top_of'], ['red', 'block', 'upright']]]
  A3 = [[['yellow', 'pyramid'], ['pointing'], ['blue', 'wedge']]]
  A4 = [[]]

Pick ONE shape. Defend your body against:
  * the empty marker (`[[]]`)
  * pieces / triples with short attribute lists (e.g. `['block']`, `['yellow', 'pyramid']`)
  * any number of pieces / triples

# Available primitives (for context — your body cannot call them, only Python builtins):
{primitive_list}

# Existing library
Existing macros: {existing_macros}
Existing predicates (do NOT redefine): {existing_predicates}

# Vocabulary (use these exact strings for OBJECT / PROPERTY / ACTION atoms):
- objects: {existing_objects}
- properties: {existing_properties}
- actions: {existing_actions}

# Task evidence (scenes from the failing task)
{scene_block}

# Best partial program found (accuracy {best_accuracy:.2f})
{best_program}

# Constraints on the body
- The body MUST be a Python lambda. CURRY multiple args: `lambda items: lambda obj: lambda prop: ...`.
- Allowed builtins ONLY: len, sum, any, all, max, min, range, abs, sorted, filter, map, zip, enumerate, isinstance, list, dict, set, tuple, str, int, bool, float.
- NO imports, NO dunders, NO IO, NO string indexing of items (items are LISTS, not strings).
- Use membership checks (`obj in attrs`), NOT positional access. Pieces / triples may have variable length.
- Action triples have fixed length 3 — you may use `t[0]`, `t[1]`, `t[2]` for src_attrs, [action], tgt_attrs respectively, but the inner attribute lists are variable-length so use membership inside them.
- HARD RULE: No vocabulary atom may appear as a quoted literal in the body. Any concrete shape / color / orientation / interaction name (`'block'`, `'wedge'`, `'pyramid'`, `'red'`, `'blue'`, `'yellow'`, `'upright'`, `'upside_down'`, `'flat'`, `'cheesecake'`, `'doorstop'`, `'touching'`, `'pointing'`, `'on_top_of'`, `'grounded'`, ...) MUST be a curried argument, not a literal. The validator scans for this and rejects on a hit.
- The predicate NAME must describe the GENERAL pattern, not the specific instance. `all_pieces_share_property` GOOD, `all_yellow_pyramids_are_stable` BAD. Names containing a vocabulary atom are rejected.

# Output format
Return ONLY a JSON object inside a fenced ```json block:
```json
{{
  "name": "snake_case_name",
  "body": "lambda items: ...",
  "return_type": "BOOL",
  "arg_types": ["List(List(STRING))"],
  "provenance": "why this predicate is needed"
}}
```

# Worked examples (all REUSABLE — none mentions a specific atom in body or name)

## Object-shape predicate: "every piece has BOTH the given object-shape and the given property"
```json
{{
  "name": "all_pieces_match_object_and_property",
  "body": "lambda items: lambda obj: lambda prop: all(isinstance(it, list) and obj in it and prop in it for it in items if it)",
  "return_type": "BOOL",
  "arg_types": ["List(List(STRING))", "OBJECT", "PROPERTY"],
  "provenance": "captures uniform conjunctive constraint over all pieces"
}}
```

## Object-shape predicate: "at least two pieces share the given property"
```json
{{
  "name": "at_least_two_share_property",
  "body": "lambda items: lambda prop: sum(1 for it in items if isinstance(it, list) and prop in it) >= 2",
  "return_type": "BOOL",
  "arg_types": ["List(List(STRING))", "PROPERTY"],
  "provenance": "reusable pluralisation pattern — applies to any color / orientation"
}}
```

## Object-shape predicate: "all pieces of the given shape share at least one common property"
```json
{{
  "name": "all_of_object_share_some_property",
  "body": "lambda items: lambda obj: (lambda matching: (not matching) or any(all(p in it for it in matching) for p in matching[0]))([it for it in items if isinstance(it, list) and obj in it])",
  "return_type": "BOOL",
  "arg_types": ["List(List(STRING))", "OBJECT"],
  "provenance": "captures 'all Xs agree on something' without naming WHAT they agree on"
}}
```

## Action-shape predicate: "at least one triple has src with given shape and tgt with given shape, under given action"
```json
{{
  "name": "any_triple_src_obj_action_tgt_obj",
  "body": "lambda triples: lambda src_obj: lambda act: lambda tgt_obj: any(isinstance(t, list) and len(t) == 3 and src_obj in t[0] and act in t[1] and tgt_obj in t[2] for t in triples)",
  "return_type": "BOOL",
  "arg_types": ["List(List(List(STRING)))", "OBJECT", "ACTION", "OBJECT"],
  "provenance": "directional relation: shape A acting on shape B"
}}
```

## Action-shape predicate: "every interaction's source and target share the given attribute"
```json
{{
  "name": "all_triples_src_tgt_share_attr",
  "body": "lambda triples: lambda prop: all(isinstance(t, list) and len(t) == 3 and prop in t[0] and prop in t[2] for t in triples) if any(isinstance(t, list) and len(t) == 3 for t in triples) else True",
  "return_type": "BOOL",
  "arg_types": ["List(List(List(STRING)))", "PROPERTY"],
  "provenance": "matching-attribute relational pattern; vacuously true when no interactions"
}}
```
"""


vlp_weighted_predicate_fix_prompt = """Your previous predicate proposal was rejected.

ERROR: {error}

PREVIOUS RESPONSE:
{previous}

Fix the issue and return ONLY the corrected JSON block. Do not include any other text.
"""


vlp_weighted_intuition_text_prompt = """You are a Zendo intuition oracle. Your job is to write ONE specific hypothesis about the hidden rule and then select the MINIMAL set of DSL primitives and vocabulary atoms needed to express it.

# Scenes
Each scene is a list of items: `item(ID, color, shape, orientation, interaction)`

{scene_block}

# Step 1 — formulate ONE specific hypothesis (this becomes global_hints)
Study positive vs. negative scenes and commit to the single most likely rule. Be concrete:
- Name the EXACT attribute (color/shape/orientation) or interaction type involved.
- State whether it is a count (how many?), a parity (odd/even?), or an interaction pattern.
- Example of a good hint: "All positive scenes have at least 2 blue pieces."
- Example of a BAD hint: "The rule may involve colors or shapes or interactions." (too vague — pick ONE)

# Step 2 — pick ONLY the primitives needed for that hypothesis
Rules:
- Pick AT MOST 3 primitives total. If you can express the hypothesis with 1 or 2, use fewer.
- ONLY include a primitive if your Step 1 hypothesis directly requires it.
- If your hypothesis does NOT mention interactions → do NOT include any `count_action_*` primitive.
- If your hypothesis does NOT mention parity → do NOT include `odd?` or `even?`.
- If your hypothesis mentions a single threshold count → include exactly one count primitive + one comparator.

Primitive reference (EXACT names only):

**Object-counting** (use when hypothesis is about pieces, not interactions):
- `count_all_objects` — total pieces in scene
- `count_object` — pieces with a given SHAPE
- `count_property` — pieces with a given COLOR or ORIENTATION
- `count_object_with_property` — pieces with a given SHAPE + COLOR/ORIENTATION
- `count_properties` — pieces with two given COLORS/ORIENTATIONS simultaneously
- `count_object_with_two_properties` — pieces with SHAPE + two PROPERTIES

**Interaction-counting** (use ONLY when hypothesis is about touching/pointing/on_top_of):
- `count_all_actions` — total pairwise interactions
- `count_action` — interactions of a specific TYPE
- `count_action_src_object` — interactions where SOURCE has given SHAPE
- `count_action_src_property` — interactions where SOURCE has given COLOR/ORIENTATION
- `count_action_tgt_object` — interactions where TARGET has given SHAPE
- `count_action_tgt_property` — interactions where TARGET has given COLOR/ORIENTATION
- `count_action_typed_object` — interactions where BOTH pieces have given SHAPES, under given action
- `count_action_typed_property` — interactions where BOTH pieces have given PROPERTIES, under given action

**Comparators / logic** (combine counts into booleans):
- `gt?` — count > N
- `eq?` — count = N
- `odd?` — count is odd and non-zero
- `even?` — count is even and non-zero
- `and` / `or` — combine two conditions (use only if hypothesis has two distinct sub-conditions)

Full primitive list (include library extensions if relevant): {primitive_list}

# Step 3 — pick ONLY the vocabulary atoms named in your hypothesis
Rules:
- Pick AT MOST 4 atoms total. If your hypothesis names only 1 attribute, list only that 1.
- DO NOT list an atom just because it appears somewhere in the scenes — only include atoms your hypothesis explicitly references.
- objects (shapes):                  {existing_objects}
- properties (colors/orientations):  {existing_properties}
- actions (interactions):            {existing_actions}

# Worked example
Hypothesis: "All positive scenes have at least 2 blue pieces."
→ focus_predicates: ["count_property", "gt?"]   (count_property counts blues; gt? checks ≥2)
→ focus_objects:    ["blue"]                     (only the atom the hypothesis names)
NOT: all counting primitives + all comparators + all colors.

# Output
Return ONLY a fenced JSON block:
```json
{{
  "global_hints": ["ONE specific sentence stating the hypothesised rule"],
  "focus_predicates": ["at most 3 exact primitive names"],
  "focus_objects": ["at most 4 exact atom strings"],
  "confidence": 0.0
}}
```
Set `confidence` to 0.0–0.4 if the pattern is unclear, 0.5–0.7 if moderately clear, 0.8–1.0 if obvious.
"""


# Visual variant — images are passed alongside this prompt via prompt_with_images().
# The first {n_positive} images are positive examples; the next {n_negative} are negative.
# No scene-description text is included because the model sees the images directly.
vlp_weighted_intuition_image_prompt = """You are a Zendo intuition oracle. Your job is to write ONE specific hypothesis about the hidden rule and then select the MINIMAL set of DSL primitives and vocabulary atoms needed to express it.

# Images provided
The first {n_positive} image(s) are POSITIVE examples (the hidden rule IS satisfied).
The next {n_negative} image(s) are NEGATIVE examples (the hidden rule is NOT satisfied).

Each image shows a set of coloured 3-D pieces (blocks, wedges, pyramids) in various orientations and spatial arrangements.

# Step 1 — formulate ONE specific hypothesis (this becomes global_hints)
Study positive vs. negative images and commit to the single most likely rule. Be concrete:
- Name the EXACT attribute (color/shape/orientation) or interaction type involved.
- State whether it is a count (how many?), a parity (odd/even?), or an interaction pattern.
- Example of a good hint: "All positive scenes have at least 2 blue pieces."
- Example of a BAD hint: "The rule may involve colors or shapes or interactions." (too vague — pick ONE)

# Step 2 — pick ONLY the primitives needed for that hypothesis
Rules:
- Pick AT MOST 3 primitives total. If you can express the hypothesis with 1 or 2, use fewer.
- ONLY include a primitive if your Step 1 hypothesis directly requires it.
- If your hypothesis does NOT mention interactions → do NOT include any `count_action_*` primitive.
- If your hypothesis does NOT mention parity → do NOT include `odd?` or `even?`.
- If your hypothesis mentions a single threshold count → include exactly one count primitive + one comparator.

Primitive reference (EXACT names only):

**Object-counting** (use when hypothesis is about pieces, not interactions):
- `count_all_objects` — total pieces in scene
- `count_object` — pieces with a given SHAPE
- `count_property` — pieces with a given COLOR or ORIENTATION
- `count_object_with_property` — pieces with a given SHAPE + COLOR/ORIENTATION
- `count_properties` — pieces with two given COLORS/ORIENTATIONS simultaneously
- `count_object_with_two_properties` — pieces with SHAPE + two PROPERTIES

**Interaction-counting** (use ONLY when hypothesis is about touching/pointing/on_top_of):
- `count_all_actions` — total pairwise interactions
- `count_action` — interactions of a specific TYPE
- `count_action_src_object` — interactions where SOURCE has given SHAPE
- `count_action_src_property` — interactions where SOURCE has given COLOR/ORIENTATION
- `count_action_tgt_object` — interactions where TARGET has given SHAPE
- `count_action_tgt_property` — interactions where TARGET has given COLOR/ORIENTATION
- `count_action_typed_object` — interactions where BOTH pieces have given SHAPES, under given action
- `count_action_typed_property` — interactions where BOTH pieces have given PROPERTIES, under given action

**Comparators / logic** (combine counts into booleans):
- `gt?` — count > N
- `eq?` — count = N
- `odd?` — count is odd and non-zero
- `even?` — count is even and non-zero
- `and` / `or` — combine two conditions (use only if hypothesis has two distinct sub-conditions)

Full primitive list (include library extensions if relevant): {primitive_list}

# Step 3 — pick ONLY the vocabulary atoms named in your hypothesis
Rules:
- Pick AT MOST 4 atoms total. If your hypothesis names only 1 attribute, list only that 1.
- DO NOT list an atom just because it appears somewhere in the images — only include atoms your hypothesis explicitly references.
- objects (shapes):                  {existing_objects}
- properties (colors/orientations):  {existing_properties}
- actions (interactions):            {existing_actions}

# Worked example
Hypothesis: "All positive scenes have at least 2 blue pieces."
→ focus_predicates: ["count_property", "gt?"]   (count_property counts blues; gt? checks ≥2)
→ focus_objects:    ["blue"]                     (only the atom the hypothesis names)
NOT: all counting primitives + all comparators + all colors.

# Output
Return ONLY a fenced JSON block:
```json
{{
  "global_hints": ["ONE specific sentence stating the hypothesised rule"],
  "focus_predicates": ["at most 3 exact primitive names"],
  "focus_objects": ["at most 4 exact atom strings"],
  "confidence": 0.0
}}
```
Set `confidence` to 0.0–0.4 if the pattern is unclear, 0.5–0.7 if moderately clear, 0.8–1.0 if obvious.
"""

# Same A/B-test role as vlp_weighted_intuition_image_prompt above, but for the
# zendo DSL (src/zendoworld/synthesis/dsl/zendo.py) instead of the VLP DSL: the primitive reference,
# Step 3 atom categories and worked example all refer to zendo's own
# predicate names (IS_RED, AT_LEAST_1, TOUCHING, ...) rather than the VLP
# count_*/comparator vocabulary.
zendo_weighted_intuition_image_prompt = """You are a Zendo intuition oracle. Your job is to write ONE specific hypothesis about the hidden rule and then select the MINIMAL set of DSL primitives needed to express it.

# Images provided
The first {n_positive} image(s) are POSITIVE examples (the hidden rule IS satisfied).
The next {n_negative} image(s) are NEGATIVE examples (the hidden rule is NOT satisfied).

Each image shows a set of coloured 3-D pieces (blocks, wedges, pyramids) in various orientations and spatial arrangements.

# Step 1 — formulate ONE specific hypothesis (this becomes global_hints)
Study positive vs. negative images and commit to the single most likely rule. Be concrete:
- Name the EXACT attribute (color/shape/orientation) or interaction type involved.
- State whether it is a count (how many?), a parity (odd/even?), or an interaction pattern.
- Example of a good hint: "All positive scenes have at least 2 blue pieces."
- Example of a BAD hint: "The rule may involve colors or shapes or interactions." (too vague — pick ONE)

# Step 2 — pick ONLY the primitives needed for that hypothesis
Rules:
- Pick AT MOST 3 primitives total. If you can express the hypothesis with 1 or 2, use fewer.
- ONLY include a primitive if your Step 1 hypothesis directly requires it.
- If your hypothesis does NOT mention interactions → do NOT include `TOUCHING`, `POINTING`, `ON_TOP_OF`, or any `*_INTERACTION` primitive.
- If your hypothesis does NOT mention parity → do NOT include `EVEN*` / `ODD*`.
- If your hypothesis mentions a single threshold count → include exactly one `AT_LEAST_*` or `EXACTLY_*` primitive.

Primitive reference (EXACT names only):

**Basic piece predicates** (use directly, no extra arguments needed):
- `IS_RED`, `IS_BLUE`, `IS_YELLOW` — piece color
- `IS_BLOCK`, `IS_WEDGE`, `IS_PYRAMID` — piece shape
- `IS_UPRIGHT`, `IS_UPSIDE_DOWN`, `IS_FLAT`, `IS_CHEESECAKE`, `IS_DOORSTOP`, `IS_VERTICAL` — piece orientation
- `IS_GROUNDED`, `IS_UNGROUNDED` — whether the piece touches the table

**Counting / threshold rules** (combine a count with a comparison):
- `AT_LEAST_1` — at least N pieces satisfy a predicate
- `AT_LEAST_2` — at least N pieces jointly satisfy two predicates
- `EXACTLY_1` / `EXACTLY_2` — exactly N pieces, same shape as above
- `EVEN_1` / `ODD_1` — an even/odd nonzero count of pieces satisfy a predicate
- `EVEN_2` / `ODD_2` — same, for two predicates jointly
- `ZERO_1` / `ZERO_2` — no piece (jointly) satisfies the predicate(s)
- `EXCLUSIVELY` — EVERY piece satisfies the predicate
- `MORE_THAN` — more pieces satisfy predicate1 than predicate2
- `EITHER_OR` — count is exactly N1 or exactly N2

**Interaction predicates** (use ONLY when hypothesis is about touching/pointing/on top of):
- `TOUCHING`, `POINTING`, `ON_TOP_OF` — combine two piece predicates into an interaction predicate
- `AT_LEAST_INTERACTION` / `EXACTLY_INTERACTION` — threshold count of interactions
- `EVEN_INTERACTION` / `ODD_INTERACTION` — parity of interaction count

**Whole-scene predicates** (no arguments):
- `ALL_THREE_SHAPES` — all three shapes appear somewhere in the scene
- `ALL_THREE_COLORS` — all three colors appear somewhere in the scene

**Logic**:
- `AND` / `OR` — combine two rules (use only if hypothesis has two distinct sub-conditions)

Full primitive list: {primitive_list}

# Step 3 — pick ONLY the basic predicates named in your hypothesis
Rules:
- Pick AT MOST 4 atoms total. If your hypothesis names only 1 attribute, list only that 1.
- DO NOT list a predicate just because it appears somewhere in the images — only include ones your hypothesis explicitly references.
- shape predicates:              {existing_objects}
- color/orientation predicates:  {existing_properties}
- interaction predicates:        {existing_actions}

# Worked example
Hypothesis: "All positive scenes have at least 2 blue pieces."
→ focus_predicates: ["AT_LEAST_1"]   (AT_LEAST_1 N IS_BLUE checks count >= N)
→ focus_objects:    ["IS_BLUE"]      (only the atom the hypothesis names)
NOT: all predicates + all colors + all shapes.

# Output
Return ONLY a fenced JSON block:
```json
{{
  "global_hints": ["ONE specific sentence stating the hypothesised rule"],
  "focus_predicates": ["at most 3 exact primitive names"],
  "focus_objects": ["at most 4 exact atom strings"],
  "confidence": 0.0
}}
```
Set `confidence` to 0.0–0.4 if the pattern is unclear, 0.5–0.7 if moderately clear, 0.8–1.0 if obvious.
"""


vlp_weighted_intuition_prompt = """You are a Zendo intuition oracle. You are shown positive (rule-satisfying) and negative (rule-violating) example scenes. Your job is to suggest which predicates and objects the synthesis search should focus on.

Available predicates: {available_predicates}
Available objects/variables: {available_variables}

Return ONLY a JSON object inside a fenced ```json block:
```json
{{
  "global_hints": ["short free-form observations about the rule"],
  "focus_predicates": ["names from the available predicate list"],
  "focus_objects": ["names from the available variable list"],
  "confidence": 0.0
}}
```
The `confidence` field is a float in [0, 1] expressing how sure you are about the focus list.
"""
