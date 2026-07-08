query_prompt = """Given the rule {rule}, please give a yes or no answer on whether the following structure conform with the rule:
{structure}
If yes, say 'yes'. If no, explain why not."""


evaluate_rule_prompt = """We are playing the game Zendo with the following attributes for the pieces:
- color: red, blue, yellow;
- shape: block, wedge, pyramid;
- orientation: upright, upside_down, flat, cheesecake, doorstop;
- relation: grounded (whether the piece is touching the ground), touching(ID) (whether the piece is touching another piece with ID), pointing(ID) (whether the piece is pointing to another piece with ID), on_top_of(ID) (whether the piece is on top of another piece with ID).
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.
Is the first DSL rule semantically equivalent to the second DSL rule?

Rule 1:
{rule1}

Rule 2:
{rule2}

----------------------------------------
Output format
----------------------------------------

You MUST return exactly one of the following:

Case 1 — Equivalent:
```json
{{"equivalent": true}}
```
Case 2 — Not equivalent:
```json
{{
  "equivalent": false,
  "counterexample": "<structure description>"
}}
```
----------------------------------------
Counterexample requirements
----------------------------------------

The counterexample evaluate differently on the two rules.

Describe the structure using ONLY this format:

piece 1: color = ..., shape = ..., orientation = ..., relation = ...

piece 2: color = ..., shape = ..., orientation = ..., relation = ...
...

Do NOT include explanations

Do NOT include reasoning

Do NOT include any text outside the JSON

Return ONLY the JSON block.
"""

propose_counter_prompt = """A structure has one or more pieces/items. Each piece should contain the following attributes: 
- color: red, blue, yellow;
- shape: block, wedge, pyramid;
- orientation: upright, upside_down, flat, cheesecake, doorstop;
- relation: grounded (whether the piece is touching the ground), touching(ID) (whether the piece is touching another piece with ID), pointing(ID) (whether the piece is pointing to another piece with ID), on_top_of(ID) (whether the piece is on top of another piece with ID).
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.

Your task is to convert the following structure description into the wanted format.
Description: {description}

ONLY return a new example within a python block, in this exact format.
Here are examples of valid formats:
["item(0, red, block, upright, grounded)", "item(1, blue, wedge, doorstop, touching(0))"]
["item(0, yellow, pyramid, upside_down, grounded)", "item(1, blue, wedge, doorstop, on_top_of(0))", "item(2, red, block, upright, pointing(1))"]
Return your answer in this exact format including the python block:
```python
["item(ID, color, shape, orientation, interaction)", ...]
```
"""

play_zendo_easy_prompt = """You are playing an inductive game with me. I'll be the moderator, and your task is to figure out the secret rule determining whether a structure of blocks is good or bad. 
You will do that by coming up with a structure of blocks and asking me whether it is a good structure according to the rule. 

The structure has one of more blocks. Each block should contain the following attributes: 
color (blue/red/green/yellow) 
size (small/medium/large)
orientation (upright/flat)

To give you a start, I'll describe one structure that follows the rule:

{text_c}"""

play_zendo_hard_prompt = """You are playing an inductive game with me. I'll be the moderator, and your task is to figure out the secret rule that I know by coming up with a structure of blocks to ask me whether it conforms with the secret rule or not. 

The structure has one of more blocks. Each block should contain the following attributes: 
{att_par}

To give you a start, I'll describe one structure that follows the rule:

{text_c}

Give a very short summary on what you currently think the secret rule is."""

x_conforms_h_prompt = """We are playing the game Zendo. Given the rule about good structure '{h}', is the following shown structure a good structure?

Say 'yes' or 'no'. Do not say anything else."""

commonalities_prompt = """Please summarize the commonalities among the good structures and the bad structures
{text_c}"""

rule_translation_prompt = """Please synthesize a python program that implements the rule '{h}'

The program should take in a ZendoStructure which represents a structure and returns True if it's a good structure and False otherwise.

The docstrings for the classes are as follow:

class ZendoStructure:
    :param pieces: list of ZendoPiece

class ZendoPiece:
    :param color: str (blue/red/yellow) 
    :param shape: str (block/wedge/pyramid)
    :param orientation: str (upright, upside_down, flat, cheesecake, doorstop)
    :param touching: list of int (index starts at 0, representing the index of the piece that this piece is touching)
    :param on_top_of: list of int (index starts at 0, representing the index of the piece that this piece is on top of)
    :param pointing: list of int (index starts at 0, representing the index of the piece that this piece is pointing to)

The signature for the synthesized program should be
def rule(structure: ZendoStructure) -> bool

Only output the 'rule' function. Do not include anything else.
"""
