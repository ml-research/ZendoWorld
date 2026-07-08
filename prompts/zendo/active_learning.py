propose_x_prompt = """Given the rule '{h}', please give one structure that conforms with the rule and another structure that violates with the rule. 

A structure has one of more pieces. Each piece should contain the following attributes: 
- color: red, blue, yellow;
- shape: block, wedge, pyramid;
- orientation: upright, upside_down, flat, cheesecake, doorstop;
- relation: grounded (whether the piece is touching the ground), touching(ID) (whether the piece is touching another piece with ID), pointing(ID) (whether the piece is pointing to another piece with ID), on_top_of(ID) (whether the piece is on top of another piece with ID).
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.

Use prolog style lists of items to represent structures. Each structure should be labeled with 1 if it conforms with the rule, and 0 if it violates the rule.
Here is an example of a valid answer:
[["item(0, red, block, upright, grounded)", "item(1, blue, wedge, doorstop, touching(0))"], 0]
[["item(0, yellow, pyramid, upside_down, grounded)", "item(1, blue, wedge, doorstop, on_top_of(0))", "item(2, red, block, upright, pointing(1))"], 1]

Return your answer in this exact format:
[["item(ID, color, shape, orientation, interaction)", ...], 0]
[["item(ID, color, shape, orientation, interaction)", ...], 1]
"""

# propose_random_x_prompt = """A structure has one of more blocks. Each block should contain the following attributes: 
# {spec}

# Please generate {n} random structures in the following format:
# Structure x:
# {example_block}

# A structure has {n_blocks} blocks.
# """

propose_llm_x_prompt = """A structure has one of more pieces/items. Each piece should contain the following attributes: 
- color: red, blue, yellow;
- shape: block, wedge, pyramid;
- orientation: upright, upside_down, flat, cheesecake, doorstop;
- relation: grounded (whether the piece is touching the ground), touching(ID) (whether the piece is touching another piece with ID), pointing(ID) (whether the piece is pointing to another piece with ID), on_top_of(ID) (whether the piece is on top of another piece with ID).
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.

You are playing a game where you are trying to figure an underlying secret rule governing structures

At this point in the game, you think the underlying secret rule could be any of the following rules:
{hs}

Please choose one structure to ask if it conforms with the underlying secret rule. You want to pick a structure that would help you gain the most information on the secret rule.
Please ONLY return a new example and its label within a python block, in this exact format, where label is 1 for good structure (following the rule) and 0 for bad structure (not following the rule).
The label should be your best guess based on the rules above, but it does not have to be correct.
Here are examples of valid formats:
[["item(0, red, block, upright, grounded)", "item(1, blue, wedge, doorstop, touching(0))"], 1]
[["item(0, yellow, pyramid, upside_down, grounded)", "item(1, blue, wedge, doorstop, on_top_of(0))", "item(2, red, block, upright, pointing(1))"], 0]
Return your answer in this exact format including the python block:
```python
[["item(ID, color, shape, orientation, interaction)", ...], label]
```
"""