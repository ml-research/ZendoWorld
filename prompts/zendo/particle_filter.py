evolve_h_modes = ['quantifier', 'additional attribute', 'change attribute']


new_evolve_h_prompt = """A structure has one or more blocks. Each block should contain the following attributes: 
- color: red, blue, yellow;
- shape: block, wedge, pyramid;
- orientation: upright, upside_down, flat, cheesecake, doorstop;
- relation: grounded (whether the piece is touching the ground), touching(ID) (whether the piece is touching another piece with ID), pointing(ID) (whether the piece is pointing to another piece with ID), on_top_of(ID) (whether the piece is on top of another piece with ID).
Wedges can be doorstop, cheesecake or flat, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.

Example of rule modifications: 
Quantifier change: 'There must be a blue block' -> 'There are two blue blocks'
Additional attribute: 'There must be a blue block' -> 'There must be a blue block that is upright'
Attribute change: 'There must be a blue block' -> 'There must be a yellow block'
These modifications are "local": only one attribute/quantifier is changed or added for each modification.

Please modify the rule '{h}'. Generate {num} rules for each type of modification (Quantifier change, Additional attribute, Attribute change) so that the appended image of a structure is {text_y} a good structure (follows the rule):

Make the format a numbered list (1., 2., ..., 5.) Remember that the new rules should be a "local" modification from the rule '{h}'. Do not use attribute values that are not mentioned earlier. Do not say anything other than the modified rules.
"""

new_evolve_h_prompt_desc = """A structure has one or more blocks. Each block should contain the following attributes: 
- color: red, blue, yellow;
- shape: block, wedge, pyramid;
- orientation: upright, upside_down, flat, cheesecake, doorstop;
- relation: grounded (whether the piece is touching the ground), touching(ID) (whether the piece is touching another piece with ID), pointing(ID) (whether the piece is pointing to another piece with ID), on_top_of(ID) (whether the piece is on top of another piece with ID).
Wedges can be doorstop, cheesecake or flat, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.

Example of rule modifications: 
Quantifier change: 'There must be a blue block' -> 'There are two blue blocks'
Additional attribute: 'There must be a blue block' -> 'There must be a blue piece that is upright' or 'There are more blue pieces than red pieces'
Attribute change: 'There must be a blue block' -> 'There must be a yellow block'
Relational modifications: 'There must be a blue block' -> 'There must be a block that is touching a red piece'
These modifications are "local": only one attribute/quantifier is changed or added for each modification.

Please modify the rule '{h}'. Generate {num} rules for each type of modification (Quantifier change, Additional attribute, Attribute change) so that the following structure is {text_y} a good structure:
{x}
Note that the number of the blocks do not matter.

Make the format a numbered list (1., 2., ..., 5.) Remember that the new rules should be a "local" modification from the rule '{h}'. Do not use attribute values that are not mentioned earlier. Do not say anything other than the modified rules.
"""

basic_propose_h_prompt = """Please list {num} possible rules about the {att} with the choices {att_choices}.

Example 1:
{example}
Do NOT propose rules containing none such as "there is a wedge pointing at nothing/none".

Task 1:
Image is attached.
Simple rules (Orders do NOT matter):
"""

basic_propose_h_prompt_desc = """Please list {num} possible rules about the {att} with the choices {att_choices}.

Example 1:
{example}
Do NOT propose rules containing none such as "there is a wedge pointing at nothing/none".

Task 1:
{x}
Simple rules (Orders do NOT matter):
"""

propose_h_all_basic_prompt = """Given the following structures described with {att_summary} of blocks in the structures:
{text_c}
Please list {num} possible rules about the attributes in a structure that differentiate the good structures from the bad structures.
Keep in mind that 
1. All bad structures must violate the rules.
2. Orders of blocks in a structure do NOT matter.
3. The rules are short, concise, single sentences.
4. The rules are very simple.
Please number them from 1-{num} and do not say anything else
"""

prior_prompt = """Your task is to list the attribute instances involved in the given rule about blocks.
Example:
'There are exactly three blocks.': [three (quantity)]
'There is no grounded blocks': [no/zero (quantity), grounded (groundedness)]
'There are at least two small blue blocks.': [two (quantity), small (size), blue (color)]
'A blue block touches a red block.' : [blue (color), red (color), touching (action)] 
Task:
'{h}':
Give your answer without explanation"""