perception_prompt = """Your task is to tell me, what pieces the structure in the image contains. A structure has one or more pieces/items. Each piece should contain the following attributes: 
- color: red, blue, yellow;
- shape: block, wedge, pyramid;
- orientation: upright, upside_down, flat, cheesecake, doorstop;
- relation: grounded (whether the piece is touching the ground), touching(ID) (whether the piece is touching another piece with ID), pointing(ID) (whether the piece is pointing to another piece with ID), on_top_of(ID) (whether the piece is on top of another piece with ID).
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Wedges have two triangular sides, two rectangular sides, and one bottom side. Blocks have six rectangular sides. Pyramids have four triangular sides and one bottom side.
A doorstop is when a wedge is flat on the ground, with on of the rectangular sides touching the ground, while cheesecake is when a wedge is flat on the ground, with one of the triangular sides touching the ground.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.
Please describe all pieces in the structure in the following format:
["item(0, color, shape, orientation, interaction)", "item(1, color, shape, orientation, interaction)", ...]

Please ONLY return the list of items within a python block, in this exact format, with no further explanation.

Here are examples of valid formats:
["item(0, red, block, upright, grounded)", "item(1, blue, wedge, doorstop, touching(0))"]
["item(0, yellow, pyramid, upside_down, grounded)", "item(1, blue, wedge, doorstop, on_top_of(2))", "item(2, red, block, upright, pointing(0))"]
Return your answer in this exact format including the python block:
```python
["item(ID, color, shape, orientation, interaction)", ...]
```
"""