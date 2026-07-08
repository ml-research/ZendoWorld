import json
import numpy as np
import random

from prompts.zendo import *


class AdvZendoConfig:
    att = ['colors', 'shapes', 'orientations', 'groundedness', 'touchings', 'pointings', 'stackings']
    att_choices = {
        'colors': 'blue/red/yellow',
        'shapes': 'pyramid/block/wedge',
        'orientations': 'flat/upright/upside_down/cheesecake/doorstop',
        'groundedness': 'grounded/ungrounded',
        'touchings': 'none/blocks it touches',
        'pointings': 'none/blocks it points to',
        "stackings": 'none/blocks it is on top of'
    }
    att_choices_list = {
        'colors': ['blue', 'red', 'yellow'],
        'shapes': ['pyramid', 'block', 'wedge'],
        'orientations': ['flat', 'upright', 'upside_down', 'cheesecake', 'doorstop'],
        'groundedness': [True, False],
    }
    spec = "color (blue/red/yellow)\nshape (pyramid/block/wedge)\norientation (flat/upright/upside_down/cheesecake/doorstop)\ngroundedness (grounded/ungrounded)\ntouching (none/blocks it touch)"
    example_block = "- Block 1: Color - color, Shape - shape, Orientation - orientation, Groundedness - groundedness, Touching - touching"
    examples_texts_per_att = {
        'colors': """Image shows a blue block.
Simple rules (Orders do NOT matter):
1. There is a blue block
2. All pieces are blue""",
        'shapes': """Image shows a wedge and a pyramid.
Simple rules (Orders do NOT matter):
1. There is a wedge
2. There is a pyramid""",
        'orientations': """Image shows a block that is upright and a wedge that is doorstop.
Simple rules (Orders do NOT matter):
1. There is an upright piece
2. There is a piece that is doorstop""",
        'groundedness': """Image shows a grounded block and an ungrounded wedge.
Simple rules (Orders do NOT matter):
1. There is a grounded piece and an ungrounded piece
2. There is at least one ungrounded piece""",
        'touchings': """Image shows a blue block touching a wedge.
Simple rules (Orders do NOT matter):
1. There is a block touching a wedge
2. There is at least one blue piece touching a wedge""",
        'pointings': """Image shows a block pointing to a red pyramid.
Simple rules (Orders do NOT matter):
1. There is a block pointing to a pyramid
2. There is a block pointing to a red piece""",
        'stackings': """Image shows a yellow block stacked on top of a wedge.
Simple rules (Orders do NOT matter):
1. There is a block on top of a wedge
2. There is at least one yellow block on top of a wedge""",
    }
    examples_texts_per_att_desc = {
        'colors': """["item(0, blue, block, upright, grounded)", "item(1, blue, pyramid, flat, grounded)"]
Simple rules (Orders do NOT matter):
1. There is a blue block
2. All pieces are blue""",
        'shapes': """["item(0, blue, wedge, upright, grounded)", "item(1, blue, pyramid, flat, grounded)"]
Simple rules (Orders do NOT matter):
1. There is a wedge
2. There is a pyramid""",
        'orientations': """["item(0, blue, block, upright, grounded)", "item(1, blue, wedge, doorstop, grounded)"]
Simple rules (Orders do NOT matter):
1. There is an upright piece
2. There is a piece that is doorstop""",
        'groundedness': """["item(0, blue, block, upright, grounded)", "item(1, red, wedge, flat, on_top_of(0))"]
Simple rules (Orders do NOT matter):
1. There is a grounded piece and an ungrounded piece
2. There is at least one ungrounded piece""",
        'touchings': """["item(0, blue, block, upright, touching(1))", "item(1, blue, wedge, flat, grounded)"]
Simple rules (Orders do NOT matter):
1. There is a block touching a wedge
2. There is at least one blue piece touching a wedge""",
        'pointings': """["item(0, yellow, block, upright, pointing(1))", "item(1, red, pyramid, flat, grounded)"]
Simple rules (Orders do NOT matter):
1. There is a block pointing to a pyramid
2. There is a block pointing to a red piece""",
        'stackings': """["item(0, yellow, block, upright, on_top_of(1))", "item(1, blue, wedge, flat, grounded)"]
Simple rules (Orders do NOT matter):
1. There is a block on top of a wedge
2. There is at least one yellow block on top of a wedge""",
    }

    def get_spec(rng=None): 
        txt = ""
        for k, v in AdvZendoConfig.att_choices.items():
            if rng is not None:
                v_list = v.split('/')
                v = '/'.join(rng.permutation(v_list))
            txt += f'{k} ({v})\n'
        return txt
    

class AdvZendoConfigStacking(AdvZendoConfig):
    att = ['colors', 'sizes', 'orientations', 'groundedness', 'touchings']
    att_choices = {
        'colors': 'blue/red/green',
        'sizes': 'small/medium/large',
        'orientations': 'upright/left/right/strange',
        'groundedness': 'grounded/ungrounded/stacking',
        'touchings': 'none/blocks it touch or stack',
    }
    att_choices_list = {
        'colors': ['blue', 'red', 'green'],
        'sizes': ['small', 'medium', 'large'],
        'orientations': ['upright', 'left', 'right', 'strange'],
        'groundedness': [True, False],
    }
    spec = "color (blue/red/green)\nsize (small/medium/large)\norientation (upright/left/right/strange)\ngroundedness (grounded/ungrounded/stacking)\ntouching (none/blocks it touch or stack)"
    example_block = "- Block 1: Color - color, Size - size, Orientation - orientation, Groundedness - groundedness, Touching - touching"

    def get_spec(rng=None): 
        txt = ""
        for k, v in AdvZendoConfigStacking.att_choices.items():
            if rng is not None:
                v_list = v.split('/')
                v = '/'.join(rng.permutation(v_list))
            txt += f'{k} ({v})\n'
        return txt
    