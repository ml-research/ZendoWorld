from zendoworld.game.symbolic_parsers import (
    parse_scientist_structure,
    parse_vlp_actions,
    parse_vlp_objects,
)


def test_vlp_objects_parse_attributes_and_groundedness():
    scene = [
        "item(4, red, block, upright, grounded)",
        "item(9, blue, wedge, doorstop, on_top_of(4))",
    ]
    assert parse_vlp_objects(scene) == [
        ["block", "red", "upright", "vertical", "grounded"],
        ["wedge", "blue", "doorstop", "flat"],
    ]


def test_vlp_actions_parse_all_subject_target_attribute_pairs():
    scene = [
        "item(1, red, block, upright, grounded)",
        "item(7, blue, wedge, flat, touching(1))",
    ]
    assert parse_vlp_actions(scene) == [
        [subject, "touching", target]
        for subject in ["wedge", "blue", "flat"]
        for target in ["block", "red", "upright", "vertical"]
    ]


def test_scientist_parser_resolves_ids_and_relations():
    scene = [
        "item(8, yellow, pyramid, upside_down, pointing(2))",
        "item(2, blue, block, upright, grounded)",
        "item(5, red, wedge, doorstop, on_top_of(2))",
    ]
    structure = parse_scientist_structure(scene)
    assert [(piece.color, piece.shape, piece.orientation) for piece in structure.pieces] == [
        ("blue", "block", "upright"),
        ("red", "wedge", "flat"),
        ("yellow", "pyramid", "upside_down"),
    ]
    assert structure.pieces[1].on_top_of == 0
    assert structure.pieces[2].pointing == 0


def test_parsers_return_empty_sentinel_for_invalid_scene():
    assert parse_vlp_objects(["not an item"]) == [[]]
    assert parse_vlp_actions([]) == [[]]
    assert parse_scientist_structure([]).pieces == []


def test_vlp_vocabulary_filtering_and_string_input():
    scene = "['item(0, red, block, upright, grounded)', 'item(1, blue, wedge, flat, touching(0))']"
    assert parse_vlp_objects(scene, ['block'], ['red']) == [['block', 'red']]
    assert parse_vlp_actions(scene, ['block', 'wedge'], ['red'], ['touching']) == [
        ['wedge', 'touching', 'block'], ['wedge', 'touching', 'red'],
    ]
    assert parse_vlp_actions(scene, actions=['pointing']) == [[]]


def test_vlp_repeated_relations_keep_counts():
    scene = ['item(0, red, block, upright, grounded)',
             'item(1, blue, wedge, flat, on_top_of(0))',
             'item(2, blue, wedge, flat, on_top_of(0))']
    assert parse_vlp_actions(scene, ['block', 'wedge'], [], ['on_top_of']) == [
        ['wedge', 'on_top_of', 'block'], ['wedge', 'on_top_of', 'block'],
    ]


def test_scientist_touching_and_stacking_are_symmetric():
    scene = ['item(9, yellow, pyramid, upright, touching(2))',
             'item(2, blue, block, flat, grounded)',
             'item(5, red, wedge, doorstop, on_top_of(2))']
    pieces = parse_scientist_structure(scene).pieces
    assert [vars(p) for p in pieces] == [
        dict(color='blue', shape='block', orientation='flat', touching=[1, 2], on_top_of=None, pointing=None),
        dict(color='red', shape='wedge', orientation='flat', touching=[0], on_top_of=0, pointing=None),
        dict(color='yellow', shape='pyramid', orientation='upright', touching=[0], on_top_of=None, pointing=None),
    ]


def test_scientist_single_item_string():
    pieces = parse_scientist_structure('item(3, red, pyramid, upside_down, grounded)').pieces
    assert [vars(p) for p in pieces] == [
        dict(color='red', shape='pyramid', orientation='upside_down', touching=[], on_top_of=None, pointing=None),
    ]


def test_vlp_dsl_parser_selection_preserves_neural_default():
    from zendoworld.synthesis.dsl.vlp_dsl_symbolic import get_dsl

    class Prompter:
        calls = 0

        def prompt_with_text(self, **kwargs):
            self.calls += 1
            return "objects = [['block', 'red']]"

    prompter = Prompter()
    scene = ['item(0, red, block, upright, grounded)']
    variables = dict(objects=['block'], properties=['red'], actions=['touching'])
    neural, _ = get_dsl(prompter, variables)
    assert neural['get_objects'](scene) == [['block', 'red']]
    assert prompter.calls == 1
    symbolic, _ = get_dsl(prompter, variables, scene_parser='symbolic')
    assert symbolic['get_objects'](scene) == [['block', 'red']]
    assert symbolic['get_actions'](scene) == [[]]
    assert prompter.calls == 1


def test_vlp_dsl_counts_orientation_groups_and_interactions():
    from zendoworld.synthesis.dsl.vlp_dsl_symbolic import get_dsl

    scene = [
        'item(0, blue, wedge, cheesecake, grounded)',
        'item(1, red, wedge, doorstop, on_top_of(0))',
        'item(2, yellow, block, flat, pointing(0))',
        'item(3, red, pyramid, upside_down, grounded)',
        'item(4, blue, block, upright, grounded)',
    ]
    variables = dict(objects=['wedge', 'block', 'pyramid'],
                     properties=['flat', 'vertical', 'doorstop', 'cheesecake',
                                 'upright', 'upside_down'],
                     actions=['on_top_of', 'pointing'])
    semantics, _ = get_dsl(None, variables, scene_parser='symbolic')
    objects = semantics['get_objects'](scene)
    count = semantics['count_property'](objects)
    assert count('flat') == 3
    assert count('vertical') == 2
    assert count('doorstop') == count('cheesecake') == 1
    actions = semantics['get_actions'](scene)
    relation_count = semantics['count_property_with_action_with_property'](actions)
    assert relation_count('doorstop')('on_top_of')('cheesecake') == 1
    assert relation_count('flat')('on_top_of')('flat') == 1
    assert relation_count('flat')('pointing')('cheesecake') == 1
    assert relation_count('cheesecake')('on_top_of')('doorstop') == 0
    assert semantics['count_property_with_action_with_object'](actions)('doorstop')('on_top_of')('wedge') == 1
    assert parse_vlp_objects(scene[:2], ['wedge'], ['flat']) == [
        ['wedge', 'flat'], ['wedge', 'flat'],
    ]


if __name__ == '__main__':
    # Also runnable without pytest: python -m tests.test_symbolic_parsers
    tests = [(name, fn) for name, fn in globals().copy().items() if name.startswith('test_')]
    for name, test in tests:
        test()
        print(f'PASS {name}')
    print(f'{len(tests)} tests passed')
