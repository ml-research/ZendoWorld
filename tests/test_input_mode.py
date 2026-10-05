"""Mode/prompt regressions runnable without model SDKs or API calls."""
import ast
from pathlib import Path
import unittest

from zendoworld.game.input_mode import uses_image_paths


def prompt_player():
    # Load the real prompt-building methods without importing model backends.
    source = Path(__file__).resolve().parents[1] / 'src/zendoworld/game/player.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FullGPTZendoPlayer')
    names = {'_collect_examples', '_build_image_prompt',
             'build_zendo_prompt_guess_rule', 'build_zendo_prompt_from_examples'}
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'tensor_to_prolog_strings': lambda tensors: tensors}
    for name in ('guess_rule_dsl_text_prompt', 'guess_rule_nl_text_prompt',
                 'propose_structure_text_prompt'):
        namespace[name] = 'Positive: {positives}; negative: {negatives}'
    for name in ('guess_rule_dsl_images_header', 'guess_rule_nl_images_header',
                 'propose_structure_images_header'):
        namespace[name] = 'Images'
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    player = namespace['FullGPTZendoPlayer']()
    player.previous_guesses = []
    player.use_dsl = False
    return player


class InputModeTests(unittest.TestCase):
    def test_symbolic_discovery_uses_scene_not_saved_image_path(self):
        source = Path(__file__).resolve().parents[1] / 'src/zendoworld/game/player.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VLPZendoPlayer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_maybe_discover_variables')
        namespace = {'tensor_to_prolog_strings': lambda scenes: scenes}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        from types import SimpleNamespace
        calls = []
        scene = "['item(0, red, block, upright, grounded)']"
        player = SimpleNamespace(
            _discovery_count=0, discovery_rounds=4, prompter=object(), symbolic=True,
            examples=[((scene, True), 'saved/image.png')],
            discovered_variables=dict(objects=[], properties=[], actions=[]),
            _variable_discovery=lambda inputs: calls.append(inputs) or {},
        )
        discover = namespace['_maybe_discover_variables']
        discover(player)
        self.assertEqual(calls.pop(), [scene])
        player.examples = [((["item(0, red, block, upright, grounded)"], True), None)]
        discover(player)
        self.assertEqual(calls.pop(), [scene])
        player.symbolic = False
        player.examples = [((None, True), 'saved/image.png')]
        discover(player)
        self.assertEqual(calls.pop(), ['saved/image.png'])

    def test_all_runner_tags(self):
        for tag in ('zendo', 'fullgpt', 'vlp', 'vlp_ablation', 'vlp_uncertainty', 'scientist',
                    'random_zendo'):
            self.assertTrue(uses_image_paths(tag))
        for tag in ('zendo_symbolic', 'fullgpt_symbolic', 'vlp_symbolic',
                    'scientist_symbolic', 'random_zendo_symbolic'):
            self.assertFalse(uses_image_paths(tag))

    def test_symbolic_prompts_ignore_existing_image_paths(self):
        player = prompt_player()
        player.create_images = uses_image_paths('fullgpt_symbolic')
        for method in (player.build_zendo_prompt_guess_rule, player.build_zendo_prompt_from_examples):
            text, paths = method([(['item(0, red, block, upright, grounded)'], True)], ['existing.png'])
            self.assertIn('item(0, red', text)
            self.assertEqual(paths, [])

    def test_visual_prompts_keep_image_paths(self):
        player = prompt_player()
        player.create_images = uses_image_paths('fullgpt')
        for method in (player.build_zendo_prompt_guess_rule, player.build_zendo_prompt_from_examples):
            parts = method([(['item(0, red, block, upright, grounded)'], True)], ['existing.png'])
            self.assertIn({'type': 'image', 'path': 'existing.png'}, parts)


if __name__ == '__main__':
    unittest.main()
