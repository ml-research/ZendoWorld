"""Integration checks for symbolic saves using the actual tensor format."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from zendoworld.datasets.pieces2tensor import prolog_strings_to_tensor
from zendoworld.experiments.runner import already_done, save_state


class SymbolicIOTests(unittest.TestCase):
    def test_symbolic_save_and_resume_without_images(self):
        scene = ['item(0, red, block, upright, grounded)']
        tensor = prolog_strings_to_tensor([scene])[0]
        state = SimpleNamespace(to_dict=lambda: (
            {'paths': [str(scene), 'counter_generated']},
            [(tensor, True), (tensor, False)], [],
        ))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_state(state, root, 0, 1, symbolic=True)
            saved = json.loads((root / 'seed_1/task_0_state.json').read_text())
            self.assertEqual(saved['input_mode'], 'symbolic')
            self.assertEqual(saved['scene_descriptions'], [scene, scene])
            self.assertTrue(already_done(root, 0, 1, symbolic=True))
            self.assertTrue(already_done(root, 0, 1))
            self.assertFalse((root / 'seed_1/task_0_images').exists())
            examples = torch.load(root / 'seed_1/examples_0.pt', weights_only=True)
            self.assertTrue(torch.equal(examples[0][0], tensor))
            # Legacy symbolic results have no explicit mode metadata.
            saved.pop('input_mode')
            saved.pop('scene_descriptions')
            (root / 'seed_1/task_0_state.json').write_text(json.dumps(saved))
            self.assertTrue(already_done(root, 0, 1, symbolic=True))
            self.assertFalse(already_done(root, 0, 1))
            (root / 'seed_1/examples_0.pt').unlink()
            self.assertFalse(already_done(root, 0, 1, symbolic=True))


if __name__ == '__main__':
    unittest.main()
