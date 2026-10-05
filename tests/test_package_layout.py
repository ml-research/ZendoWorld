"""Exercise import migration, legacy serialization, and subprocess resource paths."""
import ast
import io
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch
from types import ModuleType

import torch
import yaml

from zendoworld.paths import PACKAGE_ROOT, PROJECT_ROOT, RENDER_CONFIG, RULES_FILE
from zendoworld.serialization import load_pickle
from zendoworld.synthesis.program import BasicPrimitive
from zendoworld.synthesis.type_system import PrimitiveType


def test_legacy_pickle_class_names():
    assert load_pickle(io.BytesIO(b'cprogram\nBasicPrimitive\n.')) is BasicPrimitive
    assert load_pickle(io.BytesIO(b'ctype_system\nPrimitiveType\n.')) is PrimitiveType


def test_legacy_torch_class_names():
    import zendoworld.serialization as legacy_pickle
    data = io.BytesIO()
    legacy_module = ModuleType('program')
    legacy_module.BasicPrimitive = BasicPrimitive
    with patch.object(BasicPrimitive, '__module__', 'program'), patch.dict(sys.modules, {'program': legacy_module}):
        torch.save(BasicPrimitive, data)
    data.seek(0)
    assert torch.load(data, weights_only=False, pickle_module=legacy_pickle) is BasicPrimitive


def test_all_local_imports_resolve():
    for file in [*PACKAGE_ROOT.rglob('*.py'), *(PROJECT_ROOT / 'tests').glob('*.py')]:
        tree = ast.parse(file.read_text(), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if module and module.startswith('zendoworld.'):
                    target = PACKAGE_ROOT.joinpath(*module.split('.')[1:])
                    assert target.with_suffix('.py').is_file() or (target / '__init__.py').is_file(), (file, module)
                assert module not in {'Algorithms', 'Predictions', 'DSL', 'program', 'type_system', 'zendo'}, (file, module)


def test_bundled_render_config_resolves_assets():
    config = yaml.safe_load(RENDER_CONFIG.read_text())
    for key in ['base_scene_blendfile', 'properties_json', 'shape_dir', 'rules_prolog_file']:
        assert (RENDER_CONFIG.parent / config[key]).exists(), key


def test_blender_launcher_uses_absolute_package_paths(tmp_path, monkeypatch):
    from zendoworld.rendering import render
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ZENDO_RUN_NAMESPACE', 'layout-test')
    process = Mock()
    process.wait.return_value = 0
    with patch.object(render.subprocess, 'Popen', return_value=process) as launch, \
         patch.object(render.time, 'sleep'), \
         patch.object(render.torch, 'load', return_value='tensor'):
        assert render.render_scene([], 'example') == 'tensor'
    command = launch.call_args.args[0]
    assert Path(command[command.index('--python') + 1]) == PACKAGE_ROOT / 'rendering/render_single.py'
    assert Path(command[command.index('--config-file') + 1]) == RENDER_CONFIG
    assert str(PACKAGE_ROOT.parent) in launch.call_args.kwargs['env']['PYTHONPATH'].split(os.pathsep)


def test_prolog_launcher_uses_module_entry_point(tmp_path, monkeypatch):
    from zendoworld.game.player import call_prolog_subprocess_with_retries
    monkeypatch.chdir(tmp_path)
    with patch('zendoworld.game.player.subprocess.check_output', return_value=b'[["example"]]') as launch:
        assert call_prolog_subprocess_with_retries(1, 'query', str(RULES_FILE)) == [['example']]
    assert launch.call_args.args[0] == [sys.executable, '-m', 'zendoworld.cli.generate_prolog', '1', 'query', str(RULES_FILE)]


def test_cli_help_outside_checkout(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT.parent), PYTHONHASHSEED='0')
    result = subprocess.run([sys.executable, '-m', 'zendoworld', 'run', '--help'], cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    assert '--threads' in result.stdout and '--use-vllm' in result.stdout


def test_vlp_discovery_uses_bundled_templates_outside_checkout(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from zendoworld.game.player import VLPZendoPlayer
    player = VLPZendoPlayer.__new__(VLPZendoPlayer)
    player._discovery_count = 1
    player.symbolic = False
    player.discovered_variables = {}
    monkeypatch.chdir(tmp_path)
    cases = [
        ('_discover_objects', 'objects_old_new', 'objects', {'n_min_properties': 1}),
        ('_discover_properties', 'properties', 'properties', {'objects': ['block'], 'n_min_properties': 1}),
        ('_discover_actions', 'actions', 'actions', {'objects': ['block'], 'n_min_actions': 1}),
    ]
    for method, filename, category, kwargs in cases:
        player._prompt_with_images = Mock(return_value=f'```python\n{category} = ["block"]\n```')
        getattr(player, method)(['scene.png'], prompter=SimpleNamespace(), **kwargs)
        template = (PACKAGE_ROOT / 'prompts/vlp/discovery' / f'{filename}.txt').read_text()
        expected = template.replace('{n}', '1').replace('{objects}', str(['block']))
        assert player._prompt_with_images.call_args.kwargs['prompt_text'] == expected
