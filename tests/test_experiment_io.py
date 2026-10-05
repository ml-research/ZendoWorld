"""Dependency-free regression checks for experiment orchestration and disk I/O.

Extract the actual functions so these checks need neither CUDA nor model packages.
"""
import argparse
from contextlib import ExitStack
import ast
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import threading
import traceback
import unittest
import warnings
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def definitions(file, names, namespace):
    tree = ast.parse((ROOT / file).read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), file, 'exec'), namespace)
    return namespace


def runner_functions():
    ns = dict(globals(), _RTPT=None)
    return definitions('src/zendoworld/experiments/runner.py', {
        'save_state', 'already_done', 'copy_and_rename', 'classify_label',
        'save_task_error', 'worker', 'work_item_worker', '_split_task_seed_work',
    }, ns)


def cache_writer(path, index, lock_style=None):
    ns = definitions('src/zendoworld/backends/vllm_client.py', {'VLLMPrompter'}, dict(globals()))
    client = ns['VLLMPrompter'].__new__(ns['VLLMPrompter'])
    client.memory_file = Path(path)
    client._memory_lock = threading.Lock()
    client.memory = {}
    for n in range(10):
        if lock_style is None:
            client._save_memory({(str(index), str(n)): str(n)})
        else:
            # Simulate either already-running single-lock implementation.
            suffix = ".lock" if lock_style == "local" else ".json.lock"
            with client.memory_file.with_suffix(suffix).open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                data = json.loads(client.memory_file.read_text()) if client.memory_file.exists() else {}
                data[str((str(index), str(n)))] = str(n)
                with tempfile.TemporaryDirectory(dir=client.memory_file.parent) as staging:
                    temporary = Path(staging) / "cache.json"
                    temporary.write_text(json.dumps(data))
                    os.replace(temporary, client.memory_file)


class ExperimentIOTests(unittest.TestCase):
    def test_cli_and_distribution(self):
        tree = ast.parse((ROOT / 'src/zendoworld/experiments/runner.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        parser_nodes = []
        for n in main.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets):
                break
            parser_nodes.append(n)
        parser_ns = {'argparse': argparse}
        exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), '<parser>', 'exec'), parser_ns)
        parser = parser_ns['parser']
        args = parser.parse_args('--player vlp --prompter-model Qwen3.8-27B --use-vllm --seeds 73 --threads 4'.split())
        self.assertEqual(args.reasoning_effort, 'low')
        restored = parser.parse_args('--player vlp --run-namespace test --memory-fallback-reasoning-effort low --openai-api-key-file /tmp/unused-key'.split())
        self.assertEqual(restored.run_namespace, 'test')
        chunks = runner_functions()['_split_task_seed_work'](list(range(51)), [73], 4)
        self.assertEqual([len(c) for c in chunks], [13, 13, 13, 12])
        self.assertEqual(len(set(sum(chunks, []))), 51)

    def test_saving_and_retry(self):
        ns = runner_functions()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'source.png'
            source.write_bytes(b'image')
            state = SimpleNamespace(to_dict=lambda: ({'paths': [str(source)]}, [(None, True)], []))
            def fail(*args):
                raise OSError('injected write failure')
            ns['torch'] = SimpleNamespace(save=fail)
            with self.assertRaises(OSError):
                ns['save_state'](state, root, 28, 73)
            self.assertFalse(ns['already_done'](root, 28, 73))
            ns['save_task_error'](root, 28, 73, RuntimeError('test'))
            ns['torch'] = SimpleNamespace(save=lambda obj, path: Path(path).write_bytes(b'tensor'))
            ns['save_state'](state, root, 28, 73)
            self.assertTrue(ns['already_done'](root, 28, 73))
            self.assertFalse((root / 'seed_73/task_28_error.json').exists())
            (root / 'seed_73/examples_28.pt').unlink()
            self.assertFalse(ns['already_done'](root, 28, 73))
            (root / 'seed_73/task_28_state.json').write_text('{')
            self.assertFalse(ns['already_done'](root, 28, 73))

    def test_missing_image_fails_save(self):
        ns = runner_functions()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            directory = root / 'seed_73'
            directory.mkdir()
            marker = directory / 'task_28_state.json'
            marker.write_text('{}')
            state = SimpleNamespace(to_dict=lambda: ({'paths': [str(root / 'missing.png')]}, [(None, True)], []))
            with self.assertRaises(FileNotFoundError):
                ns['save_state'](state, root, 28, 73)
            self.assertFalse(marker.exists())
            self.assertFalse(ns['already_done'](root, 28, 73))

    def test_worker_propagates_failures(self):
        ns = runner_functions()
        ns['PLAYER_RUNNERS'] = {'vlp': lambda tasks, seeds, args: ['injected task failure']}
        with self.assertRaisesRegex(RuntimeError, 'injected task failure'):
            ns['work_item_worker']('vlp', [(28, 73)], SimpleNamespace(name='SK'))

    def test_multiprocess_cache_merges(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'memory.json')
            with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(cache_writer, path, i) for i in range(4)]
                for future in futures:
                    future.result(timeout=30)
            self.assertEqual(len(json.loads(Path(path).read_text())), 40)

    def test_cache_coordinates_with_each_legacy_lock(self):
        for style in ("local", "upstream"):
            with self.subTest(style=style), tempfile.TemporaryDirectory() as d:
                path = str(Path(d) / 'memory.json')
                with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures = [pool.submit(cache_writer, path, i, style if i % 2 else None)
                               for i in range(4)]
                    for future in futures:
                        future.result(timeout=30)
                self.assertEqual(len(json.loads(Path(path).read_text())), 40)

    def test_cache_updates_do_not_restore_stale_values(self):
        ns = definitions('src/zendoworld/backends/vllm_client.py', {'VLLMPrompter'}, dict(globals()))
        cls = ns['VLLMPrompter']
        with tempfile.TemporaryDirectory() as d:
            client = cls.__new__(cls)
            client.memory_file = Path(d) / 'memory.json'
            client._memory_lock = threading.Lock()
            client.memory = {('shared', 'seed'): 'old'}
            client.memory_file.write_text(json.dumps({str(('shared', 'seed')): 'new'}))
            client._save_memory({('other', 'seed'): 'result'})
            self.assertEqual(client._read_memory()[('shared', 'seed')], 'new')


if __name__ == '__main__':
    unittest.main()
