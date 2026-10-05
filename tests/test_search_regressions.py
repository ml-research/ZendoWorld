"""Exercise search code without loading CUDA, Ray, or vision models."""
import ast
import logging
import math
import os
from pathlib import Path
import sys
import time
import typing
from typing import Callable, List, Tuple
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from zendoworld.synthesis.program import BasicPrimitive, Function, Lambda, New, Program, Variable, strip_trailing_var0


def search_namespace():
    ns = dict(globals(), PCFG=object)
    tree = ast.parse((ROOT / 'src/zendoworld/synthesis/search_runner.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
                'COMMUTATIVE_FUNCS', 'PARTIAL_COMMUTATIVE_FUNCS', 'GROUND_PREDS',
                'INVALID_HEADS', 'total_number_programs', 'use_heap_search_cached_eval'
            } for t in node.targets)
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<search>', 'exec'), ns)
    return ns


class SearchTests(unittest.TestCase):
    def run_search(self, generator, checker=lambda p, cached: 1., **kwargs):
        ns = search_namespace()
        ns.update(list_algorithms=[(generator, 'Heap Search', {})], heap_search=generator, reconstruct=set())
        pcfg = SimpleNamespace(start=('test',), probability_program=lambda *args: .1)
        return ns['run_algorithm'](checker, pcfg, 0, **kwargs)

    def test_stops_without_requesting_extra_candidate(self):
        def generate(pcfg):
            yield BasicPrimitive('a')
            yield BasicPrimitive('b')
            raise AssertionError('Must stop immediately after two successful candidates')
        result = self.run_search(generate)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(row[1] >= 0 and row[2] >= 0 for row in result))

    def test_generator_error_propagates(self):
        def generate(pcfg):
            yield BasicPrimitive('a')
            raise RuntimeError('broken generator')
        with self.assertRaisesRegex(RuntimeError, 'broken generator'):
            self.run_search(generate)

    def test_accuracy_filter_and_explicit_fallback(self):
        generate = lambda pcfg: iter([BasicPrimitive('a')])
        self.assertEqual(self.run_search(generate, checker=lambda *args: .4), [])
        result = self.run_search(generate, checker=lambda *args: .4, return_best_if_none=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][5], .4)

    def test_vlp_retries_have_independent_budgets_and_reach_fallback(self):
        tree = ast.parse((ROOT / 'src/zendoworld/game/player.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VLPZendoPlayer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_search_programs_vlp')
        calls = []
        clock = [0.]
        sentinel = [(None, 0., 0., 0, 0., 0., 0.)]
        def gather(*args, **kwargs):
            calls.append(kwargs)
            clock[0] += kwargs['timeout'] + 1
            if kwargs['return_best_if_none']:
                return [('', [('best', 0., 0., 1, 0., .4, 0.)])]
            return [('', sentinel)]
        ns = dict(math=math, os=os, time=SimpleNamespace(monotonic=lambda: clock[0]),
                  task_set2zendodataset_vlp=lambda *args: [], gather_data=gather)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<player>', 'exec'), ns)
        player = SimpleNamespace(model=None, dsl=None, cfg=None, zendo_cfg=None,
                                 already_guessed=[], MAX_SEARCH_RETRIES=3)
        with patch.dict(os.environ, {'ZENDO_VLP_SEARCH_TIMEOUT': '100'}):
            result = ns['_search_programs_vlp'](player, [('scene', True)] * 20)
        self.assertEqual(result[0][0], 'best')
        self.assertEqual([c['timeout'] for c in calls], [100, 100, 100])
        self.assertEqual([c['accuracy'] for c in calls], [1., .95, .9])
        self.assertEqual([c['return_best_if_none'] for c in calls], [False, False, True])
        calls.clear()
        clock[0] = 0
        with patch.dict(os.environ, {'ZENDO_VLP_SEARCH_TIMEOUT': '5'}):
            ns['_search_programs_vlp'](player, [('scene', True)] * 20)
        self.assertEqual(len(calls), 3)

    def test_normal_exhaustion_returns_partial_results(self):
        self.assertEqual(len(self.run_search(lambda pcfg: iter([BasicPrimitive('a')]))), 1)

    def test_zero_timeout_does_not_initialize_search(self):
        def generate(pcfg):
            raise AssertionError('Search should not initialize')
        self.assertEqual(self.run_search(generate, timeout=0), [])

    def test_wall_time_includes_candidate_filtering(self):
        ns = search_namespace()
        now = [0.]
        def generate(pcfg):
            for _ in range(10):
                yield BasicPrimitive('a')
        def normalize(program):
            now[0] += 2
            return program
        ns.update(time=SimpleNamespace(monotonic=lambda: now[0], perf_counter=lambda: now[0]),
                  normalize_program_structure=normalize,
                  list_algorithms=[(generate, 'Heap Search', {})], heap_search=generate, reconstruct=set())
        pcfg = SimpleNamespace(start=('test',), probability_program=lambda *args: .1)
        ns['run_algorithm'](lambda *args: 0., pcfg, 0, timeout=1)
        self.assertEqual(now[0], 2.)

    def test_counting_normalization_preserves_fixed_argument(self):
        ns = search_namespace()
        for name in ('EXACTLY_2', 'AT_LEAST_2'):
            program = Function(BasicPrimitive(name), [BasicPrimitive('9'), BasicPrimitive('B'), BasicPrimitive('A')])
            normalized = ns['normalize_program_structure'](program)
            self.assertEqual([str(a) for a in normalized.arguments], ['9', 'A', 'B'])
            duplicate = Function(BasicPrimitive(name), [BasicPrimitive('9'), BasicPrimitive('A'), BasicPrimitive('A')])
            collapsed = ns['canonicalize_program'](duplicate)
            self.assertEqual(collapsed.function.primitive, name.replace('_2', '_1'))
            self.assertEqual([str(a) for a in collapsed.arguments], ['9', 'A'])

    def test_rejected_rule_matches_without_mutating_candidate(self):
        candidate = Function(BasicPrimitive('test'), [BasicPrimitive('A'), Variable(0)])
        result = self.run_search(lambda pcfg: iter([candidate]), incorrect_rules=['(test A)'])
        self.assertEqual(result, [])
        self.assertEqual(len(candidate.arguments), 2)

    def test_vlp_timeout_env_access(self):
        tree = ast.parse((ROOT / 'src/zendoworld/game/player.py').read_text())
        ns = dict(math=math, time=time, task_set2zendodataset_vlp=lambda *args: [])
        for node in tree.body:
            if isinstance(node, ast.Import) and any(a.name == 'os' for a in node.names):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<player>', 'exec'), ns)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VLPZendoPlayer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_search_programs_vlp')
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<player>', 'exec'), ns)
        player = SimpleNamespace(model=None, dsl=None, cfg=None, zendo_cfg=None)
        with patch.dict(os.environ, {'ZENDO_VLP_SEARCH_TIMEOUT': '0'}):
            result = ns['_search_programs_vlp'](player, [('image', True)])
        self.assertIsNone(result[0][0])


if __name__ == '__main__':
    unittest.main()
