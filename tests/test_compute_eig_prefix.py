"""Regression checks for per-example EIG orchestration (no expensive search)."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


import zendoworld.analysis.compute_eig as eig


class Rule:
    def __init__(self, result):
        self.result = result

    def eval_naive(self, dsl, environment):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class EIGTests(unittest.TestCase):
    def run_game(self, batches, missing=None, saved_count=5, json_count=5):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'seed_1'
            directory.mkdir()
            path = directory / 'task_0_state.json'
            path.write_text(json.dumps({'correct_program': '(EXACTLY_1 1 IS_RED)', 'examples': json_count,
                'turn_descriptions': [
                    'Turn 1, Player 0: Step: Turn.LABEL: Quiz mode correct',
                    'Turn 1, Player 0: Step: Turn.GUESS: Guessing rule: test',
                    'Turn 2, Player 0: Step: Turn.LABEL: Quiz mode correct']}))
            torch.save([(None if i == missing else torch.tensor([i]), i % 2 == 0)
                        for i in range(saved_count)], directory / 'examples_0.pt')
            calls = []
            def search(history, *args, **kwargs):
                calls.append([int(t.item()) for t, _ in history])
                return batches[len(calls)-1]
            output = directory / 'result.csv'
            with patch.object(eig, 'run_search_with_probs', side_effect=search), contextlib.redirect_stdout(io.StringIO()):
                eig.analyse_file(path, None, None, 20, 'agent', set(), set(), output)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            done, skipped = eig.load_done_keys(output)
            return calls, rows, done, skipped

    def test_extra_trailing_tensors_ignored(self):
        calls, rows, _, skipped = self.run_game(
            [[(Rule(True), 1)], [(Rule(False), 1)]], saved_count=8)
        self.assertEqual(calls, [[0, 1], [0, 1, 2, 3]])
        self.assertEqual([r['example_idx'] for r in rows], ['2', '4'])
        self.assertFalse(skipped)

    def test_short_tensor_file_still_rejected(self):
        calls, rows, _, skipped = self.run_game([], saved_count=4)
        self.assertEqual(calls, [])
        self.assertEqual(rows[0]['status'], 'data_error')
        self.assertTrue(skipped)

    def test_retained_prefix_must_match_log(self):
        calls, rows, _, skipped = self.run_game([], json_count=4)
        self.assertEqual(calls, [])
        self.assertIn('provenance mismatch', rows[0]['skip_reason'])
        self.assertTrue(skipped)

    def test_fresh_prefix_after_singleton(self):
        calls, rows, done, skipped = self.run_game([
            [(Rule(True), 1)], [(Rule(True), 1), (Rule(False), 1)]])
        self.assertEqual(calls, [[0, 1], [0, 1, 2, 3]])
        self.assertEqual([float(r['eig']) for r in rows], [0, 1])
        self.assertEqual([r['example_idx'] for r in rows], ['2', '4'])
        self.assertEqual(len(done), 2)
        self.assertFalse(skipped)

    def test_empty_batch_stops_later_search(self):
        calls, rows, _, skipped = self.run_game([[], [(Rule(False), 1)]])
        self.assertEqual(len(calls), 1)
        self.assertEqual(rows[0]['status'], 'no_hypotheses')
        self.assertEqual(rows[0]['eig'], '')
        self.assertEqual(rows[0]['skipped_task'], '1')
        self.assertTrue(skipped)

    def test_empty_batch_preserves_earlier_scores(self):
        calls, rows, done, skipped = self.run_game([[(Rule(True), 1)], []])
        self.assertEqual(len(calls), 2)
        self.assertEqual(float(rows[0]['eig']), 0)
        self.assertEqual(rows[1]['status'], 'no_hypotheses')
        self.assertEqual(len(done), 1)
        self.assertTrue(skipped)

    def test_missing_history_never_enters_search(self):
        calls, rows, _, _ = self.run_game([], missing=1)
        self.assertEqual(calls, [])
        self.assertEqual([r['status'] for r in rows], ['incomplete_history']*2)

    def test_missing_current_preserves_indices(self):
        calls, rows, _, _ = self.run_game([], missing=2)
        self.assertEqual(calls, [])
        self.assertEqual([r['example_idx'] for r in rows], ['2', '4'])
        self.assertEqual([r['status'] for r in rows], ['missing_current_example', 'incomplete_history'])

    def test_failed_evaluation_is_not_false_mass(self):
        for invalid in [ValueError('bad'), None]:
            with contextlib.redirect_stdout(io.StringIO()):
                r = eig.compute_eig([(Rule(True), 1), (Rule(invalid), 1)], None, None)
            self.assertTrue(math.isnan(r['eig']))
            self.assertEqual(r['n_eval_true'], 1)
            self.assertEqual(r['n_eval_false'], 0)

    def test_weighted_disagreement_and_boolean_tensor(self):
        with contextlib.redirect_stdout(io.StringIO()):
            r = eig.compute_eig([(Rule(torch.tensor(True)), 3), (Rule(False), 1)], None, None)
        self.assertEqual(r['p_positive'], .75)
        self.assertAlmostEqual(r['eig'], .8112781244591328)
        self.assertEqual((r['n_eval_true'], r['n_eval_false']), (1, 1))

    def test_exhausted_fallback_does_not_append_example(self):
        state = {'turn_descriptions': ['Turn 1, Player 0 proposed no input.'],
                 'game_over_reason': 'Game master examples exhausted after player failed to propose input'}
        self.assertEqual(eig.reconstruct_provenance(state), ['gm', 'gm'])

    def test_invalid_weights_rejected(self):
        for weight in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                eig.compute_eig([(Rule(True), weight)], None, None)

    def test_old_results_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'old.csv'
            path.write_text('player_tag,task_idx,seed,example_idx,skipped_task\na,0,1,2,0\n')
            with self.assertRaisesRegex(ValueError, 'older EIG computation'):
                eig.load_done_keys(path)


if __name__ == '__main__':
    unittest.main()
