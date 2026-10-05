"""Exercise the real step function without importing model backends."""
import ast
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def step_fixture(count=28, tokens=13, correct=False, has_counter=True):
    source = Path(__file__).resolve().parents[1] / 'src/zendoworld/game/states.py'
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in {'Turn', 'step'}]
    save = Mock()
    ns = dict(Enum=Enum, auto=auto, GameState=SimpleNamespace,
              ZendoPlayerInterface=object, ZendoStateGameMaster=object,
              time=SimpleNamespace(time=lambda: 100.0), save_game=save)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    turn = ns['Turn']
    state = SimpleNamespace(examples=list(range(count)), max_examples=30,
                            current_turn=turn.GUESS, turn=20, player=0,
                            turn_timer_start=10.0, turn_durations={},
                            player_guess_tokens={0: tokens}, guesses={0: []},
                            turn_descriptions=[], won=False, game_over_reason='')
    player = Mock(id=0)
    player.react.return_value = {'type': 'guess', 'rule': 'candidate'}
    gm = Mock(true_program='target')
    gm.check_guess.return_value = (correct, 'converted')
    gm.disprove_guess.return_value = (('scene', True), 'image.png') if has_counter else None
    return ns['step'], turn, state, player, gm, save


class GuessBudgetTests(unittest.TestCase):
    def test_accumulated_tokens_cannot_overrun_budget(self):
        for count, tokens in ((28, 13), (29, 13), (29, 1)):
            with self.subTest(count=count, tokens=tokens):
                step, turn, state, player, gm, save = step_fixture(count, tokens)
                result = step(state, [player], gm, path='test-cache')
                allowed = 30 - count
                self.assertIs(result, state)
                self.assertEqual(len(state.examples), 30)
                self.assertEqual(player.react.call_count, allowed)
                self.assertEqual(gm.disprove_guess.call_count, allowed)
                self.assertEqual(player.observe.call_count, allowed)
                self.assertEqual(len(state.guesses[0]), allowed)
                self.assertEqual(state.player_guess_tokens[0], tokens - allowed)
                self.assertEqual(state.current_turn, turn.END)
                self.assertFalse(state.won)
                self.assertEqual(state.game_over_reason, 'Max examples reached')
                self.assertEqual(state.turn_durations[20], 90.0)
                self.assertIsNone(state.turn_timer_start)
                save.assert_called_once_with(state, gm, [player], filename='test-cache')

    def test_token_exhaustion_below_budget_continues_game(self):
        step, turn, state, player, gm, save = step_fixture(count=28, tokens=1)
        step(state, [player], gm)
        self.assertEqual(len(state.examples), 29)
        self.assertEqual(state.current_turn, turn.PROPOSE)
        self.assertFalse(state.won)
        save.assert_called_once()

    def test_correct_guess_before_budget_still_wins(self):
        step, turn, state, player, gm, save = step_fixture(count=29, correct=True)
        step(state, [player], gm)
        self.assertTrue(state.won)
        self.assertEqual(len(state.examples), 29)
        self.assertEqual(state.current_turn, turn.END)
        gm.disprove_guess.assert_not_called()
        save.assert_called_once()

    def test_absent_counterexample_still_wins(self):
        step, turn, state, player, gm, save = step_fixture(count=29, has_counter=False)
        step(state, [player], gm)
        self.assertTrue(state.won)
        self.assertEqual(len(state.examples), 29)
        self.assertEqual(state.current_turn, turn.END)
        save.assert_called_once()

    def test_missing_timer_does_not_allow_overrun(self):
        step, turn, state, player, gm, save = step_fixture(count=29)
        state.turn_timer_start = None
        step(state, [player], gm)
        self.assertEqual(len(state.examples), 30)
        self.assertEqual(state.current_turn, turn.END)
        self.assertEqual(state.turn_durations, {})
        save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
