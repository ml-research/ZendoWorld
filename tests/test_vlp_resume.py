"""Rewind/restore checks independent of model and tensor dependencies."""
from copy import deepcopy
from enum import Enum
from types import SimpleNamespace
import unittest

from zendoworld.game.vlp_resume import rewind_timeout_cache, restore_symbolic_vlp


class Turn(Enum):
    PROPOSE = 1
    END = 5


def fixture():
    # Two initial scenes, one proposal and counter at turn 1, two skipped turns.
    examples = [((f'tensor{i}', bool(i % 2)), f'missing-image-{i}') for i in range(6)]
    history = ['Turn 1, Player 0 proposed input based on rule: old_rule',
               'Turn 1, Player 0: Step: Turn.GUESS: Guessing rule: old_rule, converted to converted',
               'Turn 2, Player 0 proposed no input.',
               'Turn 3, Player 0 proposed no input.']
    vocabulary = dict(objects=['wedge'], properties=['flat'], actions=['on_top_of'])
    state = SimpleNamespace(correct_program='true_rule', turn=3, examples=examples,
                            turn_descriptions=history, turn_durations={1: 4., 2: 300.5},
                            guesses={0: ['converted']}, player_guess_tokens={0: 0},
                            current_turn=Turn.END, won=False, game_over_reason='Max examples reached',
                            examples_proposed={0: 1})
    player = dict(id=0, **{'class': 'VLPZendoPlayer'}, examples=deepcopy(examples),
                  guessing_stones=0, incorrect_rules=['converted'], previous_guesses=[],
                  already_guessed=['old_rule'], last_label='',
                  discovery_count=4, discovered_variables=vocabulary)
    cache = SimpleNamespace(state=state, player_data=[player],
                            gm_remaining_examples=[(('future', True), 'future-path')])
    saved = dict(correct_program='true_rule', turns=3, examples=6,
                 paths=[p for _, p in examples], turn_descriptions=history.copy(),
                 turn_durations={'1': 4., '2': 300.5, '3': 300.6},
                 guesses={'0': ['converted']}, player_guess_tokens={'0': 0},
                 discovered_variables=deepcopy(vocabulary))
    return cache, saved


class ResumeTests(unittest.TestCase):
    def test_rewind_preserves_prefix_and_returns_fallbacks_to_front(self):
        cache, saved = fixture()
        before = deepcopy(cache)
        result, info = rewind_timeout_cache(cache, saved)
        self.assertEqual(cache, before)
        self.assertEqual(info, dict(restart_turn=2, retained_examples=4, discarded_turns=2))
        self.assertEqual(result.state.turn, 1)
        self.assertEqual(result.state.current_turn, Turn.PROPOSE)
        self.assertEqual(result.state.examples, cache.state.examples[:4])
        self.assertEqual(result.gm_remaining_examples, cache.state.examples[4:] + cache.gm_remaining_examples)
        self.assertEqual(result.state.turn_durations, {1: 4.})
        self.assertEqual(result.player_data[0]['already_guessed'], ['old_rule'])
        self.assertEqual(result.player_data[0]['incorrect_rules'], ['converted'])
        self.assertIsNone(result.player_data[0]['last_label'])

    def test_refuses_wrong_collection(self):
        cache, saved = fixture()
        saved['turn_descriptions'] = ['different run']
        with self.assertRaisesRegex(ValueError, 'history'):
            rewind_timeout_cache(cache, saved)

    def test_refuses_non_timeout_failure(self):
        cache, saved = fixture()
        saved['turn_durations']['3'] = 12.
        with self.assertRaisesRegex(ValueError, '300-second'):
            rewind_timeout_cache(cache, saved)

    def test_refuses_successful_action_after_failure(self):
        cache, saved = fixture()
        cache.state.turn_descriptions[-1] = 'Turn 3, Player 0 proposed input based on rule: new'
        saved['turn_descriptions'] = cache.state.turn_descriptions.copy()
        with self.assertRaisesRegex(ValueError, 'successful actions'):
            rewind_timeout_cache(cache, saved)

    def test_refuses_inconsistent_example_count(self):
        cache, saved = fixture()
        cache.state.examples.append((('extra', True), 'extra'))
        saved['examples'] += 1
        saved['paths'].append('extra')
        with self.assertRaisesRegex(ValueError, 'Example count'):
            rewind_timeout_cache(cache, saved)

    def test_refuses_unfinished_discovery(self):
        cache, saved = fixture()
        cache.player_data[0]['discovery_count'] = 3
        with self.assertRaisesRegex(ValueError, 'discovery'):
            rewind_timeout_cache(cache, saved)

    def test_restore_keeps_symbolic_examples_and_rebuilds_dsl(self):
        cache, saved = fixture()
        cache, _ = rewind_timeout_cache(cache, saved)
        rebuilt = []
        player = SimpleNamespace(symbolic=True, scene_parser='symbolic',
                                 _rebuild_dsl_from_variables=lambda: rebuilt.append(True))
        gm = SimpleNamespace(symbolic=True, true_program='true_rule',
                             _used_flat_example_paths=set(),
                             format_for_player=lambda ex: ((f'scene:{ex[0][0]}', ex[0][1]), 'symbolic scene'))
        state = restore_symbolic_vlp(cache, gm, player)
        self.assertIs(state, cache.state)
        self.assertEqual(rebuilt, [True])
        self.assertEqual(len(player.examples), 4)
        self.assertEqual(player.examples[0][0][0], 'scene:tensor0')
        self.assertEqual(player.already_guessed, ['old_rule'])
        self.assertEqual(player._discovery_count, 4)
        self.assertEqual(gm.remaining_examples, cache.gm_remaining_examples)
        self.assertNotIn('missing-image-4', gm._used_flat_example_paths)
        self.assertIsNone(player.top_guess)


if __name__ == '__main__':
    unittest.main()
