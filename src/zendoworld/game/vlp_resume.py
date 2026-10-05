"""Rewind symbolic VLP games with a trailing sequence of timed-out proposals.

Use the final cache for player state, and the published JSON to verify that the
cache belongs to the requested collection (cache filenames omit the model).
"""

from copy import deepcopy
import re


def rewind_timeout_cache(cache, saved):
    """Return an independent cache immediately before the first failed turn.

    This deliberately supports only single-player games whose entire suffix is
    failed proposals. No guesses, tokens, or discovery may have changed there.
    """
    state = cache.state
    history = state.turn_descriptions
    if history != saved['turn_descriptions']:
        raise ValueError('Cache history does not match the source result')
    if (str(state.correct_program) != saved['correct_program']
            or state.turn != saved['turns']
            or len(state.examples) != saved['examples']
            or [str(path) for _, path in state.examples] != saved['paths']):
        raise ValueError('Cache state does not match the source result')
    if len(cache.player_data) != 1 or cache.player_data[0]['id'] != 0:
        raise ValueError('Rewind requires a single player with ID 0')
    player = cache.player_data[0]
    if player['class'] != 'VLPZendoPlayer':
        raise ValueError('Rewind requires a VLPZendoPlayer cache')
    if player.get('discovered_variables') != saved.get('discovered_variables'):
        raise ValueError('Cache vocabulary does not match the source result')
    if {str(k): v for k, v in state.guesses.items()} != saved['guesses']:
        raise ValueError('Cache guesses do not match the source result')
    if {str(k): v for k, v in state.player_guess_tokens.items()} != saved['player_guess_tokens']:
        raise ValueError('Cache tokens do not match the source result')

    failure = re.compile(r'^Turn (\d+), Player 0 proposed no input\.$')
    start = next((i for i, text in enumerate(history) if failure.fullmatch(text)), None)
    if start is None:
        raise ValueError('No failed proposal to rewind')
    suffix = history[start:]
    matches = [failure.fullmatch(text) for text in suffix]
    if not all(matches):
        raise ValueError('Cannot rewind: later turns contain successful actions')
    turns = [int(match[1]) for match in matches]
    first = turns[0]
    if turns != list(range(first, state.turn + 1)):
        raise ValueError('Failed turns are not a contiguous suffix')
    for turn in turns:
        # The final cache is written before the next step records the last
        # duration; the published result includes that final timing.
        duration = saved['turn_durations'].get(str(turn))
        cached_duration = state.turn_durations.get(turn, state.turn_durations.get(str(turn)))
        if cached_duration is not None and cached_duration != duration:
            raise ValueError('Cache timings do not match the source result')
        if duration is None or not 290 <= duration <= 330:
            raise ValueError(f'Turn {turn} does not match the 300-second timeout pattern')

    prefix_count = len(state.examples) - len(turns)
    guesses = sum('Guessing rule:' in text for text in history[:start])
    if prefix_count != 2 + first - 1 + guesses:
        raise ValueError('Example count does not match initial scenes, proposals, and counters')
    if len(player['examples']) != len(state.examples):
        raise ValueError('Player and game example histories differ in length')
    if player.get('discovery_count') != 4 or prefix_count < 4:
        raise ValueError('Rewind requires completed four-example vocabulary discovery')
    if player['guessing_stones'] != state.player_guess_tokens[0]:
        raise ValueError('Player and game guess tokens disagree')

    result = deepcopy(cache)
    state = result.state
    removed = state.examples[prefix_count:]
    result.gm_remaining_examples = removed + result.gm_remaining_examples
    state.examples = state.examples[:prefix_count]
    state.turn_descriptions = history[:start]
    state.turn_durations = {k: v for k, v in state.turn_durations.items() if int(k) < first}
    state.turn = first - 1  # PROPOSE increments before executing the next turn.
    state.current_turn = type(state.current_turn).PROPOSE
    state.won = False
    state.game_over_reason = ''
    state.turn_timer_start = None
    state.last_action = None
    state.input_scene = None
    state.input_scene_rule = None
    state.quiz_mode = False
    result.player_data[0]['examples'] = player['examples'][:prefix_count]
    result.player_data[0]['last_label'] = None
    return result, {'restart_turn': first, 'retained_examples': prefix_count,
                    'discarded_turns': len(turns)}


def restore_symbolic_vlp(cache, gm, player):
    """Restore VLP state without treating symbolic scene descriptions as paths."""
    if not player.symbolic or player.scene_parser != 'symbolic' or not gm.symbolic:
        raise ValueError('This resume path requires symbolic VLP and symbolic parsing')
    if str(gm.true_program) != str(cache.state.correct_program):
        raise ValueError('Dataset rule differs from the cached rule')
    pdata = cache.player_data[0]
    for name in ('guessing_stones', 'incorrect_rules', 'previous_guesses',
                 'already_guessed', 'last_label', 'discovered_variables'):
        setattr(player, name, deepcopy(pdata[name]))
    player._discovery_count = pdata['discovery_count']
    player._rebuild_dsl_from_variables()
    player.top_guess = None
    player.second_guess = None
    # Rebuild observations from authoritative tensors, preserving old labels.
    # Do not call observe(): that could trigger new vocabulary discovery.
    player.examples = [gm.format_for_player(example) for example in cache.state.examples]
    gm.remaining_examples = list(cache.gm_remaining_examples)
    gm._used_flat_example_paths.update(str(path) for _, path in cache.state.examples)
    return cache.state


def continue_symbolic_vlp(cache, gm, player, path):
    from zendoworld.game.states import Turn, save_game, step

    state = restore_symbolic_vlp(cache, gm, player)
    save_game(state, gm, [player], filename=path)
    while state.current_turn != Turn.END:
        state = step(state, [player], gm, path=path)
    return state
