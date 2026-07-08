from dataclasses import dataclass, field
from enum import Enum, auto
import os
import pickle
from typing import Any
import json
import time
from program import Program
from zendo.game_master import ZendoStateGameMaster
from zendo.player import ZendoPlayerInterface
import random
import numpy as np

class Turn(Enum):
      PROPOSE = auto()
      LABEL = auto()
      GUESS = auto()
      GUESS_BRAMLEY = auto()
      END = auto()

def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if is_json_serializable(v)}
    elif isinstance(value, list):
        return [sanitize(v) for v in value if is_json_serializable(v)]
    elif is_json_serializable(value):
        return value
    return str(value)
        
@dataclass
class GameState:
      correct_program: str | None
      difficulty: str
      examples: list[tuple]
      examples_proposed: dict[int, int]
      guesses: dict[int, list[str]]
      player_guess_tokens: dict[int, int]  # player_id -> tokens
      current_turn: Turn
      last_action: dict | None
      input_scene: Any | None = None
      input_scene_rule: str | None = None
      quiz_mode: bool = False
      player_label_guesses: dict[int, bool] = field(default_factory=dict)
      bramley_guesses: dict[int, list[list[bool], list[bool], list[bool]]] = field(default_factory=dict)
      bramley_rule: str | None = None
      bramley_test_examples: list[tuple] = field(default_factory=list)
      won: bool = False
      max_examples: int = 30
      game_over_reason: str = ""
      turn_timer_start: float = None
      turn_durations: dict[int, float] = field(default_factory=dict)
      turn = 0
      player = 0
      turn_descriptions: list[str] = field(default_factory=list)
      def to_dict(self):
            (examples, paths) = zip(*self.examples)
            (bramley_examples, bramley_paths) = zip(*self.bramley_test_examples) if self.bramley_test_examples else ([], [])
            return [{
                  "correct_program": str(self.correct_program) if self.correct_program else None,
                  "difficulty": self.difficulty,
                  "turns": self.turn,
                  "examples": len(self.examples),
                  "guesses": self.guesses,
                  "player_guess_tokens": self.player_guess_tokens,
                  "last_action": sanitize(self.last_action),
                  "player_label_guesses": self.player_label_guesses,
                  "won": self.won,
                  "max_examples": self.max_examples,
                  "game_over_reason": self.game_over_reason,
                  "paths": [str(p) for p in paths],
                  "bramley_test_examples": [str(p) for p in bramley_paths],
                  "turn_durations": {str(k): v for k, v in self.turn_durations.items()},
                  "turn_descriptions": self.turn_descriptions,
                  "bramley": self.bramley_guesses,
                  "bramley_rule": self.bramley_rule,
            }, examples, bramley_examples]

def step(state: GameState, players: list[ZendoPlayerInterface], gm: ZendoStateGameMaster, bramley=False, path="zendo_cache.pkl") -> GameState:
      if len(state.examples) >= state.max_examples:
            state.current_turn = Turn.END
            duration = time.time() - state.turn_timer_start
            state.turn_durations[state.turn] = duration
            print(f"⏱️ Turn {state.turn} duration: {duration:.2f} seconds")
            state.game_over_reason = "Max examples reached"
            return state
      if state.current_turn in (Turn.PROPOSE, Turn.END):
        if state.turn_timer_start is not None:
            duration = time.time() - state.turn_timer_start
            state.turn_durations[state.turn] = duration
            print(f"⏱️ Turn {state.turn} duration: {duration:.2f} seconds")
            state.turn_timer_start = None
      if state.current_turn == Turn.PROPOSE:
            state.turn += 1
            state.player = state.turn % len(players)
            if bramley and state.turn > 7:
                  state.current_turn = Turn.GUESS_BRAMLEY
                  return state
            print(f"========Turn: {state.turn}, Player: {state.player}========")
            state.turn_timer_start = time.time()
            proposer = players[state.player]
            action = proposer.react(state)
            state.last_action = action
            print(f"Player {proposer.id} action: {action}")
            if action["input"] is None:
                  print(f"Player {proposer.id} proposed no input, skipping turn")
                  turn_description = f"Turn {state.turn}, Player {state.player} proposed no input."
                  state.turn_descriptions.append(turn_description)
                  state.current_turn = Turn.PROPOSE
                  example = gm.get_next_example()
                  player_view = gm.format_for_player(example)
                  for i, p in enumerate(players):
                        p.observe(player_view)
                  state.examples.append(example)
                  save_game(state, gm, players, filename=path)
                  return state

            if action["type"] == "propose_input":
                  state.input_scene = action["input"]
                  state.examples_proposed[proposer.id] = state.examples_proposed.get(proposer.id, 0) + 1
                  state.quiz_mode = action["mode"] == "QUIZ"
                  state.input_scene_rule = action.get("rule", "")
                  if state.input_scene_rule != "":
                        turn_description = f"Turn {state.turn}, Player {state.player} proposed input based on rule: {state.input_scene_rule}"
                        state.turn_descriptions.append(turn_description)
                  state.current_turn = Turn.LABEL
                  if bramley:
                        if state.turn <= 7:
                              state.quiz_mode = False

      elif state.current_turn == Turn.LABEL:
            label = gm.label_input(state.input_scene[0])
            state.examples.append(((state.input_scene[0], label), state.input_scene[1]))

            if state.quiz_mode:
                  print("QUIZ mode: players guessing label")
                  num_players = len(players)
                  for i in range(num_players):
                        player_index = (state.player + i) % num_players
                        p = players[player_index]
                        guess = p.react(state)["label"]
                        correct = (guess == label)
                        state.player_label_guesses[p.id] = correct
                        if correct:
                              turn_description = f"Turn {state.turn}, Player {i}: Step: {state.current_turn}: Quiz mode correct, guessed {guess}"
                              state.turn_descriptions.append(turn_description)
                              print(f"Player {i} guessed correctly: {guess}")
                              state.player_guess_tokens[p.id] = state.player_guess_tokens.get(p.id, 0) + 1
                              p.quiz_correct()
                        else:
                              p.quiz_incorrect()
                              turn_description = f"Turn {state.turn}, Player {i}: Step: {state.current_turn}: Quiz mode incorrect, guessed {guess}, correct was {label}"
                              state.turn_descriptions.append(turn_description)
                              print(f"Player {i} guessed incorrectly: {guess}")
                        p.observe(((state.input_scene[0], label), state.input_scene[1]))
                  if len(state.examples) >= state.max_examples:
                        state.current_turn = Turn.END
                        duration = time.time() - state.turn_timer_start
                        state.turn_durations[state.turn] = duration
                        print(f"⏱️ Turn {state.turn} duration: {duration:.2f} seconds")
                        state.game_over_reason = "Max examples reached"
                  else:
                        state.current_turn = Turn.GUESS
            else:
                  turn_description = f"Turn {state.turn}, Player {state.player}: Step: {state.current_turn}: Tell mode"
                  state.turn_descriptions.append(turn_description)
                  print("TELL mode: GM reveals label")
                  for p in players:
                        p.observe(((state.input_scene[0], label), state.input_scene[1]))
             
                  if len(state.examples) >= state.max_examples:
                        state.current_turn = Turn.END
                        duration = time.time() - state.turn_timer_start
                        state.turn_durations[state.turn] = duration
                        print(f"⏱️ Turn {state.turn} duration: {duration:.2f} seconds")
                        state.game_over_reason = "Max examples reached"
                  else:
                        state.current_turn = Turn.GUESS
                        if bramley:
                              state.current_turn = Turn.PROPOSE

      elif state.current_turn == Turn.GUESS:
            print("Players guessing rules")
            p = players[state.player]
            print(f"Player {p.id} has {state.player_guess_tokens.get(p.id, 0)} guess tokens")
            while state.player_guess_tokens.get(p.id, 0) > 0:
                  guess_action = p.react(state)
                  if guess_action is None or guess_action.get("type") == "no_guess":
                        turn_description = f"Turn {state.turn}, Player {state.player}: Step: {state.current_turn}: Not guessing rule"
                        state.turn_descriptions.append(turn_description)
                        break
                  
                  rule = guess_action["rule"]
                  correct, converted_rule = gm.check_guess(rule)
                  turn_description = f"Turn {state.turn}, Player {state.player}: Step: {state.current_turn}: Guessing rule: {str(rule)}, converted to {str(converted_rule)}"
                  state.turn_descriptions.append(turn_description)
                  print(f"Player {p.id} guessed: {rule}, correct: {correct}, correct rule: {gm.true_program}")
                  state.guesses[p.id].append(str(converted_rule))
                  state.player_guess_tokens[p.id] -= 1

                  if correct:
                        state.won = True
                        state.current_turn = Turn.END
                        state.game_over_reason = f"Player {p.id} guessed rule correctly"
                        duration = time.time() - state.turn_timer_start
                        state.turn_durations[state.turn] = duration
                        print(f"⏱️ Turn {state.turn} duration: {duration:.2f} seconds")
                        state.turn_timer_start = None
                        save_game(state, gm, players, filename=path)
                        return state
                  else:
                        p.wrong_rule(converted_rule)
                        counter = gm.disprove_guess(converted_rule)
                        print(f"Counter example for guess {converted_rule}: {counter}")
                        if counter:
                              counter_view = gm.format_for_player(counter)
                              for _, ps in enumerate(players):
                                    ps.observe(counter_view)
                              state.examples.append(counter)
                        else:
                              print(f"No counter example found for guess: Player won")
                              state.won = True
                              state.current_turn = Turn.END
                              state.game_over_reason = f"Player {p.id} guessed different rule but no counter example found"
                              duration = time.time() - state.turn_timer_start
                              state.turn_durations[state.turn] = duration
                              print(f"⏱️ Turn {state.turn} duration: {duration:.2f} seconds")
                              state.turn_timer_start = None
                              save_game(state, gm, players, filename=path)
                              return state

            state.current_turn = Turn.PROPOSE
       
      elif state.current_turn == Turn.GUESS_BRAMLEY:
            print("Players guessing rules")
            test_examples = gm.test_scenes()
            state.bramley_test_examples = test_examples
            random.shuffle(test_examples)
            examples, path = zip(*test_examples)
            tensors, labels = zip(*examples)
            num_players = len(players)
            for i in range(num_players):
                  player_index = (state.player + i) % num_players
                  p = players[player_index]
                  correct_guesses = []
                  guessed_labels, rule = p.guess_labels(path)
                  for guessed_label, label in zip(guessed_labels, labels):
                      correct_guesses.append(guessed_label == label)
                  state.bramley_guesses[p.id] = [labels, guessed_labels, correct_guesses]
                  state.bramley_rule = str(rule)
            state.current_turn = Turn.END
      save_game(state, gm, players, filename=path)
      return state

@dataclass
class GameCache:
    state: GameState
    gm_remaining_examples: list
    player_data: list[dict]

    def to_file(self, filename="zendo_cache.pkl"):
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, filename="zendo_cache.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)

def safe_get(obj, attr, default=None):
    return getattr(obj, attr, default)
       
def save_game(state, gm, players, filename="zendo_cache.pkl"):
      player_data = []

      for p in players:
            data = {
                  "id": p.id,
                  "class": p.__class__.__name__,
                  "guessing_stones": safe_get(p, "guessing_stones"),
                  "incorrect_rules": safe_get(p, "incorrect_rules"),
                  "last_label": safe_get(p, "last_label"),
                  "examples": safe_get(p, "examples"),
                  "previous_guesses": safe_get(p, "previous_guesses"),
                  "already_guessed": safe_get(p, "already_guessed"),
            }

            if hasattr(p, "particles"):
                  data["particles"] = list(p.particles) if isinstance(p.particles, np.ndarray) else p.particles
                  data["particle_weights"] = list(p.particle_weights) if hasattr(p, "particle_weights") else None
                  data["first_propose"] = safe_get(p, "first_propose")
                  data["pf_checkpoint"] = safe_get(p, "pf_checkpoint")
                  data["_pf_dirty"] = safe_get(p, "_pf_dirty")
                  data["_pf_last_len"] = safe_get(p, "_pf_last_len")
                  data["_pf_support_cache"] = safe_get(p, "_pf_support_cache")
                  data["thetas"] = safe_get(p, "thetas")
                  data["deltas"] = safe_get(p, "deltas")
                  data["all_priors"] = safe_get(p, "all_priors")
                  data["already_guessed"] = safe_get(p, "already_guessed")

            if hasattr(p, "program_cache"):
                  data["program_cache"] = p.program_cache
            if hasattr(p, "label_cache"):
                  data["label_cache"] = p.label_cache
            if hasattr(p, "cache_h"):
                  data["cache_h"] = p.cache_h
            if hasattr(p, "discovered_variables"):
                 data["discovered_variables"] = p.discovered_variables
            if hasattr(p, "_discovery_count"):
                 data["discovery_count"] = p._discovery_count


            player_data.append(data)
            
      cache = GameCache(
            state=state,
            gm_remaining_examples=gm.remaining_examples,
            player_data=player_data
      )
      cache.to_file(filename)