from zendo.game_master import ZendoStateGameMaster
from zendo.player import ZendoPlayerInterface
from zendo.states import GameCache, GameState, Turn, step
from zendo.scientist_player import LLMScientistPlayer
import os

def difficulty(program: str) -> str:
    if program is None:
        return "unknown"
    prog = str(program)
    if "INTERACTION" in prog or "AND " in prog or "(OR " in prog:
        return "difficult"
    elif "IS_GROUNDED" in prog or "IS_UNGROUNDED" in prog or "MORE_THAN" in prog or "EVEN_" in prog or "ODD_" in prog or "SAME_" in prog:
        return "medium"
    else:
        return "easy"


def play_game_state(gm: ZendoStateGameMaster, players: list[ZendoPlayerInterface], cached=False, path="zendo_cache.pkl") -> GameState:
    diff = difficulty(str(gm.true_program))
    state = GameState(
        correct_program=str(gm.true_program),
        difficulty=diff,
        examples=[],
        guesses={i: [] for i in range(len(players))},
        examples_proposed={i: 0 for i in range(len(players))},
        player_guess_tokens={i: 0 for i in range(len(players))},
        current_turn=Turn.PROPOSE,
        last_action=None
    )
    if cached and path != "":
        cache_file=path
        cache = GameCache.from_file(cache_file)
        state = cache.state
        gm.remaining_examples = cache.gm_remaining_examples
        cleaned_examples = []
        for p, pdata in zip(players, cache.player_data):
            for (example, path) in pdata["examples"]:
                if example[0] is None or example[0] == "" or not os.path.exists(path):
                    print("skipping example: ", path)
                    continue
                cleaned_examples.append((example, path))
            p.guessing_stones = pdata["guessing_stones"]
            p.incorrect_rules = pdata["incorrect_rules"]
            p.previous_guesses = pdata.get("previous_guesses", [])
            p.last_label = pdata["last_label"]
            p.examples = cleaned_examples
            if pdata["class"] == "LLMScientistPlayer":
                p.particles = pdata.get("particles", [])
                p.particle_weights = pdata.get("particle_weights", [])
                p.first_propose = pdata.get("first_propose", True)
                p.pf_checkpoint = pdata.get("pf_checkpoint", None)
                p._pf_dirty = pdata.get("_pf_dirty", False)
                p._pf_last_len = pdata.get("_pf_last_len", 0)
                p._pf_support_cache = pdata.get("_pf_support_cache", {})
                p.program_cache = pdata.get("program_cache", {})
                p.label_cache = pdata.get("label_cache", {})
                p.cache_h = pdata.get("cache_h", None)
                p.thetas = pdata.get("thetas", None)
                p.deltas = pdata.get("deltas", None)
                p.all_priors = pdata.get("all_priors", None)
                p.already_guessed = pdata.get("already_guessed", [])
            if pdata["class"] in ("VLPZendoPlayer", "VLPUncertaintyPlayer"):
                p._discovery_count = pdata.get("discovery_count", 0)
                p.discovered_variables = pdata.get("discovered_variables", {
                    "objects": [],
                    "properties": [],
                    "actions": [],
                })
    else:
        for ex in gm.initial_examples():
            player_view = gm.format_for_player(ex)
            for p in players:
                p.observe(player_view)
            state.examples.append(ex)

    while state.current_turn != Turn.END:
        state = step(state, players, gm, path=path)

    print("Finished: ", state.game_over_reason)
    return state

def play_bramley_game(gm: ZendoStateGameMaster, players: list[ZendoPlayerInterface]) -> GameState:
    diff = difficulty(str(gm.true_program))
    state = GameState(
        correct_program=str(gm.true_program),
        difficulty=diff,
        examples=[],
        guesses={i: [] for i in range(len(players))},
        examples_proposed={i: 0 for i in range(len(players))},
        player_guess_tokens={i: 0 for i in range(len(players))},
        current_turn=Turn.PROPOSE,
        last_action=None
    )

    ex = gm.initial_example()
    player_view = gm.format_for_player(ex)
    for p in players:
        p.observe(player_view)
    state.examples.append(ex)

    while state.current_turn != Turn.END:
        state = step(state, players, gm, bramley=True)

    print("Finished: ", state.game_over_reason)
    return state