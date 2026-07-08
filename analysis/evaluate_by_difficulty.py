import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style setup (copied from evaluate_players.py)
# ---------------------------------------------------------------------------

colors = [
    "#A8E6A3", "#7FD1B9", "#80BDE3", "#A3A8E6",
    "#D4A3E6", "#F4A3C0", "#F4A3A3", "#F4CBA3", "#E9EB8C",
]


def _load_player_design() -> dict:
    path = Path(__file__).parent.parent / "player_design.json"
    with open(path) as f:
        return json.load(f)


_DESIGN = _load_player_design()


def _player_color(name: str, fallback_idx: int = 0) -> str:
    return _DESIGN.get(name, {}).get("color", colors[fallback_idx % len(colors)])


def _player_hatch(name: str):
    return _DESIGN.get(name, {}).get("hatch", None)


def neurips_fig():
    inches_per_pt = 1 / 72.27
    golden = (1 + 5 ** 0.5) / 2
    width_pt = 246
    return (width_pt * inches_per_pt, width_pt * inches_per_pt / golden)


_COL_W, _COL_H = neurips_fig()
GUTTER_IN = 0.375
FULL_WIDTH_IN = _COL_W * 2 + GUTTER_IN

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'savefig.dpi': 300,
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

EDGE_COLOR = "black"
EDGE_LINEWIDTH = 0.6
DIFFICULTIES = ["first_order", "second_order", "complex", "ood"]

# ---------------------------------------------------------------------------
# Difficulty classification from correct_program string
# ---------------------------------------------------------------------------

def classify_difficulty(program: str) -> str:
    """Derive difficulty class from the correct_program string.

    - ood:           contains SAME_AMOUNT (out-of-distribution predicate)
    - complex: contains AND / OR / relational predicates (_INTERACTION, TOUCHING)
    - second_order:  contains EVEN, ODD, MORE_THAN  (numeric comparisons)
    - first_order:   contains AT_LEAST, EXACTLY, ZERO  (counting quantifiers)
    - simple:        everything else (single attribute predicates)

    Priority: ood > complex > second_order > first_order > simple
    """
    if program is None:
        return "unknown"
    if "SAME_AMOUNT" in program:
        return "ood"
    if (re.search(r'\bAND\b', program) or re.search(r'\bOR\b', program)
            or "_INTERACTION" in program or "EITHER_OR" in program):
        return "complex"
    if any(tok in program for tok in ("EVEN", "ODD", "MORE_THAN")):
        return "second_order"
    return "first_order"


def classify_difficulty_no_ood(program: str) -> str:
    """Difficulty classification without a separate OOD bucket.

    Identical to ``classify_difficulty`` except SAME_AMOUNT is folded into
    ``second_order``. Use this for plots that do not show OOD as its own
    category; tables should keep using ``classify_difficulty``.
    """
    diff = classify_difficulty(program)
    return "second_order" if diff == "ood" else diff


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def evaluate_single_file(filepath: Path) -> dict:
    with open(filepath) as f:
        data = json.load(f)

    won = bool(data.get("won", False))
    turns = data.get("turns", 0)
    correct_program = data.get("correct_program") or data.get("correct_Program")
    difficulty = classify_difficulty(correct_program)

    return {
        "won": 1.0 if won else 0.0,
        "turns": float(turns) if turns is not None else None,
        "difficulty": difficulty,
        "correct_program": correct_program,
    }


def average_records(records: list) -> dict | None:
    if not records:
        return None

    diff = Counter(r["difficulty"] for r in records).most_common(1)
    difficulty = diff[0][0] if diff else "unknown"

    won = sum(r["won"] for r in records) / len(records)

    won_records = [r for r in records if r["won"] > 0.0 and r["turns"] is not None]
    avg_turns = (sum(r["turns"] for r in won_records) / len(won_records)) if won_records else None

    return {
        "won": won,
        "turns": avg_turns,
        "difficulty": difficulty,
        "n_files": len(records),
    }


def load_player_averaged(player_dir: Path) -> dict:
    """Return {filename: averaged_metrics} handling both flat and seed-subdir layouts."""
    subdirs = sorted(p for p in player_dir.iterdir() if p.is_dir())
    has_reps = any(list(sd.glob("*.json")) for sd in subdirs)

    per_file = defaultdict(list)
    if not has_reps:
        for f in player_dir.glob("*.json"):
            per_file[f.name].append(evaluate_single_file(f))
    else:
        for sd in subdirs:
            for f in sd.glob("*.json"):
                try:
                    per_file[f.name].append(evaluate_single_file(f))
                except Exception as e:
                    print(f"Error reading {f}: {e}")

    return {fname: average_records(recs) for fname, recs in per_file.items()}


# ---------------------------------------------------------------------------
# Aggregate into a DataFrame
# ---------------------------------------------------------------------------

def build_dataframe(dirs: list[tuple[str, Path]]) -> pd.DataFrame:
    # Load all players
    all_data = {name: load_player_averaged(d) for name, d in dirs}

    # Find shared files
    shared = set(next(iter(all_data.values())).keys())
    for player_files in all_data.values():
        shared &= player_files.keys()
    print(f"Shared files: {len(shared)}")

    rows = []
    for name, player_files in all_data.items():
        by_diff = defaultdict(lambda: {"wins": 0.0, "wins_list": [], "total": 0, "turns": []})
        for fname in shared:
            rec = player_files[fname]
            if rec is None:
                continue
            diff = rec["difficulty"]
            if diff == "unknown":
                continue
            by_diff[diff]["wins"] += rec["won"]
            by_diff[diff]["wins_list"].append(rec["won"])
            by_diff[diff]["total"] += 1
            if rec["turns"] is not None:
                by_diff[diff]["turns"].append(rec["turns"])

        for diff in DIFFICULTIES:
            s = by_diff[diff]
            total = s["total"]
            win_rate = (s["wins"] / total) if total > 0 else 0.0
            wins_vals = s["wins_list"]
            turns_vals = s["turns"]

            def _ci95(vals):
                n = len(vals)
                if n < 2:
                    return 0.0
                return 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(n)

            rows.append({
                "player": name,
                "difficulty": diff,
                "win_rate": win_rate,
                "avg_turns": (sum(turns_vals) / len(turns_vals)) if turns_vals else 0.0,
                "win_rate_ci": _ci95(wins_vals),
                "avg_turns_ci": _ci95(turns_vals),
                "n_tasks": total,
                "n_wins": int(s["wins"]),
            })

    df = pd.DataFrame(rows)
    df["difficulty"] = pd.Categorical(df["difficulty"], categories=DIFFICULTIES, ordered=True)
    return df.sort_values(["player", "difficulty"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grouped_bars(ax, df, y_col, players, group_width=0.75, ci_col=None):
    diffs = DIFFICULTIES
    n_groups = len(diffs)
    n_bars = len(players)
    bar_w = group_width / max(n_bars, 1)
    base_x = np.arange(n_groups)

    proxy_handles, proxy_labels = [], []
    for i, p in enumerate(players):
        sub = df[df["player"] == p]
        ys, cis = [], []
        for d in diffs:
            row = sub[sub["difficulty"] == d]
            ys.append(float(row[y_col].values[0]) if not row.empty else 0.0)
            if ci_col and ci_col in df.columns:
                cis.append(float(row[ci_col].values[0]) if not row.empty else 0.0)

        xs = base_x + (i - (n_bars - 1) / 2) * bar_w
        face = _player_color(p, i)
        hatch = _player_hatch(p)
        if cis:
            # Asymmetric error bars: clip lower end so whiskers never go below 0
            yerr = [
                [min(y, ci) for y, ci in zip(ys, cis)],
                cis,
            ]
        else:
            yerr = None

        ax.bar(xs, ys, width=bar_w, color=face, edgecolor=EDGE_COLOR,
               linewidth=EDGE_LINEWIDTH, hatch=hatch, zorder=3,
               yerr=yerr, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "black"})

        proxy_handles.append(
            mpatches.Patch(facecolor=face, edgecolor=EDGE_COLOR,
                           linewidth=EDGE_LINEWIDTH, hatch=hatch, label=p)
        )
        proxy_labels.append(p)

    ax.set_xticks(base_x)
    ax.set_xticklabels([d.capitalize() for d in diffs], fontsize=7)
    ax.margins(y=0.12)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, zorder=0)

    return proxy_handles, proxy_labels


def plot_win_and_turns_by_difficulty(dirs: list[tuple[str, Path]], save_dir: str = "figs",
                                     suffix: str = ""):
    df = build_dataframe(dirs)
    players = [name for name, _ in dirs]

    fig_w = FULL_WIDTH_IN
    fig_h = fig_w * 0.45
    fig, (ax_win, ax_turns) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    # Panel A: Win rate
    handles, labels = plot_grouped_bars(ax_win, df, "win_rate", players, ci_col="win_rate_ci")
    ax_win.set_title("(A) Win rate by difficulty", fontsize=9, pad=5)
    ax_win.set_ylabel("Win rate", fontsize=8)
    ax_win.set_ylim(0, 1.05)
    ax_win.spines["top"].set_visible(False)
    ax_win.spines["right"].set_visible(False)

    # Panel B: Average turns
    plot_grouped_bars(ax_turns, df, "avg_turns", players, ci_col="avg_turns_ci")
    ax_turns.set_title("(B) Average turns by difficulty", fontsize=9, pad=5)
    ax_turns.set_ylabel("Average turns", fontsize=8)
    ax_turns.set_ylim(bottom=0)
    ax_turns.spines["top"].set_visible(False)
    ax_turns.spines["right"].set_visible(False)

    # Legend below figure
    legend = fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=min(len(players), 2),
        frameon=True,
        facecolor="whitesmoke",
        fontsize=7,
        handlelength=1.2,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.97])

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_base = str(Path(save_dir) / f"win_turns_by_difficulty{suffix}")
    fig.canvas.draw()
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))
    legend.remove()
    plt.close(fig)
    print("Saved:", out_base + ".pdf", out_base + ".png")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DIRS_DEFAULT = [
    ("Oracle Agent",    Path("gamestates/gamestates_zendo")),
    ("VLP Agent",       Path("gamestates/gamestates_vlp_old_dsl")),
    ("Bayesian Agent", Path("gamestates/gamestates_scientist")),
    ("VLM Agent",       Path("gamestates/gamestates_fullgpt")),
]

_DIRS_SYMBOLIC = [
    ("Oracle Agent",              Path("gamestates/gamestates_zendo_symbolic")),
    ("Random Oracle Agent Symbolic", Path("gamestates/gamestates_random_zendo_symbolic")),
    ("VLP Agent",                 Path("gamestates/gamestates_vlp_symbolic")),
    ("Bayesian Agent",           Path("gamestates/gamestates_scientist_symbolic")),
    ("VLM Agent",                 Path("gamestates/gamestates_fullgpt_symbolic")),
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbolic", action="store_true",
                        help="Use symbolic gamestates and append _symbolic to output names.")
    args = parser.parse_args()

    dirs   = _DIRS_SYMBOLIC if args.symbolic else _DIRS_DEFAULT
    suffix = "_symbolic" if args.symbolic else ""

    df = plot_win_and_turns_by_difficulty(dirs, suffix=suffix)
    print(df.to_string(index=False))
