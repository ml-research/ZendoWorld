import json
from pathlib import Path
from collections import defaultdict, Counter
import re
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_by_difficulty import classify_difficulty, DIFFICULTIES as _DIFF_CLASSES

colors = [
    "#A8E6A3",  # pastel green
    "#7FD1B9",  # minty teal
    "#80BDE3",  # soft sky blue – good accent
    "#A3A8E6",  # lavender blue
    "#D4A3E6",  # pastel lilac
    "#F4A3C0",  # rosy pink
    "#F4A3A3",  # light coral
    "#F4CBA3",  # peach
    "#E9EB8C",  # light yellow
]

def _load_player_design() -> dict:
    path = Path(__file__).parent.parent / "player_design.json"
    with open(path) as f:
        return json.load(f)

_DESIGN = _load_player_design()

def _player_color(name: str, fallback_idx: int = 0) -> str:
    return _DESIGN.get(name, {}).get("color", colors[fallback_idx % len(colors)])

def _player_hatch(name: str) -> str | None:
    return _DESIGN.get(name, {}).get("hatch", None)

def neurips_fig():
    """NeurIPS-ready figsize."""
    inches_per_pt = 1/72.27
    golden = (1 + 5**0.5)/2
    width_pt = 246  # \columnwidth
    return (width_pt * inches_per_pt, width_pt * inches_per_pt / golden)

plt.rcParams.update({
    'figure.figsize': neurips_fig(),
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

GRAY_FILLS = [0.2, 0.45, 0.7, 0.85, 0.35, 0.55]
EDGE_COLOR = "black"
EDGE_LINEWIDTH = 0.6

def _choose_fill(i, mode="grayscale"):
    if mode == "grayscale":
        return str(GRAY_FILLS[i % len(GRAY_FILLS)])
    else:
        color_cycle = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
        return color_cycle[i % len(color_cycle)]

_COL_W, _COL_H = neurips_fig()
GUTTER_IN = 0.375
FULL_WIDTH_IN = _COL_W * 2 + GUTTER_IN
def get_figsize(columns=1, height_ratio=0.6):
    w = _COL_W if columns == 1 else FULL_WIDTH_IN
    h = w * height_ratio
    return (w, h)


def grouped_bar_publication_no_hatch(ax, df, x, y, hue,
                                     columns=1,
                                     mode="grayscale",
                                     group_width=0.75,
                                     show_n_labels=True,
                                     compact_n=True,
                                     label_fontsize=6,
                                     label_y_offset_rel=0.02,
                                     return_proxy_legend=True,
                                     label_col="n_files_won",
                                     std_col=None):
    diffs = list(df[x].cat.categories) if hasattr(df[x], "cat") else sorted(df[x].unique())
    players = list(df[hue].unique())
    n_groups = len(diffs)
    n_bars = len(players)
    bar_w = group_width / max(n_bars, 1)
    base_x = np.arange(n_groups)

    proxy_handles = []
    proxy_labels = []

    ax.set_autoscale_on(True)

    for i, p in enumerate(players):
        sub = df[df[hue] == p]
        ys = []
        ns = []
        for d in diffs:
            row = sub[sub[x] == d]
            if row.empty:
                ys.append(0.0)
                ns.append(0)
            else:
                ys.append(float(row[y].values[0]))
                try:
                    ns.append(int(row[label_col].values[0]))
                except Exception:
                    ns.append(0)

        xs = base_x + (i - (n_bars - 1) / 2) * bar_w
        face = _player_color(p, i)
        hatch = _player_hatch(p)

        yerr = None
        if std_col is not None and std_col in df.columns:
            yerr = []
            for d in diffs:
                row = sub[sub[x] == d]
                if row.empty:
                    yerr.append(0.0)
                else:
                    val = row[std_col].values[0]
                    yerr.append(0.0 if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val))

        rects = ax.bar(xs, ys,
                       width=bar_w,
                       label=p,
                       color=face,
                       edgecolor=EDGE_COLOR,
                       linewidth=EDGE_LINEWIDTH,
                       hatch=hatch,
                       yerr=yerr,
                       error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "black"},
                       zorder=3)

        proxy = mpatches.Patch(facecolor=face, edgecolor=EDGE_COLOR, linewidth=EDGE_LINEWIDTH, hatch=hatch, label=p)
        proxy_handles.append(proxy)
        proxy_labels.append(p)

        if show_n_labels:
            try:
                ylo, yhi = ax.get_ylim()
            except Exception:
                ylo, yhi = 0.0, 1.0
            offset = (yhi - ylo) * label_y_offset_rel if (yhi - ylo) > 0 else 0.01

            for rect, nval in zip(rects, ns):
                h = rect.get_height()
                txt = f"{nval}" if compact_n else f"n={nval}"

                if h <= 0:
                    y_text = ylo + offset
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            y_text,
                            txt,
                            ha="center",
                            va="bottom",
                            fontsize=label_fontsize,
                            rotation=0,
                            clip_on=False)
                else:
                    y_text = h + offset
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            y_text,
                            txt,
                            ha="center",
                            va="bottom",
                            fontsize=label_fontsize,
                            rotation=0,
                            clip_on=False)

    ax.set_xticks(base_x)
    ax.set_xticklabels([str(d).capitalize() for d in diffs], fontsize=7)
    ax.margins(y=0.12)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, zorder=0)

    if return_proxy_legend:
        return (proxy_handles, proxy_labels)
    return None


def save_fig_with_top_legend(fig, path_without_ext, legend_handles_labels=None, ncol=None, dpi=300):
    pdf_path = path_without_ext + ".pdf"
    png_path = path_without_ext + ".png"

    if legend_handles_labels:
        handles, labels = legend_handles_labels
        if ncol is None:
            ncol = max(1, len(labels))
        legend = fig.legend(handles, labels, loc="upper center",
                            bbox_to_anchor=(0.5, 1.02),
                            ncol=ncol, frameon=False, fontsize=9)
        fig.canvas.draw()
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", bbox_extra_artists=(legend,))
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", bbox_extra_artists=(legend,))
        legend.remove()
    else:
        fig.tight_layout()
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    return pdf_path, png_path

def save_pubfig(fig, path, transparent=False):
    """
    Save figure as vector PDF and as high-res PNG fallback.
    Embeds fonts (pdf.fonttype=42 set above).
    """
    fig.savefig(path + ".pdf", bbox_inches="tight", dpi=300)
    fig.savefig(path + ".png", bbox_inches="tight", dpi=300)
    print("Saved", path + ".pdf and .png")

def evaluate_averaged_players(dirs: list[tuple[str, Path]], task_ids: set | None = None):
    """
    Updated to handle per-task averaging across repetition subdirectories.
    Each player directory may contain either:
      - *.json files directly (no repetitions), OR
      - subdirectories, each containing the task *.json files for that repetition.

    For each player, we average metrics per task filename across repetitions, then compute summaries.
    """

    # ----------------------------
    # Helpers
    # ----------------------------
    def evaluate_single_file(filepath: Path):
        with open(filepath, "r") as f:
            data = json.load(f)

        # Raw fields (handle missing safely)
        won = bool(data.get("won", False))
        if won:
            print("Won game:", filepath)
        game_over_reason = str(data.get("game_over_reason", ""))
        guesses = data.get("guesses", {}).get("0", [])
        correct_program = data.get("correct_program") or data.get("correct_Program")
        difficulty = classify_difficulty(correct_program)
        turns = data.get("turns", 0)
        examples = data.get("examples", 0)

        # Mean per-turn duration
        turn_durations = data.get("turn_durations", {}) or {}
        mean_duration = (sum(turn_durations.values()) / len(turn_durations)) if turn_durations else 0.0

        # Count "quiz mode" labels in turn_descriptions
        turn_descriptions = data.get("turn_descriptions", {}) or {}
        labels_correct = 0
        labels_incorrect = 0
        guessed_true = 0
        for desc in turn_descriptions:
            d = str(desc).lower()
            if "quiz mode correct" in d:
                labels_correct += 1
            elif "quiz mode incorrect" in d:
                labels_incorrect += 1
            if "guessed true" in d:
                guessed_true += 1

        no_counter_found = "no counter example found" in game_over_reason.lower()
        if no_counter_found:
            print("No counter example found.", filepath)

        return {
            "won": 1.0 if won else 0.0,                 # as numeric so we can average into a rate
            "game_over_reason": game_over_reason,
            "no_counter_found": 1.0 if no_counter_found else 0.0,
            "num_guesses": len(guesses),
            "difficulty": difficulty,
            "turns": float(turns),
            "examples": float(examples),
            "mean_turn_duration": float(mean_duration),
            "labels_correct": float(labels_correct),
            "labels_incorrect": float(labels_incorrect),
            "guessed_true": float(guessed_true),
        }

    def average_records(records):
        """Average a list of per-run dictionaries for a single task filename."""
        if not records:
            return None

        # Difficulty may (rarely) vary; pick majority.
        diff = Counter(r["difficulty"] for r in records if r.get("difficulty")).most_common(1)
        difficulty = diff[0][0] if diff else "unknown"

        # Average fields that always use all records
        keys_avg_all = [
            "won", "no_counter_found",
            "mean_turn_duration", "labels_correct", "labels_incorrect", "guessed_true"
        ]
        out = {k: 0.0 for k in keys_avg_all}
        for r in records:
            for k in keys_avg_all:
                out[k] += float(r.get(k, 0.0))
        n = float(len(records))
        for k in keys_avg_all:
            out[k] = out[k] / n

        won_records = [r for r in records if float(r.get("won", 0.0)) > 0.0]
        if won_records:
            out["num_guesses"] = sum(float(r.get("num_guesses", 0.0)) for r in won_records) / len(won_records)
            out["turns"]       = sum(float(r.get("turns", 0.0))        for r in won_records) / len(won_records)
            out["examples"]    = sum(float(r.get("examples", 0.0))     for r in won_records) / len(won_records)
        else:
            out["num_guesses"] = None
            out["turns"]       = None
            out["examples"]    = None

        out["difficulty"] = difficulty
        out["n_files"] = len(records)
        return out

    def load_player_averaged(player_dir: Path):
        """
        Return (averaged_dict, per_seed_list, has_multiple_seeds).
        averaged_dict: { filename -> averaged_metrics_dict }
        per_seed_list: list of { filename -> metrics_dict }, one entry per seed subdirectory
        has_multiple_seeds: True if there are >1 seed subdirectories with json files
        Handles either direct JSON files or repetition subdirectories.
        """
        averaged = {}
        per_seed = []

        subdirs = sorted([p for p in player_dir.iterdir() if p.is_dir()])
        has_reps = False
        for sd in subdirs:
            if any(sd.glob("*.json")):
                has_reps = True
                break

        if not has_reps:
            per_file = defaultdict(list)
            for f in player_dir.glob("task_*_state.json"):
                per_file[f.name].append(evaluate_single_file(f))
            for fname, recs in per_file.items():
                averaged[fname] = average_records(recs)
            return averaged, [], False

        for sd in subdirs:
            seed_dict = {}
            for f in sd.glob("task_*_state.json"):
                try:
                    seed_dict[f.name] = evaluate_single_file(f)
                except Exception as e:
                    print(e)
                    continue
            if seed_dict:
                per_seed.append(seed_dict)

        per_file = defaultdict(list)
        for seed_dict in per_seed:
            for fname, data in seed_dict.items():
                per_file[fname].append(data)

        for fname, recs in per_file.items():
            avg = average_records(recs)
            if avg is not None:
                averaged[fname] = avg

        return averaged, per_seed, len(per_seed) > 1

    players = []
    per_seed_data = {}  # name -> (per_seed_list, has_multiple_seeds)
    for name, d in dirs:
        averaged, per_seed, has_multiple = load_player_averaged(d)
        players.append(averaged)
        per_seed_data[name] = (per_seed, has_multiple)
    shared_files = players[0].keys()
    for p in players:
        shared_files &= p.keys()

    if task_ids is not None:
        _tid_re = re.compile(r"task_(\d+)_")
        def _has_task_id(fname):
            m = _tid_re.search(fname)
            return m is not None and int(m.group(1)) in task_ids
        shared_files = {f for f in shared_files if _has_task_id(f)}

    DIFFICULTIES = _DIFF_CLASSES + ["all"]

    def init_stats():
        d = {
            "wins": 0.0,
            "guess_counts": [],
            "turn_counts": [],
            "example_counts": [],
            "turn_durations": [],
            "no_counter_found": 0.0,
            "n_files_won": 0,
        }
        for diff in _DIFF_CLASSES:
            d[f"total_{diff}"] = 0
            d[f"{diff}_label_correct"] = 0.0
            d[f"{diff}_label_incorrect"] = 0.0
            d[f"{diff}_guessed_true"] = 0.0
        return d
    results = {}
    for name, _ in dirs:
        results[name] = defaultdict(init_stats)

    def update_for_player(player_name, data):
        diff = data["difficulty"]
        stats = results[player_name][diff]
        stats_all = results[player_name]["all"]

        # Track denominators + label tallies
        if diff in _DIFF_CLASSES:
            stats[f"total_{diff}"] += 1
            stats[f"{diff}_label_correct"] += data["labels_correct"]
            stats[f"{diff}_label_incorrect"] += data["labels_incorrect"]
            stats[f"{diff}_guessed_true"] += data["guessed_true"]

        # Rates always accumulate
        stats["no_counter_found"] += data["no_counter_found"]
        stats_all["no_counter_found"] += data["no_counter_found"]
        stats["wins"] += data["won"]
        stats_all["wins"] += data["won"]

        # Append ONLY if present (i.e., there was at least one win for that task)
        if data["num_guesses"] is not None:
            stats["guess_counts"].append(data["num_guesses"])
            stats_all["guess_counts"].append(data["num_guesses"])
        if data["turns"] is not None:
            stats["turn_counts"].append(data["turns"])
            stats_all["turn_counts"].append(data["turns"])
        if data["examples"] is not None:
            stats["example_counts"].append(data["examples"])
            stats_all["example_counts"].append(data["examples"])

        # Turn time stays averaged over all runs as before
        stats["turn_durations"].append(data["mean_turn_duration"])
        stats_all["turn_durations"].append(data["mean_turn_duration"])
        nf = int(data.get("n_files", 0))
        if float(data.get("won", 0.0)) > 0.0:
            stats["n_files_won"] += nf
            stats_all["n_files_won"] += nf


    # ----------------------------
    # Aggregate over shared tasks
    # ----------------------------
    for fname in sorted(shared_files):
        for (name, path), player in zip(dirs, players):
            print(f"Processing {name} / {fname}")
            d = player[fname]
            update_for_player(name, d)

    # ----------------------------
    # Summary
    # ----------------------------
    def safe_avg(lst):
        return (sum(lst) / len(lst)) if lst else 0.0

    def summarize(player, stats_by_difficulty):
        print(f"\n===== {player} =====")
        for diff in DIFFICULTIES:
            stats = stats_by_difficulty[diff]
            if not stats["guess_counts"]:
                continue

            avg_guesses = safe_avg(stats["guess_counts"])
            avg_turns = safe_avg(stats["turn_counts"])
            avg_examples = safe_avg(stats["example_counts"])
            avg_duration = safe_avg(stats["turn_durations"])

            # Number of files that contributed
            n_files = len(stats["guess_counts"])
            if diff == "all":
                denom = float(len(shared_files))
                print(f"\nDifficulty: {diff.capitalize()} (n={int(denom)} files)")
            else:
                denom = stats.get(f"total_{diff}", 0)
                print(f"\nDifficulty: {diff.capitalize()} (n={denom} files)")

            print(f"  Wins (sum of win rates): {stats['wins']:.2f} / {denom}")
            print(f"  Avg. Guesses:     {avg_guesses:.2f} (from {n_files} winning runs)")
            print(f"  Avg. Turns:       {avg_turns:.2f}")
            print(f"  Avg. Examples:    {avg_examples:.2f}")
            print(f"  Avg. Turn Time:   {avg_duration:.2f} sec")
            print(f"  No Counter Found (sum of rates): {stats['no_counter_found']:.2f} / {stats['wins']:.2f}")

    for name, _ in dirs:
        summarize(name, results[name])

    # ----------------------------
    # Std across seeds
    # ----------------------------
    def _seed_difficulty_metrics(seed_files, task_files):
        """Aggregate per-difficulty metrics for one seed, returning rates."""
        by_diff = {}
        for fname in task_files:
            if fname not in seed_files:
                continue
            d = seed_files[fname]
            diff = d["difficulty"]
            for key in (diff, "all"):
                if key not in by_diff:
                    by_diff[key] = {
                        "wins": 0.0, "total": 0,
                        "turns": [], "labels_correct": 0.0,
                        "labels_incorrect": 0.0, "guessed_true": 0.0,
                        "turn_durations": [],
                    }
                s = by_diff[key]
                s["wins"] += d["won"]
                s["total"] += 1
                if d["won"] > 0.0 and d["turns"] is not None:
                    s["turns"].append(d["turns"])
                s["labels_correct"] += d["labels_correct"]
                s["labels_incorrect"] += d["labels_incorrect"]
                s["guessed_true"] += d["guessed_true"]
                s["turn_durations"].append(d["mean_turn_duration"])

        result = {}
        for diff, s in by_diff.items():
            n = s["total"]
            lc, li = s["labels_correct"], s["labels_incorrect"]
            lt = lc + li
            result[diff] = {
                "win_rate": s["wins"] / n if n > 0 else 0.0,
                "avg_turns": (sum(s["turns"]) / len(s["turns"])) if s["turns"] else 0.0,
                "label_acc": (lc / lt) if lt > 0 else 0.0,
                "label_true": (s["guessed_true"] / lt) if lt > 0 else 0.0,
                "avg_turn_time": (sum(s["turn_durations"]) / len(s["turn_durations"])) if s["turn_durations"] else 0.0,
            }
        return result

    std_results = {}
    for name, (per_seed, has_multiple) in per_seed_data.items():
        if not has_multiple or len(per_seed) < 2:
            std_results[name] = None
            continue
        seed_metrics_list = [_seed_difficulty_metrics(seed, shared_files) for seed in per_seed]
        all_diffs = set()
        for sm in seed_metrics_list:
            all_diffs.update(sm.keys())
        metric_keys = ["win_rate", "avg_turns", "label_acc", "label_true", "avg_turn_time"]
        std_by_diff = {}
        for diff in all_diffs:
            values = [sm.get(diff, {}) for sm in seed_metrics_list]
            n = len(values)
            std_by_diff[diff] = {
                k: float(np.std([v.get(k, 0.0) for v in values], ddof=1)) / np.sqrt(n) if n > 1 else 0.0
                for k in metric_keys
            }
        std_results[name] = std_by_diff

    return results, std_results, len(shared_files)

DIFFICULTIES = _DIFF_CLASSES + ["all"]

def _zero_std_cols():
    return {"wins_std": 0.0, "avg_turns_std": 0.0, "label_acc_std": 0.0,
            "label_true_std": 0.0, "avg_turn_time_std": 0.0}

def _std_cols(std_results, player, diff, wins_den):
    player_std = std_results.get(player) if std_results else None
    if player_std is None:
        # No seed data at all — use None so _merge_cis knows to fall back.
        return {k: None for k in ["wins_std", "avg_turns_std", "label_acc_std",
                                   "label_true_std", "avg_turn_time_std"]}
    sd = player_std.get(diff, {})
    return {
        "wins_std": sd.get("win_rate", 0.0),
        "avg_turns_std": sd.get("avg_turns", 0.0),
        "label_acc_std": sd.get("label_acc", 0.0),
        "label_true_std": sd.get("label_true", 0.0),
        "avg_turn_time_std": sd.get("avg_turn_time", 0.0),
    }

def _merge_cis(seed_ci, task_ci):
    """Return seed_ci values when seed data exists (even if CI=0), else fall back to task_ci.

    seed_ci values are None when there is no seed data at all; 0.0 means seed
    data exists but cross-seed variance was zero.  Only the None case should
    fall back to the task-level CI (Agresti-Coull etc.).
    """
    return {k: task_ci.get(k, 0.0) if v is None else v for k, v in seed_ci.items()}


def _sem_prop(p, n):
    """Standard error for a proportion using Agresti-Coull-style smoothing.

    Unlike the plain Wald SE sqrt(p(1-p)/n) (which is 0 when p=0 or p=1),
    this adds a small pseudo-count so edge cases still show a non-zero SE —
    e.g. an agent that wins every game still has some uncertainty. Used as
    a fallback only when no cross-seed data is available.
    """
    if n < 1 or p != p:
        return 0.0
    z = 1.96
    z2 = z * z
    n_tilde = n + z2                          # effective sample size
    p_tilde = (p * n + z2 / 2) / n_tilde     # shrunk proportion
    return float(np.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde))


def _sem_mean(vals):
    """Standard error of the mean for a list of values (n-1 in the denominator)."""
    n = len(vals)
    if n < 2:
        return 0.0
    return float(np.std(vals, ddof=1)) / np.sqrt(n)


def results_to_dataframe(results, std_results=None, shared_total=None):
    rows = []
    for player, by_diff in results.items():
        all_c = sum(by_diff.get(d, {}).get(f"{d}_label_correct",   0.0) for d in _DIFF_CLASSES)
        all_i = sum(by_diff.get(d, {}).get(f"{d}_label_incorrect", 0.0) for d in _DIFF_CLASSES)
        all_g = sum(by_diff.get(d, {}).get(f"{d}_guessed_true",    0.0) for d in _DIFF_CLASSES)

        for diff in DIFFICULTIES:
            s = by_diff.get(diff, {})

            if diff in _DIFF_CLASSES:
                wins_den = s.get(f"total_{diff}", 0)
                lc = s.get(f"{diff}_label_correct",   0.0)
                li = s.get(f"{diff}_label_incorrect",  0.0)
                gt = s.get(f"{diff}_guessed_true",     0.0)
            else:  # "all"
                wins_den = shared_total if shared_total is not None else 0
                lc, li, gt = all_c, all_i, all_g

            wins = float(s.get("wins", 0.0))
            win_rate = (wins / wins_den) if wins_den else 0.0

            lab_tot = lc + li
            lab_acc = (lc / lab_tot) if lab_tot > 0 else 0.0
            lab_true = (gt / lab_tot) if lab_tot > 0 else 0.0

            def safe_avg(a): return (sum(a) / len(a)) if a else 0.0
            def safe_avg_wins(a): return (sum(a) / len(a)) if a else float("nan")

            guess_list   = s.get("guess_counts", [])
            turn_list    = s.get("turn_counts", [])
            example_list = s.get("example_counts", [])
            dur_list     = s.get("turn_durations", [])

            avg_guesses      = safe_avg_wins(guess_list)
            avg_turns        = safe_avg_wins(turn_list)
            avg_examples     = safe_avg_wins(example_list)
            avg_turn_time    = safe_avg(dur_list)
            median_turn_time = float(np.median(dur_list)) if dur_list else 0.0

            ncf = float(s.get("no_counter_found", 0.0))
            ncf_den = wins if wins > 0 else 0.0

            rows.append({
                "player": player,
                "difficulty": diff,
                # plotted values
                "wins": wins,
                "label_acc": lab_acc,
                "label_true": lab_true,
                "avg_guesses": avg_guesses,
                "avg_turns": avg_turns,
                "avg_examples": avg_examples,
                "avg_turn_time": avg_turn_time,
                "median_turn_time": median_turn_time,
                "no_counter_found": ncf,
                "wins_den": wins_den,
                "label_n": lab_tot,
                "n_guess": len(guess_list),
                "n_turns": len(turn_list),
                "n_examples": len(example_list),
                "n_turn_time": len(dur_list),
                "no_counter_den": ncf_den,
                "n_files_won": int(s.get("n_files_won", 0)),
                # 95% CIs: prefer cross-seed CI when available, else across-task CI.
                # Win rate: only show CI when multiple seeds exist. If cross-seed
                # variance happens to be 0 (e.g. Oracle always wins 21/22), fall back
                # to Agresti-Coull on the proportion so the CI is still shown.
                **_merge_cis(
                    _std_cols(std_results, player, diff, wins_den),
                    {
                        "wins_std":          _sem_prop(win_rate, wins_den) if std_results else 0.0,
                        "avg_turns_std":     _sem_mean(turn_list),
                        "label_acc_std":     _sem_prop(lab_acc,  lab_tot),
                        "label_true_std":    _sem_prop(lab_true, lab_tot),
                        "avg_turn_time_std": _sem_mean(dur_list),
                    }
                ),
            })

    df = pd.DataFrame(rows)
    df["difficulty"] = pd.Categorical(df["difficulty"], categories=DIFFICULTIES, ordered=True)
    return df.sort_values(["player", "difficulty"]).reset_index(drop=True)


def plot_turn_time_by_difficulty(results, std_results=None, shared_total=None, save_dir=None):
    """
    Single-panel publication figure: average time per turn (seconds),
    grouped by player, x-axis = difficulty (easy / medium / difficult / all).
    Style matches the existing summary_stacked figure.
    """
    df = results_to_dataframe(results, std_results=std_results, shared_total=shared_total)
    has_std = std_results is not None and any(v is not None for v in std_results.values())

    fig_w, fig_h = get_figsize(columns=2, height_ratio=0.9)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    legend_info = grouped_bar_publication_no_hatch(
        ax, df, x="difficulty", y="median_turn_time", hue="player",
        mode="grayscale", group_width=0.95,
        show_n_labels=False, label_col="n_turn_time",
        return_proxy_legend=True,
        std_col="avg_turn_time_std",
    )

    ax.set_title("Median time per turn by difficulty", fontsize=9, pad=5)
    ax.set_ylabel("Median time per turn (s)", fontsize=8, labelpad=2)
    ax.yaxis.labelpad = 6

    proxy_handles, proxy_labels = legend_info if legend_info else ([], [])
    if not proxy_handles:
        df_all_row = df[df["difficulty"] == "all"].reset_index(drop=True)
        proxy_labels = list(df_all_row["player"].unique())
        for i, p in enumerate(proxy_labels):
            face = _player_color(p, i)
            proxy_handles.append(
                mpatches.Patch(facecolor=face, edgecolor=EDGE_COLOR,
                               linewidth=EDGE_LINEWIDTH, label=p)
            )

    legend_fontsize = 7
    legend = fig.legend(
        proxy_handles, proxy_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        frameon=True,
        facecolor="whitesmoke",
        fontsize=legend_fontsize,
        handlelength=1.2,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=[0.06, 0.12, 0.98, 0.94])

    if save_dir is not None:
        from pathlib import Path as _Path
        _Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_base = str(_Path(save_dir) / f"turn_time_by_difficulty{_SUFFIX}")
        fig.canvas.draw()
        fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))
        legend.remove()
        plt.close(fig)
        print("Saved:", out_base + ".pdf", out_base + ".png")
    else:
        plt.show()

    return df

import sys as _sys

_SYMBOLIC = "--symbolic" in _sys.argv
_SUFFIX   = "_symbolic" if _SYMBOLIC else ""

_DIRS_DEFAULT = [
    ("Oracle Agent",    Path("gamestates/gamestates_zendo")),
    ("VLM Agent",       Path("gamestates/gamestates_fullgpt")),
    ("Bayesian Agent", Path("gamestates/gamestates_scientist")),
    ("VLP Agent",       Path("gamestates/gamestates_vlp")),
]

# Human study data is evaluated separately (different task set, player IDs instead of seeds)
HUMAN_STUDY_DIR = Path("gamestates/gamestates_study")

_DIRS_SYMBOLIC = [
    ("zendo_symbolic",        Path("gamestates/gamestates_zendo_symbolic")),
    ("random_zendo_symbolic", Path("gamestates/gamestates_random_zendo_symbolic")),
    ("fullgpt_symbolic",      Path("gamestates/gamestates_fullgpt_symbolic")),
    ("scientist_symbolic",    Path("gamestates/gamestates_scientist_symbolic")),
    ("vlp_symbolic",          Path("gamestates/gamestates_vlp_symbolic")),
]

_dirs = _DIRS_SYMBOLIC if _SYMBOLIC else _DIRS_DEFAULT

results, std_results, shared_total = evaluate_averaged_players(_dirs)

has_std = any(v is not None for v in std_results.values())
# Make and save plots into ./figs
df = results_to_dataframe(results, std_results=std_results, shared_total=shared_total)
fig_w, fig_h = get_figsize(columns=2, height_ratio=0.9)
fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))

ax_win   = axes[0, 0]
ax_turns = axes[0, 1]
ax_acc   = axes[1, 0]
ax_bias  = axes[1, 1]
df["win_rate"] = df["wins"] / df["wins_den"]
grouped_bar_publication_no_hatch(
    ax_win, df, x="difficulty", y="win_rate", hue="player",
    mode="grayscale", group_width=0.85,
    show_n_labels=False,
    std_col="wins_std" if has_std else None,
)

ax_win.set_title("Win rate by difficulty", fontsize=9, pad=5)
ax_win.set_ylabel("Win rate", fontsize=8)
# Top: avg_turns per difficulty (grouped by player)
legend_info = grouped_bar_publication_no_hatch(
    ax_turns, df, x="difficulty", y="avg_turns", hue="player",
    mode="grayscale", group_width=0.85,
    show_n_labels=False,
    std_col="avg_turns_std",
)

ax_turns.set_title("Average turns per difficulty", fontsize=9, pad=5)
ax_turns.set_ylabel("Average turns", fontsize=8)

grouped_bar_publication_no_hatch(
    ax_acc, df, x="difficulty", y="label_acc", hue="player",
    mode="grayscale", group_width=0.85,
    show_n_labels=False,
    std_col="label_acc_std",
)

ax_acc.set_title("Label accuracy", fontsize=9, pad=5)
ax_acc.set_ylabel("Accuracy", fontsize=8)

grouped_bar_publication_no_hatch(
    ax_bias, df, x="difficulty", y="label_true", hue="player",
    mode="grayscale", group_width=0.85,
    show_n_labels=False,
    std_col="label_true_std",
)

ax_bias.set_title("Label bias (fraction guessed true)", fontsize=9, pad=5)
ax_bias.set_ylabel("Bias", fontsize=8)

if legend_info is None:
    proxy_handles = []
    proxy_labels = list(df["player"].unique())
    for i, p in enumerate(proxy_labels):
        face = _player_color(p, i)
        proxy_handles.append(mpatches.Patch(facecolor=face, edgecolor=EDGE_COLOR,
                                            linewidth=EDGE_LINEWIDTH, label=p))
else:
    proxy_handles, proxy_labels = legend_info

legend = fig.legend(
    proxy_handles, proxy_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),
    ncol=2,
    frameon=True,
    facecolor="whitesmoke",
    fontsize=7,
    handlelength=1.2,
    handletextpad=0.6,
)


fig.tight_layout(rect=[0.06, 0.08, 0.98, 0.95])

# Make sure output dir exists
out_dir = Path("figs")
out_dir.mkdir(parents=True, exist_ok=True)
out_base = out_dir / f"player_eval{_SUFFIX}"

# Force draw and save including the legend in bbox_extra_artists to avoid cropping
fig.canvas.draw()
fig.savefig(str(out_base) + ".pdf", dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))
fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))

# remove legend afterwards so subsequent saves don't duplicate
legend.remove()
plt.close(fig)
print("Saved:", str(out_base) + ".pdf", str(out_base) + ".png")

plot_turn_time_by_difficulty(results, std_results=std_results, shared_total=shared_total, save_dir="figs")
