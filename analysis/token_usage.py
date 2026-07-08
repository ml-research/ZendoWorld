"""
token_usage.py

Aggregates OpenAI token usage per player based on exact Berlin-time windows.

Players and their Berlin-time windows:
  VLM:       9.3–11.3,  26.3–27.3,  1.4 18:00–2.4 05:49
  VLP:       28.3–1.4 15:00
  Bayesian: 26.2–28.2, 2.4 12:00–4.4

OpenAI data is at UTC-day granularity (each row = one UTC calendar day).
Berlin is CET (UTC+1) in winter / CEST (UTC+2) from 2026-03-29.

For days where two players ran at non-overlapping hours (no true conflict),
tokens are split proportionally by each player's coverage of that UTC day.
E.g. if VLP covers 13h and VLM covers 8h of the same UTC day, VLP receives
13/21 of that day's tokens and VLM receives 8/21.  Proportionally-split days
are marked with ~ in the per-player tables.

Run from the repo root:
    python analysis/token_usage.py
"""

import glob
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USER_ID = "user-Hfp3UYTfQMr3zqhdkRZDib9s"
MODEL   = "gpt-5-mini-2025-08-07"

BERLIN = ZoneInfo("Europe/Berlin")
UTC    = ZoneInfo("UTC")


def b(year, month, day, hour=0, minute=0):
    """Convert Berlin local time to a UTC-aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=BERLIN).astimezone(UTC)


# Each tuple: [start_utc_inclusive, end_utc_exclusive)
PLAYER_WINDOWS = {
    "VLM": [
        (b(2026, 4,  21),         b(2026, 4,  22)),
        (b(2026, 4, 22),         b(2026, 4, 23)),
    ],
    "VLP": [
        (b(2026, 4, 21),         b(2026, 4,  22)),
    ],
    "Bayesian": [
        (b(2026, 4, 24),         b(2026, 4,  25)),
        (b(2026, 4, 23),         b(2026, 4,  24)),
    ],
}

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

csv_files = sorted(glob.glob("completions_usage_2026-04-15*.csv"))
if not csv_files:
    raise FileNotFoundError("No completions_usage_*.csv files found in the current directory.")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

mask = (df["user_id"] == USER_ID) & (df["model"] == MODEL)
df = df[mask].copy()

df["ts_start"] = pd.to_datetime(df["start_time_iso"], utc=True)
df["ts_end"]   = pd.to_datetime(df["end_time_iso"],   utc=True)
df["date"]     = df["ts_start"].dt.date

# ---------------------------------------------------------------------------
# For each row compute each player's coverage in seconds
# ---------------------------------------------------------------------------

def player_coverage(row):
    """Returns {player: overlap_seconds} for all matching players."""
    row_s = row["ts_start"].to_pydatetime()
    row_e = row["ts_end"].to_pydatetime()
    result = {}
    for player, windows in PLAYER_WINDOWS.items():
        for win_s, win_e in windows:
            if row_s < win_e and row_e > win_s:
                secs = (min(row_e, win_e) - max(row_s, win_s)).total_seconds()
                result[player] = secs
                break
    return result

df["coverage"]  = df.apply(player_coverage, axis=1)
df["players"]   = df["coverage"].apply(lambda c: list(c.keys()))
df["n_players"] = df["players"].apply(len)

shared_dates = set(df.loc[df["n_players"] > 1, "date"])

# ---------------------------------------------------------------------------
# Sanity-check: confirm no player windows overlap each other
# ---------------------------------------------------------------------------

conflicts = []
pnames = list(PLAYER_WINDOWS.keys())
for i, pa in enumerate(pnames):
    for pb in pnames[i+1:]:
        for ws_a, we_a in PLAYER_WINDOWS[pa]:
            for ws_b, we_b in PLAYER_WINDOWS[pb]:
                if ws_a < we_b and we_a > ws_b:
                    conflicts.append((pa, pb, max(ws_a, ws_b), min(we_a, we_b)))

# ---------------------------------------------------------------------------
# Build per-player daily data with proportional splitting for shared days
# ---------------------------------------------------------------------------
# For each row, each matching player gets fraction = own_coverage / total_coverage
# of that row's tokens.  Non-shared rows: fraction = 1.0.

METRICS = ["input_tokens", "output_tokens", "num_model_requests"]

player_rows = {p: [] for p in PLAYER_WINDOWS}

for _, row in df.iterrows():
    cov = row["coverage"]
    if not cov:
        continue
    total_cov = sum(cov.values())
    for player, secs in cov.items():
        frac = secs / total_cov
        entry = {"date": row["date"], "shared": len(cov) > 1}
        for m in METRICS:
            entry[m] = row[m] * frac
        player_rows[player].append(entry)

player_dfs = {}
for player, rows in player_rows.items():
    if not rows:
        player_dfs[player] = pd.DataFrame(columns=["date"] + METRICS)
        continue
    pdf = pd.DataFrame(rows)
    daily = (
        pdf.groupby("date", as_index=False)
        .agg(
            input_tokens       = ("input_tokens",       "sum"),
            output_tokens      = ("output_tokens",      "sum"),
            num_model_requests = ("num_model_requests", "sum"),
            shared             = ("shared",              "any"),
        )
        .sort_values("date")
    )
    daily["total_tokens"] = daily["input_tokens"] + daily["output_tokens"]
    player_dfs[player] = daily

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------

print(f"User:  {USER_ID}")
print(f"Model: {MODEL}")
print(f"Files: {', '.join(csv_files)}")
print()

print("Player windows (Berlin → UTC):")
for player, windows in PLAYER_WINDOWS.items():
    for win_s, win_e in windows:
        print(f"  {player:<12} {win_s.strftime('%Y-%m-%d %H:%M')} UTC  →  {win_e.strftime('%Y-%m-%d %H:%M')} UTC")
print()

if conflicts:
    print("WARNING: player windows overlap in time — results may be inaccurate:")
    for pa, pb, ov_s, ov_e in conflicts:
        print(f"  {pa} ∩ {pb}: {ov_s.strftime('%Y-%m-%d %H:%M')} – {ov_e.strftime('%Y-%m-%d %H:%M')} UTC")
    print()
else:
    print("Player windows do not overlap — no double-counting.")
    print()

if shared_dates:
    print("Shared UTC days (two players ran at non-overlapping hours → proportional split):")
    for d in sorted(shared_dates):
        row0 = df[df["date"] == d].iloc[0]
        cov  = row0["coverage"]
        total = sum(cov.values())
        parts = [f"{p} {cov[p]/3600:.1f}h ({100*cov[p]/total:.0f}%)" for p in cov]
        print(f"  {d}  →  {',  '.join(parts)}")
    print()

# ---------------------------------------------------------------------------
# Per-player daily breakdown + subtotal
# ---------------------------------------------------------------------------

HEADER = f"{'Date':<12}  {'Requests':>10}  {'Input':>12}  {'Output':>12}  {'Total':>12}"
SEP    = "-" * len(HEADER)

all_totals = {}

for player in ["VLM", "VLP", "Bayesian"]:
    daily = player_dfs[player]

    print(f"=== {player} ===")
    print(HEADER)
    print(SEP)

    for _, row in daily.iterrows():
        flag = " ~" if row["shared"] else ""
        print(
            f"{str(row['date']):<12}  "
            f"{int(round(row['num_model_requests'])):>10,}  "
            f"{int(round(row['input_tokens'])):>12,}  "
            f"{int(round(row['output_tokens'])):>12,}  "
            f"{int(round(row['total_tokens'])):>12,}{flag}"
        )

    print(SEP)
    totals = {m: daily[m].sum() for m in METRICS}
    totals["total_tokens"] = daily["total_tokens"].sum()
    print(
        f"{'TOTAL':<12}  "
        f"{int(round(totals['num_model_requests'])):>10,}  "
        f"{int(round(totals['input_tokens'])):>12,}  "
        f"{int(round(totals['output_tokens'])):>12,}  "
        f"{int(round(totals['total_tokens'])):>12,}"
    )
    print()
    all_totals[player] = totals

# ---------------------------------------------------------------------------
# Grand summary — each token counted exactly once (fractions sum to 1 per row)
# ---------------------------------------------------------------------------

HEADER2 = f"{'Player':<14}  {'Requests':>10}  {'Input':>12}  {'Output':>12}  {'Total':>12}"
SEP2    = "-" * len(HEADER2)

print("=== GRAND TOTAL (shared-day tokens split proportionally by hours covered) ===")
print(HEADER2)
print(SEP2)

grand_req = grand_in = grand_out = grand_tot = 0
for player in ["VLM", "VLP", "Bayesian"]:
    t = all_totals[player]
    req = int(round(t["num_model_requests"]))
    inp = int(round(t["input_tokens"]))
    out = int(round(t["output_tokens"]))
    tot = int(round(t["total_tokens"]))
    print(f"{player:<14}  {req:>10,}  {inp:>12,}  {out:>12,}  {tot:>12,}")
    grand_req += req
    grand_in  += inp
    grand_out += out
    grand_tot += tot

print(SEP2)
print(f"{'ALL':<14}  {grand_req:>10,}  {grand_in:>12,}  {grand_out:>12,}  {grand_tot:>12,}")
print()
print("~ = day shared between two players; tokens split by hours each player's window")
print("    covers of that UTC day (proportional estimate, not exact).")
