"""
Walkthrough: run expectancy by count, and the value of ABS challenges.

Run from the repo root:

    .venv/bin/python examples/challenge_analysis.py

It:
  1. loads the data, the runner model and a league-average player,
  2. builds the results matrix (PA outcome probabilities by count),
  3. simulates run expectancy for every base state / outs / count,
  4. looks up one challenge: R1, 1 out, 1-1 pitch called a strike,
  5. lists the calls where a challenge swings the most runs.
"""

import os
import sys
import time

# Let `import source...` work when this file is run directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd

from source.dataloaders import load_pa_table
from source.players import Player
from source.baserunning import EmpiricalTransitions
from source.counts import count_probabilities
from source.challenges import run_expectancy_table, re_matrix, challenge_table, challenge_value

N_INNINGS = 100_000   # per situation; the 95% CI on each RE is roughly +/- 0.007 runs
SEED = 2026

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

# ---------------------------------------------------------------------------
# 1. Data, runner model, players
# ---------------------------------------------------------------------------
pa = load_pa_table('TruMedia2026.feather')
empirical = EmpiricalTransitions.fit(pa, min_obs=20)
league_avg = Player.league_average(pa)

# ---------------------------------------------------------------------------
# 2. Results matrix: P(PA outcome | the PA reached this count)
# ---------------------------------------------------------------------------
# League average = SEC and ACC teams playing each other. Pass
# conferences=None for all of D1.
matrix = count_probabilities(pa)
print("\nPA outcome probabilities by count (SEC/ACC):")
print(matrix.round(3))

# ---------------------------------------------------------------------------
# 3. Run expectancy for every base state, outs and count
# ---------------------------------------------------------------------------
# A real lineup works too: pass a list of 9 Players and `batter_idx=` for
# who is at the plate.
print(f"\nSimulating 336 situations x {N_INNINGS:,} half-innings...")
start = time.time()
re_table = run_expectancy_table(league_avg, empirical, matrix, n_innings=N_INNINGS, seed=SEED)
print(f"done in {time.time() - start:.0f}s")

print("\nRun expectancy to the end of the half-inning (BB / SO = the call ended the PA):")
print(re_matrix(re_table).round(2))

# ---------------------------------------------------------------------------
# 4. One challenge
# ---------------------------------------------------------------------------
# Runner on 1st, 1 out, a 1-1 pitch is called a strike. If it stands the
# count is 1-2; if the offense challenges and wins it is 2-1.
one = challenge_value(re_table, 'R1', outs=1, count='1-1', call='strike')
print("\nR1, 1 out, 1-1 pitch called a strike:")
print(f"  not challenged: count {one['count_if_stands']}, RE {one['re_stands']:.3f}")
print(f"  overturned:     count {one['count_if_overturned']}, RE {one['re_overturned']:.3f}")
print(f"  run value of a successful challenge: {one['challenger_gain']:+.3f} "
      f"(+/- {one['delta_ci']:.3f})")

# If challenges like this one win only some of the time, weight by that.
half = challenge_value(re_table, 'R1', outs=1, count='1-1', call='strike', p_overturn=0.5)
print(f"  ...expected value if it wins half the time: {half['challenger_gain']:+.3f}")

# ---------------------------------------------------------------------------
# 5. Every call
# ---------------------------------------------------------------------------
# 576 rows: 8 base states x 3 outs x 12 counts x called strike / ball.
challenges = challenge_table(re_table)
cols = ['base_state', 'outs', 'count', 'call', 'challenger', 'count_if_stands',
        'count_if_overturned', 're_stands', 're_overturned', 'challenger_gain']

print("\nBiggest swings:")
print(challenges.sort_values('challenger_gain', ascending=False)[cols].head(10)
      .round(3).to_string(index=False))

# A called ball and a called strike at the same count move the PA between
# the same two counts, just in opposite directions. So a challenge is worth
# the same to either side, and one call is enough to show here.
print("\nRun value of a successful challenge by count and outs "
      "(simple average over the 8 base states):")
print(challenges[challenges['call'] == 'strike']
      .pivot_table(index='count', columns='outs', values='challenger_gain').round(3))

# Save it to join onto tracked challenge events (by base_state, outs, count, call):
# challenges.to_csv('challenge_run_values.csv', index=False)
