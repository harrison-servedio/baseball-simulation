"""
Walkthrough of the first version of the simulator.

Run from the repo root:

    .venv/bin/python examples/basic_simulation.py

It:
  1. loads the TruMedia data (and caches a PA table on first run),
  2. builds the two runner models,
  3. makes the three kinds of players (league average, real, custom),
  4. computes each player's NRGG (paper Section 3) under both runner models,
  5. simulates a real 9-man lineup,
  6. compares against the runs actually scored in the data.
"""

import os
import sys

# Let `import source...` work when this file is run directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from source.dataloaders import load_pa_table, observed_runs_per_nine
from source.outcomes import league_subset
from source.players import Player
from source.baserunning import SimpleRules, EmpiricalTransitions
from source.simulation import nrgg, simulate_games

N_GAMES = 50_000   # per simulation; the 95% CI is roughly +/- 0.035 runs
SEED = 2026

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
pa = load_pa_table('TruMedia2026.feather')

# ---------------------------------------------------------------------------
# 2. Runner models
# ---------------------------------------------------------------------------
simple = SimpleRules()
empirical = EmpiricalTransitions.fit(pa, min_obs=20)

# Peek at one learned situation: single with a runner on 2nd and 2 outs.
# (Compare with the simple rules, where the runner always scores.)
table = empirical.describe('1B', 'R2', outs=2)
print(f"\nRunner advancement for {table.attrs['situation']} (n={table.attrs['n']}):")
print(table.head(5).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Players
# ---------------------------------------------------------------------------
# League average = SEC and ACC teams playing each other (see
# source.outcomes.LEAGUE_CONFERENCES). Pass conferences=None for all of D1.
league_avg = Player.league_average(pa)

# A real player, by name (use player_id= if a name is shared by two players).
real = Player.from_data(pa, name='Aaron Piasecki')

# A made-up power hitter: set a few rates, the rest are filled in from
# league average (scaled to make the total 1).
slugger = Player.custom('Custom Slugger', baseline=league_avg, HR=0.07, BB=0.14, SO=0.27)

print("\nPlayers:")
for p in (league_avg, real, slugger):
    print(' ', p)

# ---------------------------------------------------------------------------
# 4. NRGG under both runner models
# ---------------------------------------------------------------------------
print(f"\nNRGG (runs per 9 innings with the player in all 9 spots), {N_GAMES:,} games each:")
print(f"  {'player':<20} {'simple rules':>14} {'empirical':>14}")
for p in (league_avg, real, slugger):
    a = nrgg(p, simple, N_GAMES, seed=SEED)
    b = nrgg(p, empirical, N_GAMES, seed=SEED)
    print(f"  {p.name:<20} {a.mean:>14.2f} {b.mean:>14.2f}")

# ---------------------------------------------------------------------------
# 5. A real lineup: the 9 batters with the most PAs on one team
# ---------------------------------------------------------------------------
team = pa.loc[pa['batter'] == real.name, 'team'].mode().iloc[0]
top9 = (pa[pa['team'] == team]
        .groupby('batter_id').size()
        .sort_values(ascending=False).head(9).index)
lineup = [Player.from_data(pa, player_id=pid) for pid in top9]

print(f"\n{team} lineup (most PAs first):")
for i, p in enumerate(lineup, 1):
    print(f"  {i}. {p.name} ({p.n_pa:.0f} PA)")
print("  ", simulate_games(lineup, empirical, N_GAMES, seed=SEED))

# ---------------------------------------------------------------------------
# 6. Sanity check against the real data
# ---------------------------------------------------------------------------
print(f"\nActual runs per 9 innings, SEC/ACC vs SEC/ACC games: "
      f"{observed_runs_per_nine(league_subset(pa)):.2f}")
print(f"Actual runs per 9 innings, all of D1:                 {observed_runs_per_nine(pa):.2f}")
print("(The league-average lineup with empirical transitions should be close to the\n"
      " SEC/ACC number. It can't match exactly: steals, wild pitches and pickoffs\n"
      " aren't simulated, and runner advancement is learned from all of D1.)")
