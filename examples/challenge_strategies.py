"""
Walkthrough: testing ABS challenge strategies with the pitch-by-pitch simulator.

Unlike the other examples, this one uses the pitch-by-pitch half-inning
simulator (`source.challenge_sim`): every pitch of every PA is played out, so
close calls can be challenged as they happen.

Run from the repo root:

    .venv/bin/python examples/challenge_strategies.py

It:
  1. loads the data, the runner model, a league-average player and the
     pitch model,
  2. sets how often close calls happen (RULES, below),
  3. plays one half-inning and prints every close call,
  4. compares challenge strategies over many half-innings, including
     `my_strategy`, which is yours to fill in.
"""

import os
import random
import sys
import time

# Let `import source...` work when this file is run directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd

from source.dataloaders import load_pa_table
from source.players import Player
from source.baserunning import EmpiricalTransitions
from source.challenges import run_expectancy_table
from source.challenge_sim import (PitchModel, ChallengeRules, PitchSituation,
                                  simulate_half_inning_by_pitch, compare_strategies_by_pitch,
                                  never_challenge, always_challenge, confidence_at_least,
                                  challenge_gains, gain_at_least)

N_INNINGS = 50_000    # half-innings per strategy; 95% CI on each mean is roughly +/- 0.011 runs
SEED = 2026

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)

# ---------------------------------------------------------------------------
# 1. Data, runner model, player, pitch model
# ---------------------------------------------------------------------------
pa = load_pa_table('TruMedia2026.feather')
empirical = EmpiricalTransitions.fit(pa, min_obs=20)
league_avg = Player.league_average(pa)
pitch_model = PitchModel.fit(pa)    # count probabilities + pitch-level transitions

# ---------------------------------------------------------------------------
# 2. How often close calls happen: set these by hand
# ---------------------------------------------------------------------------
# A league-average half-inning has about 10 called pitches (balls and taken
# strikes), so a rate of 0.05 gives about 0.5 close calls per half-inning.
# Other forms:
#   close_call_rate={'ball': 0.04, 'strike': 0.06}
#   close_call_rate=lambda count, call: 0.10 if count == '3-2' else 0.04
# overturn_prob=None draws each close call's chance of being wrong uniformly
# from 0-1; use a float for a fixed chance, or e.g.
#   overturn_prob=lambda rng: rng.betavariate(2, 2)
RULES = ChallengeRules(
    close_call_rate=0.05,
    overturn_prob=None,
    offense_challenges=4,
    defense_challenges=4,
    keep_on_success=True,
)


# ---------------------------------------------------------------------------
# 3. Your strategy
# ---------------------------------------------------------------------------
def my_strategy(s: PitchSituation) -> bool:
    """
    Return True to challenge this close call.

    Only asked when your team has a challenge left. As the offense you see
    close called strikes; as the defense, close called balls.

    What you can use:
        s.count                  count before the pitch, e.g. '1-2'
        s.call                   'strike' or 'ball'
        s.count_if_stands        count if the call stands ('SO' / 'BB' if it ends the PA)
        s.count_if_overturned    count if it's overturned
        s.confidence             chance the challenge wins
        s.outs, s.bases          outs and runners (s.base_state reads like 'R1-R3')
        s.challenges_left        your challenges left, including this one
        s.opponent_challenges_left
        s.runs                   runs scored so far this half-inning
        s.batter, s.batter_idx   who's up
        s.pa_of_inning, s.pitch_of_pa
        s.challenger             'offense' or 'defense'
    """
    # TODO: your rule here. For example:
    #   if s.count_if_stands == 'SO':
    #       return s.confidence > 0.4
    #   return s.confidence > 0.75 and s.challenges_left == 2
    return False


# ---------------------------------------------------------------------------
# 4. One half-inning, close call by close call
# ---------------------------------------------------------------------------
# Play half-innings until one has at least 2 close calls, and show it.
rng = random.Random(SEED)
while True:
    example = simulate_half_inning_by_pitch(league_avg, empirical, pitch_model, RULES,
                                            offense=confidence_at_least(0.6),
                                            defense=confidence_at_least(0.6),
                                            rng=rng, log=True)
    if example.close_calls >= 2:
        break
print(f"\nOne half-inning (both teams challenge at confidence >= 0.6): {example.runs} runs, "
      f"{example.called_pitches} called pitches, {example.close_calls} close calls")
for e in example.close_call_log:
    s = e.situation
    result = 'overturned' if e.overturned else ('upheld' if e.challenged else 'not challenged')
    print(f"  PA {s.pa_of_inning}, {s.base_state}, {s.outs} out, {s.count} called {s.call}, "
          f"confidence {s.confidence:.2f}: {result}")

# ---------------------------------------------------------------------------
# 5. Compare strategies
# ---------------------------------------------------------------------------
# Each entry is (offense strategy, defense strategy). The first is the
# baseline: 'vs_baseline' is the difference in runs per half-inning from it.
# To test an offense strategy, hold the defense fixed (and vice versa).
strategies = {
    'never': (never_challenge, never_challenge),
    'offense always': (always_challenge, never_challenge),
    'offense conf >= 0.6': (confidence_at_least(0.6), never_challenge),
    # 'offense my_strategy': (my_strategy, never_challenge),
    'defense conf >= 0.6': (never_challenge, confidence_at_least(0.6)),
}

print(f"\nPitch-by-pitch: {N_INNINGS:,} half-innings per strategy, bases empty, 0 out...")
start = time.time()
results = compare_strategies_by_pitch(league_avg, empirical, pitch_model, RULES, strategies,
                                      n_innings=N_INNINGS, seed=SEED)
print(f"done in {time.time() - start:.0f}s")
cols = ['runs', 'vs_baseline', 'vs_baseline_ci', 'close_calls', 'offense_challenges', 'offense_won',
        'offense_left', 'defense_challenges', 'defense_won', 'defense_left']
print(results[cols].round(3))

# ---------------------------------------------------------------------------
# 6. Strategies that use run values
# ---------------------------------------------------------------------------
# The RE table (PA-level, from source.challenges) says how many runs winning
# each challenge is worth. `gains[s.key]` looks one up inside a strategy.
print("\nBuilding the run expectancy table for run-value strategies...")
re_table = run_expectancy_table(league_avg, empirical, pitch_model.matrix, n_innings=50_000, seed=SEED)
gains = challenge_gains(re_table)

strategies = {
    'never': (never_challenge, never_challenge),
    'offense always': (always_challenge, never_challenge),
    'offense conf >= 0.6': (confidence_at_least(0.6), never_challenge),
    'offense gain >= 0.03': (gain_at_least(gains, 0.03), never_challenge),
    'offense gain >= 0.08': (gain_at_least(gains, 0.08), never_challenge),
}
results = compare_strategies_by_pitch(league_avg, empirical, pitch_model, RULES, strategies,
                                      n_innings=N_INNINGS, seed=SEED)
print(results[cols].round(3))

# Other starting states: pass outs=, bases= ('R1-R3') or count= to
# compare_strategies_by_pitch. A real lineup works too: pass a list of 9 Players.
