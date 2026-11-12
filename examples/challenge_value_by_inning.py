"""
Walkthrough: how much an ABS challenge is worth at each point in the game.

Run from the repo root:

    .venv/bin/python examples/challenge_value_by_inning.py

A challenge you are holding is worth the runs you expect to gain with it
before the game ends. That depends on two things:

  - the run value of winning a challenge in each situation (base state,
    outs, count, call), from the run expectancy table, and
  - how likely each situation is to come up, from the pitch-by-pitch
    simulator.

The team here does not pick its spots: it challenges a fixed share of the
close calls it sees (`CHALLENGE_RATE`, 50%), chosen at random, whatever the
situation and however sure it is. Early in the game there are many close
calls still to come, so a challenge will probably get used, maybe several
times if it keeps winning. Late in the game there are few left, and at the
last out an unused challenge is worth nothing. So the value falls as the
game goes on.

It:
  1. loads the data, the runner model, a league-average player and the
     pitch model,
  2. finds the run value of winning a challenge in every situation,
  3. finds how often each situation comes up in a half-inning,
  4. works backward from the last out to get the value of a challenge at
     the start of every half-inning,
  5. plots it and saves `challenge_value_by_inning.png` next to this file.

How step 4 works
----------------
`value[n]` is the runs that n challenges are worth from now to the end of
the game. Start at the last out, where value = 0, and step backward in
time. In each small step a close call may come up, and the team challenges
it with chance `CHALLENGE_RATE`. The challenge:

    wins  (chance = confidence):  gain the run value, keep the challenge
    loses (1 - confidence):       down to n - 1 challenges

so, with price = value[n] - value[n - 1] (what the nth challenge is worth
if kept), a challenge changes the runs the team expects by

    confidence * gain  -  (1 - confidence) * price

and the value of holding the challenge grows by `CHALLENGE_RATE` times the
average of that over every close call that could come up, weighted by how
likely it is.

The price is the curve this script plots: what one more challenge in hand
is worth.

Simplifications: close calls are treated as independent draws from the
situation frequencies in step 3, spread evenly through the half-inning.
Everything is in runs from a run expectancy table, so the score doesn't
matter (a run in a tie game counts the same as in a blowout). The team is
the visitors (they bat in the top half) and the game always goes the full
`INNINGS`.
"""

import os
import random
import sys
import time
from collections import Counter

# Let `import source...` work when this file is run directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source.dataloaders import load_pa_table
from source.players import Player
from source.baserunning import EmpiricalTransitions
from source.challenges import run_expectancy_table, challenge_table
from source.challenge_sim import PitchModel, ChallengeRules, simulate_half_inning_by_pitch

# ---------------------------------------------------------------------------
# Settings: change these by hand
# ---------------------------------------------------------------------------
CLOSE_CALL_RATE = 0.05    # chance a called pitch is close enough to challenge
CHALLENGE_RATE = 1      # share of close calls the team challenges, picked at random
CHALLENGES = 4            # challenges the team starts the game with
INNINGS = 9

# The chance a challenge wins is different for every close call (see
# `ChallengeRules.overturn_prob`). These are the values it can take, all
# equally likely: 0.005, 0.015, ..., 0.995, i.e. uniform between 0 and 1.
# The team challenges at random, so only the average (0.5) matters here.
CONFIDENCE = (np.arange(100) + 0.5) / 100
OVERTURN_PROBABILITY = 0.5

N_RE = 100_000            # half-innings per situation in the RE table
N_FREQ = 100_000          # half-innings simulated to count the situations
STEPS = 50                # time steps per half-inning in the backward pass
SEED = 2026

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

# ---------------------------------------------------------------------------
# 1. Data, runner model, player, pitch model
# ---------------------------------------------------------------------------
pa = load_pa_table('TruMedia2026.feather')
empirical = EmpiricalTransitions.fit(pa, min_obs=20)
league_avg = Player.league_average(pa)
pitch_model = PitchModel.fit(pa)

# ---------------------------------------------------------------------------
# 2. Run value of winning a challenge, in every situation
# ---------------------------------------------------------------------------
# 576 rows: 8 base states x 3 outs x 12 counts x called strike / ball.
# 'challenger_gain' is the runs the challenging team gains if the call is
# overturned.
print(f"\nSimulating run expectancy: 336 situations x {N_RE:,} half-innings...")
start = time.time()
re_table = run_expectancy_table(league_avg, empirical, pitch_model.matrix, n_innings=N_RE, seed=SEED)
situations = challenge_table(re_table, p_overturn=1.0)
print(f"done in {time.time() - start:.0f}s")

# ---------------------------------------------------------------------------
# 3. How often each situation comes up
# ---------------------------------------------------------------------------
# Play half-innings pitch by pitch and count the called pitches in each
# situation. The trick: make every called pitch a "close call"
# (close_call_rate=1), and use a strategy that writes the situation down
# and never challenges.
seen = Counter()


def count_it(s):
    seen[s.key] += 1    # s.key = (bases, outs, count, call)
    return False        # never challenge


count_every_call = ChallengeRules(close_call_rate=1.0, overturn_prob=OVERTURN_PROBABILITY)
rng = random.Random(SEED)
for _ in range(N_FREQ):
    simulate_half_inning_by_pitch(league_avg, empirical, pitch_model, count_every_call,
                                  offense=count_it, defense=count_it, rng=rng)

# 'per_half_inning': called pitches like this one in an average half-inning.
situations['per_half_inning'] = [seen[(row.bases, row.outs, row.count, row.call)] / N_FREQ
                                 for row in situations.itertuples()]

# The team challenges called strikes when it bats and called balls when it
# fields.
strikes = situations[situations['call'] == 'strike']
balls = situations[situations['call'] == 'ball']

cols = ['base_state', 'outs', 'count', 'call', 'challenger_gain', 'per_half_inning']
print("\nMost valuable calls to overturn (they are rare):")
print(situations.sort_values('challenger_gain', ascending=False)[cols].head(6)
      .round(4).to_string(index=False))
print("\nMost common called pitches (they are worth little):")
print(situations.sort_values('per_half_inning', ascending=False)[cols].head(6)
      .round(4).to_string(index=False))

print("\nPer half-inning:")
for name, calls in (('called strikes (batting)', strikes), ('called balls (fielding)', balls)):
    n = calls['per_half_inning'].sum()
    average_gain = np.average(calls['challenger_gain'], weights=calls['per_half_inning'])
    print(f"  {n:.2f} {name}, {n * CLOSE_CALL_RATE:.2f} of them close; "
          f"overturning an average one is worth {average_gain:.3f} runs")


# ---------------------------------------------------------------------------
# 4. Work backward from the last out
# ---------------------------------------------------------------------------
def gain_from_next_close_call(calls, price):
    """
    Runs the team expects to gain from one close call, when using up a
    challenge costs `price` runs.

    The team challenges it with chance `CHALLENGE_RATE`, whatever the call
    and the confidence. A challenge is worth
    confidence * gain - (1 - confidence) * price; the result is the average
    of that over calls (weighted by how often they come up) and over
    `CONFIDENCE`, times `CHALLENGE_RATE`.
    """
    gain = calls['challenger_gain'].to_numpy()
    likelihood = calls['per_half_inning'].to_numpy()
    likelihood = likelihood / likelihood.sum()
    # One row per confidence, one column per kind of call.
    if_challenged = np.outer(CONFIDENCE, gain) - (1 - CONFIDENCE)[:, None] * price
    return CHALLENGE_RATE * (if_challenged.mean(axis=0) * likelihood).sum()


# value[n] = runs that n challenges are worth from now to the end of the game.
value = np.zeros(CHALLENGES + 1)
rows = [{'inning': INNINGS + 1, 'half': 'end', 'value': value.copy()}]

for inning in range(INNINGS, 0, -1):
    for half in ('bottom', 'top'):          # backward in time: the bottom half first
        calls = strikes if half == 'top' else balls     # the visitors bat in the top
        # Chance of a close call in one time step.
        chance = calls['per_half_inning'].sum() * CLOSE_CALL_RATE / STEPS
        for _ in range(STEPS):
            price = np.diff(value)          # price[n - 1] = value[n] - value[n - 1]
            value[1:] += [chance * gain_from_next_close_call(calls, p) for p in price]
        rows.append({'inning': inning, 'half': half, 'value': value.copy()})

rows.reverse()      # in game order: top 1st, bottom 1st, ..., end

# One column per challenge: what the nth one is worth (the price above).
timeline = pd.DataFrame({'inning': [r['inning'] for r in rows], 'half': [r['half'] for r in rows]})
for n in range(1, CHALLENGES + 1):
    timeline[f'challenge_{n}'] = [r['value'][n] - r['value'][n - 1] for r in rows]
timeline['all'] = [r['value'][CHALLENGES] for r in rows]

# `timeline` has a row for the start of every half-inning. (The value falls
# faster in the bottom halves: the team is fielding, and there are about
# twice as many called balls to challenge as called strikes.) Print and plot
# the start of each inning and the end of the game.
by_inning = timeline[timeline['half'] != 'bottom']
print(f"\nRun value of each challenge at the start of each inning (inning {INNINGS + 1} = the "
      f"game is over).\nchallenge_1 = the last one you have, challenge_2 = a spare, ...; "
      f"all = the {CHALLENGES} together:")
print(by_inning.drop(columns='half').round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = ('#fcfcfb', '#0b0b0b', '#52514e', '#898781',
                                                  '#e1e0d9', '#c3c2b7')
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

plt.rcParams['font.family'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

x = by_inning['inning']      # the game ends at INNINGS + 1
for n in range(1, CHALLENGES + 1):
    y = by_inning[f'challenge_{n}']
    color = SERIES[(n - 1) % len(SERIES)]
    label = 'Your last challenge' if n == 1 else f'Challenge {n} (a spare)'
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle='round', label=label)
    # Mark and label the value at the first pitch.
    ax.plot(x.iloc[0], y.iloc[0], 'o', color=color, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, clip_on=False)
    ax.annotate(f'{y.iloc[0]:.2f} runs', (x.iloc[0], y.iloc[0]), xytext=(9, 7),
                textcoords='offset points', color=INK, fontsize=10)

ax.set_xlim(0.8, INNINGS + 1.1)
ax.set_ylim(0, None)
ax.set_xticks(range(1, INNINGS + 2))
ax.set_xticklabels([str(i) for i in range(1, INNINGS + 1)] + ['End'])
ax.set_xlabel('Start of inning', color=INK_SECONDARY, fontsize=10)
ax.set_ylabel('Runs', color=INK_SECONDARY, fontsize=10)
ax.tick_params(colors=MUTED, length=0, labelsize=10)
ax.grid(axis='y', color=GRID, linewidth=1)
ax.set_axisbelow(True)
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color(AXIS)
if CHALLENGES > 1:
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=10, loc='upper right')

fig.suptitle('An ABS challenge is worth less as the game goes on', x=0.065, y=0.965, ha='left',
             color=INK, fontsize=14, fontweight='bold')
ax.set_title(f'Runs a challenge is expected to gain before the game ends, for a team that '
             f'challenges {CHALLENGE_RATE:.0%} of close calls\nat random. League-average visiting '
             f'team, {CLOSE_CALL_RATE:.0%} of called pitches are close, a winning challenge is kept.',
             loc='left', color=INK_SECONDARY, fontsize=10, pad=14)
fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.11)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'challenge_value_by_inning.png')
fig.savefig(out, dpi=200, facecolor=SURFACE)
print(f"\nSaved the plot to {out}")
plt.show()
