"""
Run expectancy by base state, outs and count, and the value of an ABS
challenge.

Run expectancy (RE) is the average number of runs scored from a point in a
half-inning until the 3rd out. Runs already scored don't count. Here the
point is "a pitch at this count, with these runners and outs", and RE is
found by simulating the rest of the half-inning many times:

    1. The batter at the plate finishes his PA with an outcome drawn from
       his count-adjusted rates (`source.counts.at_count`).
    2. The rest of the half-inning is simulated as usual
       (`source.simulation.simulate_half_inning`).

ABS challenges
--------------
When a pitch at count b-s is called, the PA moves to one of two counts:

    call            if it stands (no challenge)   if overturned
    -------------   ---------------------------   ----------------------
    called strike   b-(s+1), or SO if s = 2       (b+1)-s, or BB if b = 3
    called ball     (b+1)-s, or BB if b = 3       b-(s+1), or SO if s = 2

The offense challenges called strikes and the defense challenges called
balls. Comparing RE at the two counts gives the run value of the challenge.
'BB' and 'SO' mean the call ended the PA with a walk or strikeout. A
challengeable call is a simple play, so these don't use the runner model:
ball four forces runners up, and strike three is an out with the runners
holding. (The runner model's strikeout data includes dropped third strikes
and strike-'em-out-throw-'em-out double plays, which almost all come on
swinging strikes.)

This is on a half-inning scale only. The value of a challenge you lose (and
so can't use later in the game) is not modeled.

Usage
-----
    from source.counts import count_probabilities
    from source.challenges import run_expectancy_table, challenge_table, challenge_value

    matrix = count_probabilities(pa)
    re_table = run_expectancy_table(avg, model, matrix, seed=1)  # ~1 minute
    challenges = challenge_table(re_table)                       # instant
    challenge_value(re_table, 'R1', outs=1, count='1-1', call='strike')
"""

import math
import random

import pandas as pd

from source.baserunning import bases_to_str, str_to_bases, force_advance
from source.counts import (COUNTS, ALL_COUNTS, CALLS, WALK_COUNT, STRIKEOUT_COUNT, at_count,
                           count_after_call, count_label, parse_count)
from source.simulation import as_lineup, simulate_half_inning, SimulationResult


def run_expectancy(lineup, runner_model, matrix, bases=0, outs=0, count='0-0',
                   n_innings=100_000, seed=None, batter_idx=0):
    """
    Run expectancy for one situation, by simulation.

    Parameters
    ----------
    lineup : Player or list of Player
        A single Player bats in every spot; a list is the batting order.
    runner_model : SimpleRules or EmpiricalTransitions
    matrix : pd.DataFrame
        Outcome probabilities by count, from
        `source.counts.count_probabilities`.
    bases : int or str, optional
        Runners on base, as a 3-bit int or a string like 'R1-R3' (default
        empty).
    outs : int, optional
        Outs (0, 1 or 2).
    count : str or (int, int), optional
        Count of the batter at the plate, e.g. '1-2'. 'BB' or 'SO' means
        a call just ended his PA with ball four or strike three, played as
        a simple walk (forced runners move up) or out (runners hold); the
        simulation then starts with the next batter.
    n_innings : int, optional
        Half-innings to simulate. RE's standard deviation is about 1 run,
        so 100,000 gives a 95% CI of about +/-0.007.
    seed : int, optional
        Seed for reproducible results.
    batter_idx : int, optional
        Position in `lineup` of the batter at the plate (0-indexed). Only
        matters for a lineup of different players.

    Returns
    -------
    SimulationResult
        `.mean` is the run expectancy. `.runs` holds the runs scored in
        each simulated half-inning.
    """
    lineup = as_lineup(lineup)
    if isinstance(bases, str):
        bases = str_to_bases(bases)
    if outs not in (0, 1, 2):
        raise ValueError(f"outs must be 0, 1 or 2, got {outs!r}")

    rng = random.Random(seed)
    if count in (WALK_COUNT, STRIKEOUT_COUNT):
        # The call ended the PA: play it out, then start with the next batter.
        bases, scored, outs = _end_pa(count, bases, outs)
        if outs >= 3:
            return SimulationResult.from_runs([scored] * n_innings, per='half-inning')
        next_idx = (batter_idx + 1) % len(lineup)
        runs = [scored + simulate_half_inning(lineup, next_idx, runner_model, rng, outs, bases)[0]
                for _ in range(n_innings)]
    else:
        batter = at_count(lineup[batter_idx], matrix, count)
        runs = [simulate_half_inning(lineup, batter_idx, runner_model, rng, outs, bases, batter)[0]
                for _ in range(n_innings)]
    return SimulationResult.from_runs(runs, per='half-inning')


def _end_pa(count, bases, outs):
    """
    (bases, runs, outs) after a call ends the PA: 'BB' forces runners up,
    'SO' is an out with the runners holding.
    """
    if count == WALK_COUNT:
        bases, runs, _ = force_advance(bases)
        return bases, runs, outs
    return bases, 0, outs + 1


def run_expectancy_table(lineup, runner_model, matrix, n_innings=100_000, seed=None, batter_idx=0):
    """
    Run expectancy for every base state, number of outs and count.

    That is 8 base states x 3 outs x 14 counts (the 12 counts plus 'BB'
    and 'SO') = 336 situations, each simulated `n_innings` times. With the
    default 100,000 this takes about a minute.

    Every situation is simulated with the same `seed`, so the table is
    reproducible.

    Parameters
    ----------
    lineup, runner_model, matrix, n_innings, seed, batter_idx
        See `run_expectancy`.

    Returns
    -------
    pd.DataFrame
        One row per situation with columns:
          - 'bases' (3-bit int), 'base_state' (e.g. 'R1-R3'), 'outs',
            'count'
          - 're': run expectancy, plus 'sd', 'n' (innings simulated) and
            the 95% CI 'ci_low' / 'ci_high'
        Pass it to `challenge_table`, `challenge_value` or `re_matrix`.
    """
    rows = []
    for outs in range(3):
        for bases in range(8):
            for count in ALL_COUNTS:
                result = run_expectancy(lineup, runner_model, matrix, bases, outs, count,
                                        n_innings, seed, batter_idx)
                rows.append({
                    'bases': bases, 'base_state': bases_to_str(bases), 'outs': outs, 'count': count,
                    're': result.mean, 'sd': result.sd, 'n': result.n_games,
                    'ci_low': result.ci_low, 'ci_high': result.ci_high,
                })
    return pd.DataFrame(rows)


def re_matrix(re_table, value='re'):
    """
    The RE table laid out as a grid: one row per (outs, base state), one
    column per count.

    Parameters
    ----------
    re_table : pd.DataFrame
        From `run_expectancy_table`.
    value : str, optional
        Which column to show ('re' by default, or e.g. 'ci_high').

    Returns
    -------
    pd.DataFrame
    """
    grid = re_table.pivot(index=['outs', 'bases'], columns='count', values=value)
    grid = grid[list(ALL_COUNTS)]
    grid.index = grid.index.set_levels(
        [bases_to_str(b) for b in grid.index.levels[1]], level='bases')
    grid.index = grid.index.rename('base_state', level='bases')
    grid.columns.name = None
    return grid


def challenge_value(re_table, bases, outs, count, call, p_overturn=1.0) -> pd.Series:
    """
    Run expectancy if a called pitch is challenged vs. if it isn't.

    Parameters
    ----------
    re_table : pd.DataFrame
        From `run_expectancy_table`.
    bases : int or str
        Runners on base when the pitch was thrown, e.g. 'R1' or 1.
    outs : int
        Outs when the pitch was thrown.
    count : str or (int, int)
        Count when the pitch was thrown, e.g. '1-1' (*before* the call).
    call : str
        How the pitch was called: 'strike' (the offense would challenge)
        or 'ball' (the defense would challenge).
    p_overturn : float, optional
        Chance the challenge wins. The default 1.0 gives the RE if the call
        is overturned.

    Returns
    -------
    pd.Series
        - 'count_if_stands' / 'count_if_overturned': the count after the
          call, either way
        - 're_stands': RE if not challenged (or the challenge loses)
        - 're_overturned': RE if the call is overturned
        - 're_challenged': p * re_overturned + (1 - p) * re_stands
        - 'delta_re': re_challenged - re_stands, in the batting team's runs
        - 'delta_ci': half-width of a 95% CI for delta_re
        - 'challenger_gain': runs gained by the challenging team
          (delta_re for a called strike, -delta_re for a called ball)
    """
    if isinstance(bases, str):
        bases = str_to_bases(bases)
    if not 0 <= p_overturn <= 1:
        raise ValueError(f"p_overturn must be between 0 and 1, got {p_overturn}")
    re = re_table.set_index(['bases', 'outs', 'count'])
    return pd.Series(_challenge_row(re, bases, outs, count, call, p_overturn))


def challenge_table(re_table, p_overturn=1.0) -> pd.DataFrame:
    """
    `challenge_value` for every base state, outs, count and call.

    8 base states x 3 outs x 12 counts x 2 calls = 576 rows. Only looks
    values up in `re_table`, so it's instant: re-run it with a different
    `p_overturn` freely.

    Parameters
    ----------
    re_table : pd.DataFrame
        From `run_expectancy_table`.
    p_overturn : float, optional
        Chance a challenge wins (default 1.0).

    Returns
    -------
    pd.DataFrame
        Columns 'bases', 'base_state', 'outs', 'count', 'call',
        'challenger' ('offense' / 'defense') and the values described in
        `challenge_value`.
    """
    if not 0 <= p_overturn <= 1:
        raise ValueError(f"p_overturn must be between 0 and 1, got {p_overturn}")
    re = re_table.set_index(['bases', 'outs', 'count'])
    rows = [_challenge_row(re, bases, outs, count, call, p_overturn)
            for outs in range(3) for bases in range(8) for count in COUNTS for call in CALLS]
    return pd.DataFrame(rows)


def _challenge_row(re, bases, outs, count, call, p_overturn, z=1.96):
    """One row of `challenge_table`. `re` is the RE table indexed by (bases, outs, count)."""
    if call not in CALLS:
        raise ValueError(f"call must be one of {CALLS}, got {call!r}")
    count = count_label(*parse_count(count))
    other = 'ball' if call == 'strike' else 'strike'
    if_stands = count_after_call(count, call)
    if_overturned = count_after_call(count, other)

    stands = re.loc[(bases, outs, if_stands)]
    overturned = re.loc[(bases, outs, if_overturned)]

    re_challenged = p_overturn * overturned['re'] + (1 - p_overturn) * stands['re']
    delta = re_challenged - stands['re']
    se_stands = stands['sd'] / math.sqrt(stands['n'])
    se_overturned = overturned['sd'] / math.sqrt(overturned['n'])

    return {
        'bases': bases, 'base_state': bases_to_str(bases), 'outs': outs, 'count': count,
        'call': call, 'challenger': 'offense' if call == 'strike' else 'defense',
        'count_if_stands': if_stands, 'count_if_overturned': if_overturned,
        're_stands': stands['re'], 're_overturned': overturned['re'], 're_challenged': re_challenged,
        'delta_re': delta,
        'delta_ci': z * p_overturn * math.sqrt(se_stands ** 2 + se_overturned ** 2),
        'challenger_gain': delta if call == 'strike' else -delta,
    }
