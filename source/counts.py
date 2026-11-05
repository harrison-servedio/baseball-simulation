"""
Ball-strike counts: PA outcome probabilities given the count.

The simulator normally draws a PA's outcome from the batter's overall
rates, i.e. as if every PA started at 0-0. To start a simulation in the
middle of a PA (for example right after a called pitch that might be
challenged), we need the chance of each outcome *from a given count on*.

The "results matrix" is that table, built from the data:

    rows    : the 12 counts, '0-0' ... '3-2'
    columns : the 9 outcomes in `source.outcomes.OUTCOMES`
    value   : P(the PA ends in that outcome | the PA reached that count)

It is league-level and does not depend on outs or runners on base.

    from source.dataloaders import load_pa_table
    from source.players import Player
    from source.counts import count_probabilities, at_count

    pa = load_pa_table()
    matrix = count_probabilities(pa)        # 12 x 9, plus a 'PA' column
    avg = Player.league_average(pa)
    avg_31 = at_count(avg, matrix, '3-1')   # same batter, now ahead 3-1

Counts are written as 'balls-strikes' strings, like the raw file's
'Count' column. Two extra labels mark a call that ended the PA:

    'BB'  ball four (a walk)
    'SO'  strike three (a strikeout)

See `source/challenges.py` for run expectancy by count and ABS challenges.

Pitch-level model
-----------------
`pitch_transitions` goes one level down: the chance of each kind of pitch
(ball, called strike, other strike, end of the PA) at each count, given how
the PA ends. It is the model the pitch-by-pitch simulator in
`source/challenge_sim.py` walks through.
"""

import numpy as np
import pandas as pd

from source.outcomes import (OUTCOMES, OUTCOME_INDEX, LEAGUE_CONFERENCES, WALK, STRIKEOUT,
                             outcome_counts, league_subset)
from source.players import Player


# The 12 counts. A count's position in this tuple is `3 * balls + strikes`,
# which is also its bit in the PA table's `counts_seen` mask (see
# `source.dataloaders`).
COUNTS = tuple(f'{b}-{s}' for b in range(4) for s in range(3))

# Labels for a call that ends the PA.
WALK_COUNT = 'BB'
STRIKEOUT_COUNT = 'SO'

# Every state a PA can be in once a call is made: the 12 counts plus the 2
# endings.
ALL_COUNTS = COUNTS + (WALK_COUNT, STRIKEOUT_COUNT)

CALLS = ('strike', 'ball')

# What a pitch can do in the pitch-level model (`pitch_transitions`):
#   'ball'           a ball (taken, so it was called)
#   'called_strike'  a strike taken by the batter: the umpire's call
#   'other_strike'   a swinging strike, or a foul with fewer than 2 strikes
#   'end'            the PA ends on the pitch: in play, HBP, or a no-pitch
#                    intentional walk
# Fouls with 2 strikes don't change the count, so they are not modeled.
PITCH_STEPS = ('ball', 'called_strike', 'other_strike', 'end')
BALL_STEP, CALLED_STRIKE_STEP, OTHER_STRIKE_STEP, END_STEP = range(4)


def count_label(balls, strikes):
    """
    The label for a count.

    >>> count_label(1, 2), count_label(4, 2), count_label(3, 3)
    ('1-2', 'BB', 'SO')
    """
    if balls >= 4:
        return WALK_COUNT
    if strikes >= 3:
        return STRIKEOUT_COUNT
    return f'{balls}-{strikes}'


def parse_count(count):
    """
    Balls and strikes from '1-2' or (1, 2).

    >>> parse_count('3-1'), parse_count((0, 2))
    ((3, 1), (0, 2))
    """
    if isinstance(count, str):
        if count not in COUNTS:
            raise ValueError(f"Unknown count {count!r}; use one of {COUNTS}")
        balls, strikes = count.split('-')
        return int(balls), int(strikes)
    balls, strikes = count
    if not (0 <= balls <= 3 and 0 <= strikes <= 2):
        raise ValueError(f"Count {balls}-{strikes} is not a count a pitch can be thrown in")
    return int(balls), int(strikes)


def count_after_call(count, call):
    """
    The count after a pitch at `count` is called a ball or a strike.

    Parameters
    ----------
    count : str or (int, int)
        Count when the pitch was thrown, e.g. '1-1'.
    call : str
        'ball' or 'strike'.

    Returns
    -------
    str
        The new count, or 'BB' / 'SO' if the call ended the PA.

    >>> count_after_call('1-1', 'strike'), count_after_call('3-2', 'ball')
    ('1-2', 'BB')
    """
    balls, strikes = parse_count(count)
    if call == 'ball':
        return count_label(balls + 1, strikes)
    if call == 'strike':
        return count_label(balls, strikes + 1)
    raise ValueError(f"call must be one of {CALLS}, got {call!r}")


def count_probabilities(pa: pd.DataFrame, conferences=LEAGUE_CONFERENCES) -> pd.DataFrame:
    """
    The results matrix: outcome probabilities for a PA that reached each count.

    Every PA that passed through a count counts toward that row, so a PA
    that went 0-0, 1-0, 1-1 and ended in a single adds a single to the 0-0,
    1-0 and 1-1 rows. The 0-0 row is (almost exactly) the league-average
    rates from `source.outcomes.league_probabilities`.

    Parameters
    ----------
    pa : pd.DataFrame
        The PA table from `source.dataloaders.load_pa_table` (needs the
        `counts_seen` column).
    conferences : iterable of str or None, optional
        Which conferences make up the league. Defaults to SEC + ACC (games
        where both teams are in one of them). `None` uses all of D1. See
        `source.outcomes.league_subset`.

    Returns
    -------
    pd.DataFrame
        Indexed by count ('0-0' ... '3-2'). Columns are the 9 `OUTCOMES`
        (probabilities summing to 1 across a row) and 'PA', the number of
        PAs that reached the count.
    """
    pa = league_subset(pa, conferences)
    seen = pa['counts_seen']

    rows = {}
    for i, count in enumerate(COUNTS):
        rows[count] = outcome_counts(pa[(seen & (1 << i)) > 0], by=None).iloc[0]
    counts = pd.DataFrame(rows).T

    probs = counts[list(OUTCOMES)].div(counts['PA'], axis=0)
    probs['PA'] = counts['PA']
    probs.index.name = 'count'
    probs.columns.name = None
    return probs


def at_count(player, matrix, count):
    """
    A player's outcome probabilities once his PA has reached `count`.

    Each of the player's rates is scaled by how much the league's rate for
    that outcome changes between 0-0 and `count`, then the total is put
    back to 1:

        p_i(count) ~ p_i * matrix[count, i] / matrix['0-0', i]

    So at 0-0 you get the player back unchanged, a league-average player
    gets the matrix row (to within ~0.0001, since a few PAs are missing
    their 0-0 pitch), and a high-strikeout hitter at 0-2 still strikes out
    more than the league does at 0-2.

    Parameters
    ----------
    player : Player
    matrix : pd.DataFrame
        From `count_probabilities`.
    count : str or (int, int)
        e.g. '3-1'. Must be one of the 12 `COUNTS`: a call that ended the
        PA ('BB' / 'SO') is played out directly by
        `source.challenges.run_expectancy`.

    Returns
    -------
    Player
        A new Player named like 'League Average (3-1)'.
    """
    count = count_label(*parse_count(count))
    cols = list(OUTCOMES)
    start = matrix.loc['0-0', cols].to_numpy(dtype=float)
    now = matrix.loc[count, cols].to_numpy(dtype=float)
    # An outcome that never happens at 0-0 can't be scaled; leave it at 0.
    ratio = np.divide(now, start, out=np.zeros_like(now), where=start > 0)
    probs = player.probs * ratio
    probs = probs / probs.sum()

    return Player(name=f'{player.name} ({count})', probs=probs,
                  player_id=player.player_id, n_pa=player.n_pa)


def _count_path(mask):
    """
    The counts a PA passed through, in order, from its `counts_seen` mask,
    or None if they don't form a clean path from 0-0 (each step adds one
    ball or one strike). Paths with a gap come from pitches missing in the
    data.
    """
    path = [i for i in range(len(COUNTS)) if mask >> i & 1]
    if not path or path[0] != 0:
        return None
    for i, j in zip(path, path[1:]):
        is_ball = j == i + 3
        is_strike = j == i + 1 and i % 3 < 2
        if not (is_ball or is_strike):
            return None
    return path


# Every possible `counts_seen` mask -> its path (or None).
_PATHS = [_count_path(mask) for mask in range(1 << len(COUNTS))]


def pitch_transitions(pa: pd.DataFrame, conferences=LEAGUE_CONFERENCES) -> pd.DataFrame:
    """
    The pitch-level model: what the next pitch does, given the count and
    how the PA will end.

    This is what the pitch-by-pitch simulator (`source/challenge_sim.py`)
    walks through. For each count and each of the 9 outcomes it gives the
    chance that the next pitch is a ball, a called strike, another strike
    (swinging, or a foul with fewer than 2 strikes) or ends the PA, among
    PAs that reached that count and ended in that outcome. Fouls with 2
    strikes don't change the count and are not modeled.

    Why condition on the outcome: the simulator draws the PA's outcome first,
    from the batter's own rates (`at_count`), and then the pitches that lead
    to it. So every batter's outcome rates are exactly what they are in the
    PA-level simulator, and the count path still looks like real PAs that
    ended that way (a walk goes through 3 balls, a strikeout through 2
    strikes).

    How each step of a PA is classified:
      - between two counts on the PA's path: 'ball' or a strike, 'called'
        if the PA's `called_strikes` mask has that count
      - at the last count: ball four ('BB' with 3 balls) is a 'ball';
        strike three ('SO' with 2 strikes) is a called or other strike;
        anything else is 'end'

    Parameters
    ----------
    pa : pd.DataFrame
        The PA table from `source.dataloaders.load_pa_table` (needs the
        `counts_seen` and `called_strikes` columns).
    conferences : iterable of str or None, optional
        Which conferences make up the league, as in `count_probabilities`.
        Defaults to SEC + ACC. `None` uses all of D1.

    Returns
    -------
    pd.DataFrame
        Indexed by ('count', 'outcome'): 12 counts x 9 outcomes. Columns are
        the 4 `PITCH_STEPS` (probabilities summing to 1 across a row) and
        'PA', the number of PAs behind the row.

    Notes
    -----
    - Only PAs whose `counts_seen` is a clean path from 0-0 are used (about
      99%). Balls in play with an unknown hit type ('BIP_UNK') are left
      out; GO and FO PAs go through the same counts.
    - A (count, outcome) with no PAs gets a fixed path: a walk keeps taking
      balls, a strikeout keeps taking (non-called) strikes, and anything
      else ends the PA on the next pitch. Its 'PA' is 0.
    """
    pa = league_subset(pa, conferences)
    pa = pa[pa['outcome'].isin(OUTCOMES)]
    groups = pa.groupby(['counts_seen', 'called_strikes', 'outcome']).size()

    n_counts, n_outcomes = len(COUNTS), len(OUTCOMES)
    tally = np.zeros((n_counts, n_outcomes, len(PITCH_STEPS)))
    for (seen, called, outcome), n in groups.items():
        path = _PATHS[seen]
        if path is None:
            continue
        o = OUTCOME_INDEX[outcome]
        for i, j in zip(path, path[1:]):
            if j == i + 3:
                step = BALL_STEP
            else:
                step = CALLED_STRIKE_STEP if called >> i & 1 else OTHER_STRIKE_STEP
            tally[i, o, step] += n

        last = path[-1]
        balls, strikes = divmod(last, 3)
        if o == WALK and balls == 3:
            step = BALL_STEP
        elif o == STRIKEOUT and strikes == 2:
            step = CALLED_STRIKE_STEP if called >> last & 1 else OTHER_STRIKE_STEP
        else:
            step = END_STEP
        tally[last, o, step] += n

    n_pa = tally.sum(axis=2)
    probs = np.divide(tally, n_pa[:, :, None], out=np.zeros_like(tally), where=n_pa[:, :, None] > 0)
    for c, o in zip(*np.nonzero(n_pa == 0)):
        if o == WALK:
            probs[c, o, BALL_STEP] = 1.0
        elif o == STRIKEOUT:
            probs[c, o, OTHER_STRIKE_STEP] = 1.0
        else:
            probs[c, o, END_STEP] = 1.0

    index = pd.MultiIndex.from_product([COUNTS, OUTCOMES], names=['count', 'outcome'])
    table = pd.DataFrame(probs.reshape(-1, len(PITCH_STEPS)), index=index, columns=list(PITCH_STEPS))
    table['PA'] = n_pa.reshape(-1).astype(int)
    return table
