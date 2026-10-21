"""
Runner advancement: where everyone ends up after a PA.

Once the simulator has drawn an outcome (single, ground out, ...), it needs
to know what happens to the batter and the runners. Beaudoin's main
improvement over earlier simulators is learning this from data instead of
using fixed rules, so this module has both:

    SimpleRules           - the fixed rules of D'Esopo & Lefkowitz (1977)
                            that most earlier work used (paper p. 275).
                            Useful as a baseline to compare against.
    EmpiricalTransitions  - probabilities learned from the TruMedia data,
                            for every (outcome, runners on base, outs).

Both have the same method, so the simulator can use either one:

    new_bases, runs, outs_added = model.advance(outcome, bases, outs, rng)

where
    outcome   : int, index into `source.outcomes.OUTCOMES` (0 = 1B ... 8 = FO)
    bases     : int, 3-bit base state before the play
                (1 = runner on 1st, 2 = on 2nd, 4 = on 3rd; 0 = empty, 7 = loaded)
    outs      : int, outs before the play (0, 1, 2)
    rng       : random.Random, used to draw random numbers
    new_bases : int, base state after the play (ignored if the inning ended)
    runs      : int, runs that scored on the play
    outs_added: int, outs made on the play (the inning is over when
                outs + outs_added >= 3)
"""

import bisect
from collections import defaultdict

import pandas as pd

from source.outcomes import (OUTCOMES, OUTCOME_INDEX, SINGLE, DOUBLE, TRIPLE, HOME_RUN,
                             WALK, HIT_BY_PITCH)


# --------------------------------------------------------------------------
# Base-state helpers
# --------------------------------------------------------------------------

FIRST, SECOND, THIRD = 1, 2, 4


def runners_on(bases: int) -> int:
    """Number of runners on base for a 3-bit base state."""
    return (bases & 1) + ((bases >> 1) & 1) + ((bases >> 2) & 1)


def bases_to_str(bases: int) -> str:
    """
    Readable base state, in the paper's notation.

    >>> bases_to_str(0), bases_to_str(5), bases_to_str(7)
    ('empty', 'R1-R3', 'R1-R2-R3')

    -1 (used in the data for "the inning ended") is shown as 'inning over'.
    """
    if bases < 0:
        return 'inning over'
    if bases == 0:
        return 'empty'
    return '-'.join(f'R{i + 1}' for i in range(3) if bases & (1 << i))


def str_to_bases(text: str) -> int:
    """
    Parse a base state written like 'R1-R3', '13', '1_3' or 'empty'.

    >>> str_to_bases('R1-R3'), str_to_bases('23'), str_to_bases('empty')
    (5, 6, 0)
    """
    text = text.strip().lower()
    if text in ('', 'empty', '0', '---', '000'):
        return 0
    bases = 0
    for ch in text:
        if ch in '123':
            bases |= 1 << (int(ch) - 1)
    return bases


# --------------------------------------------------------------------------
# Model 1: fixed rules (D'Esopo & Lefkowitz 1977)
# --------------------------------------------------------------------------

class SimpleRules:
    """
    The simple runner rules used by most simulators before this paper.

    From the bulleted list on p. 275 of the paper:

    - Single:  batter and all runners move up one base, except a runner on
               second, who scores.
    - Double:  batter and all runners move up two bases.
    - Triple:  batter and all runners move up three bases (everyone scores).
    - Home run: batter and all runners score.
    - Walk / HBP: batter to first; only runners who are forced move up.
    - Any out (SO, GO, FO): batter is out and the runners stay put.

    So there are no double plays, no sac flies, no runner thrown out, and
    no taking an extra base. The paper argues this misleads results,
    especially late in close games.

    Nothing here is random, so `rng` is ignored.
    """

    name = 'Simple rules (D\'Esopo & Lefkowitz)'

    def advance(self, outcome, bases, outs, rng=None):
        """Apply the rules above. See the module docstring for arguments."""
        on1, on2, on3 = bases & FIRST, bases & SECOND, bases & THIRD

        if outcome == SINGLE:
            # Runners on 2nd and 3rd score; runner on 1st goes to 2nd.
            runs = (1 if on2 else 0) + (1 if on3 else 0)
            return FIRST | (SECOND if on1 else 0), runs, 0

        if outcome == DOUBLE:
            # Runners on 2nd and 3rd score; runner on 1st goes to 3rd.
            runs = (1 if on2 else 0) + (1 if on3 else 0)
            return SECOND | (THIRD if on1 else 0), runs, 0

        if outcome == TRIPLE:
            return THIRD, runners_on(bases), 0

        if outcome == HOME_RUN:
            return 0, runners_on(bases) + 1, 0

        if outcome in (WALK, HIT_BY_PITCH):
            return force_advance(bases)

        # SO, GO, FO: batter out, runners hold.
        return bases, 0, 1


def force_advance(bases):
    """
    Walk / HBP: put the batter on first and push forced runners up one base.

    A runner is forced only if every base behind him is occupied, e.g. with
    runners on 1st and 3rd, the runner on 3rd stays.

    Returns
    -------
    (new_bases, runs, outs_added)
    """
    if not bases & FIRST:
        return bases | FIRST, 0, 0
    if not bases & SECOND:
        return bases | FIRST | SECOND, 0, 0
    if not bases & THIRD:
        return FIRST | SECOND | THIRD, 0, 0
    return FIRST | SECOND | THIRD, 1, 0     # bases loaded: a run is forced in


# --------------------------------------------------------------------------
# Model 2: learned from data
# --------------------------------------------------------------------------

class EmpiricalTransitions:
    """
    Runner advancement probabilities learned from the TruMedia data.

    For every combination of
        (outcome, base state before, outs before)
    the model stores how often each result
        (base state after, runs scored, outs made)
    happened in the data, and draws from that distribution.

    Relation to the paper
    ---------------------
    The paper uses fixed rules for "obvious situations" (HR, BB, HBP, or no
    runners on base) and data-based probabilities for 23 "non-obvious
    situations" (Tables 3-4). For each one it ran an ANOVA to decide
    whether to keep separate probabilities by number of outs or pool them.
    It also adds throwing errors as a separate random step, and handles
    "batter reaches on error" inside GO.

    This model is simpler but covers more:
      * Every situation gets its own distribution, always split by outs.
        TODO: test whether outs matter and pool cells that don't, as the
        paper does with ANOVA.
      * Double plays, sac flies, fielder's choices, runners thrown out
        trying to take an extra base, reaching on errors and throwing
        errors on the play are all already in the data, so no extra step
        is needed. (Pickoff errors between pitches are not modeled.)
      * HR, BB and HBP always use the deterministic rules, like the paper.
      * Combinations seen fewer than `min_obs` times in the data fall back
        to `SimpleRules`.

    Like the paper, a combination that never happened in the data gets
    probability 0, e.g. the runner on 3rd holding while the runner from
    2nd is thrown out on a fly ball.

    Usage
    -----
        model = EmpiricalTransitions.fit(pa)
        model.describe('1B', 'R2', outs=2)     # look at one situation
        new_bases, runs, outs_added = model.advance(SINGLE, 2, 2, rng)
    """

    name = 'Empirical transitions (TruMedia)'

    # Outcomes whose result is always the same; handled by the rules.
    DETERMINISTIC = (HOME_RUN, WALK, HIT_BY_PITCH)

    def __init__(self, table, min_obs=20):
        """
        Usually built with `EmpiricalTransitions.fit(pa)`, not directly.

        Parameters
        ----------
        table : dict
            {(outcome, bases, outs): (results, cumulative, n)} where
            `results` is a list of (new_bases, runs, outs_added) tuples,
            `cumulative` the matching cumulative probabilities, and `n` the
            number of PAs the cell is based on. Cells with fewer than
            `min_obs` PAs should already be removed.
        min_obs : int
            The threshold used to build `table` (kept for reference).
        """
        self.table = table
        self.min_obs = min_obs
        self.fallback = SimpleRules()

    @classmethod
    def fit(cls, pa: pd.DataFrame, min_obs=20):
        """
        Learn transition probabilities from the PA table.

        Parameters
        ----------
        pa : pd.DataFrame
            The PA table from `source.dataloaders.load_pa_table`. Only rows
            with `transition_ok == True` are used (see
            `source.dataloaders.build_pa_table` for that check).
        min_obs : int, optional
            Smallest number of PAs a situation needs before its observed
            distribution is trusted. Below this, `SimpleRules` is used.

        Returns
        -------
        EmpiricalTransitions
        """
        usable = pa[pa['transition_ok']].copy()
        usable['outcome_idx'] = usable['outcome'].map(OUTCOME_INDEX)

        keys = ['outcome_idx', 'bases_before', 'outs_before']
        results = ['bases_after', 'runs', 'outs_on_play']
        counts = usable.groupby(keys + results).size()

        cells = defaultdict(list)
        for (o, b, outs, b_after, runs, outs_added), n in counts.items():
            cells[(int(o), int(b), int(outs))].append(((int(b_after), int(runs), int(outs_added)), int(n)))

        table = {}
        for key, rows in cells.items():
            total = sum(n for _, n in rows)
            if total < min_obs:
                continue
            # Most common results first; makes `describe` output tidier.
            rows.sort(key=lambda r: -r[1])
            cumulative, running = [], 0
            for _, n in rows:
                running += n
                cumulative.append(running / total)
            cumulative[-1] = 1.0
            table[key] = ([r for r, _ in rows], cumulative, total)

        return cls(table, min_obs=min_obs)

    def advance(self, outcome, bases, outs, rng):
        """Draw a result for this situation. See the module docstring."""
        if outcome in self.DETERMINISTIC:
            return self.fallback.advance(outcome, bases, outs)

        cell = self.table.get((outcome, bases, outs))
        if cell is None:
            return self.fallback.advance(outcome, bases, outs)

        results, cumulative, _ = cell
        return results[bisect.bisect_right(cumulative, rng.random())]

    def describe(self, outcome, bases, outs) -> pd.DataFrame:
        """
        Show the result distribution for one situation, like the paper's
        supplementary Tables 13-19.

        Parameters
        ----------
        outcome : str or int
            e.g. '1B' or 0.
        bases : str or int
            e.g. 'R2', 'R1-R3', '13', or the 3-bit int.
        outs : int
            0, 1 or 2.

        Returns
        -------
        pd.DataFrame
            One row per possible result with columns 'bases_after',
            'runs', 'outs_added' and 'prob'. `df.attrs['n']` holds the
            number of PAs the cell is based on (None for rule-based).
        """
        o = OUTCOME_INDEX[outcome] if isinstance(outcome, str) else outcome
        b = str_to_bases(bases) if isinstance(bases, str) else bases

        cell = None if o in self.DETERMINISTIC else self.table.get((o, b, outs))
        if cell is None:
            results = [self.fallback.advance(o, b, outs)]
            probs, n = [1.0], None
        else:
            results, cumulative, n = cell
            probs = [c - p for c, p in zip(cumulative, [0.0] + cumulative[:-1])]

        df = pd.DataFrame({
            'bases_after': [bases_to_str(r[0]) if outs + r[2] < 3 else 'inning over' for r in results],
            'runs': [r[1] for r in results],
            'outs_added': [r[2] for r in results],
            'prob': probs,
        })
        df.attrs['n'] = n
        df.attrs['situation'] = f"{OUTCOMES[o]}, {bases_to_str(b)}, {outs} out"
        return df

    def coverage(self) -> pd.DataFrame:
        """
        How many PAs back each situation; missing ones use `SimpleRules`.

        Returns
        -------
        pd.DataFrame
            Rows: (outcome, bases) pairs. Columns: 0, 1, 2 outs. Values: the
            number of PAs, or 0 if the cell fell back to the simple rules.
        """
        rows = []
        for o in range(len(OUTCOMES)):
            if o in self.DETERMINISTIC:
                continue
            for b in range(8):
                rows.append({
                    'outcome': OUTCOMES[o],
                    'bases': bases_to_str(b),
                    **{outs: self.table.get((o, b, outs), (None, None, 0))[2] for outs in range(3)},
                })
        return pd.DataFrame(rows).set_index(['outcome', 'bases'])
