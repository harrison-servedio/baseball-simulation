"""
Players for the simulator.

A `Player` is just a name plus a probability vector over the 9 outcomes
in `source.outcomes.OUTCOMES`. Every time the player bats, the simulator
draws one outcome from that vector.

There are three ways to make one:

    from source.dataloaders import load_pa_table
    from source.players import Player

    pa = load_pa_table()

    # 1. A league-average batter (all PAs in SEC/ACC vs SEC/ACC games pooled)
    avg = Player.league_average(pa)

    # 2. A real batter from the data, by name or by BatterId
    guy = Player.from_data(pa, name='Aaron Piasecki')
    guy = Player.from_data(pa, player_id='1750930555')

    # 3. A made-up batter. Give any of the 9 rates; the ones you leave out
    #    are filled in from `baseline` (here, league average), scaled so
    #    the total is 1.
    slugger = Player.custom('Slugger', baseline=avg, HR=0.08, SO=0.25)

Pitchers are not modeled yet: every batter is assumed to face a
league-average pitcher (see `source/outcomes.py`).
"""

import difflib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from source.outcomes import (OUTCOMES, OUTCOME_INDEX, LEAGUE_CONFERENCES, outcome_counts,
                             league_probabilities, league_subset)


@dataclass
class Player:
    """
    A batter, described by his probability of each of the 9 outcomes.

    Attributes
    ----------
    name : str
        Display name.
    probs : np.ndarray, shape (9,)
        Probability of each outcome in `OUTCOMES` order
        ('1B', '2B', '3B', 'HR', 'BB', 'HBP', 'SO', 'GO', 'FO').
        Must be non-negative and sum to 1 (checked on creation).
    player_id : str or None
        TruMedia 'BatterId' when the player comes from the data.
    n_pa : float or None
        Number of PAs the rates are based on, when known.

    Examples
    --------
    Usually you'll use one of the class methods below, but you can also
    pass a full vector directly:

    >>> Player('Test', [0.15, 0.05, 0.005, 0.025, 0.1, 0.03, 0.2, 0.22, 0.22])
    """
    name: str
    probs: np.ndarray
    player_id: str = None
    n_pa: float = None
    # Cumulative probabilities, used by the simulator to draw an outcome
    # quickly. Filled in automatically; not part of the constructor.
    cumulative: list = field(init=False, repr=False)

    def __post_init__(self):
        probs = np.asarray(self.probs, dtype=float)
        if probs.shape != (len(OUTCOMES),):
            raise ValueError(f"probs must have {len(OUTCOMES)} entries in the order {OUTCOMES}, "
                             f"got shape {probs.shape}")
        if (probs < 0).any():
            raise ValueError(f"probs can't be negative: {dict(zip(OUTCOMES, probs))}")
        if not np.isclose(probs.sum(), 1.0, atol=1e-6):
            raise ValueError(f"probs must sum to 1, got {probs.sum():.6f}")
        self.probs = probs / probs.sum()

        # Plain Python list: `bisect` on a list is faster than numpy for
        # 9 elements. The last entry is forced to exactly 1.0 so that
        # rounding can never leave a random number with nowhere to go.
        cum = np.cumsum(self.probs).tolist()
        cum[-1] = 1.0
        self.cumulative = cum

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def league_average(cls, pa: pd.DataFrame, name='League Average', conferences=LEAGUE_CONFERENCES):
        """
        A batter whose rates equal the league's.

        Parameters
        ----------
        pa : pd.DataFrame
            The PA table from `source.dataloaders.load_pa_table`.
        name : str, optional
            Display name.
        conferences : iterable of str or None, optional
            Which conferences make up the league. Defaults to SEC + ACC
            (games where both teams are in one of them). `None` uses all
            of D1. See `source.outcomes.league_subset`.
        """
        counts = outcome_counts(league_subset(pa, conferences), by=None).iloc[0]
        return cls(name=name, probs=league_probabilities(pa, conferences), n_pa=counts['PA'])

    @classmethod
    def from_data(cls, pa: pd.DataFrame, name=None, player_id=None, prior_pa=0):
        """
        A real batter, using his observed rates in the data.

        Give either `name` or `player_id`.

        Parameters
        ----------
        pa : pd.DataFrame
            The PA table.
        name : str, optional
            Batter name as it appears in the data, e.g. 'Aaron Piasecki'.
            Not case-sensitive. If two different batters share the name,
            an error lists their ids and teams so you can pick one with
            `player_id`.
        player_id : str or int, optional
            TruMedia 'BatterId'.
        prior_pa : float, optional
            Shrink the batter's rates toward league average by adding this
            many league-average PAs to his record. 0 (default) uses his raw
            rates. See `source.outcomes.batter_probabilities` for details.

        Raises
        ------
        ValueError
            If the batter can't be found or the name is ambiguous.
        """
        if (name is None) == (player_id is None):
            raise ValueError("Give exactly one of `name` or `player_id`")

        if player_id is not None:
            player_id = str(player_id)
            rows = pa[pa['batter_id'] == player_id]
            if rows.empty:
                raise ValueError(f"No batter with id {player_id!r}")
        else:
            rows = pa[pa['batter'].str.lower() == name.lower()]
            if rows.empty:
                close = difflib.get_close_matches(name, pa['batter'].dropna().unique(), n=5)
                hint = f" Did you mean one of {close}?" if close else ""
                raise ValueError(f"No batter named {name!r}.{hint}")
            ids = rows['batter_id'].unique()
            if len(ids) > 1:
                options = (rows.groupby('batter_id')
                           .agg(team=('team', 'first'), PA=('team', 'size')))
                raise ValueError(
                    f"{len(ids)} different batters are named {name!r}. "
                    f"Use player_id= with one of:\n{options}")
            player_id = ids[0]

        counts = outcome_counts(rows, by=None).iloc[0]
        observed = counts[list(OUTCOMES)].to_numpy()
        smoothed = observed + prior_pa * league_probabilities(pa)

        return cls(
            name=rows['batter'].mode().iloc[0],
            probs=smoothed / smoothed.sum(),
            player_id=player_id,
            n_pa=counts['PA'],
        )

    @classmethod
    def custom(cls, name, rates=None, baseline=None, normalize=False, **kwargs):
        """
        A made-up batter with whatever rates you want.

        Rates can be passed as a dict, as keyword arguments, or both:

            Player.custom('A', rates={'HR': 0.08}, baseline=avg)
            Player.custom('A', baseline=avg, HR=0.08, SO=0.25)

        Keyword names are the outcome labels: 1B and 2B/3B aren't valid
        Python names, so use the dict form for those, or the aliases
        `single`, `double`, `triple`.

        Parameters
        ----------
        name : str
            Display name.
        rates : dict, optional
            Maps outcome label (e.g. 'HR') to probability.
        baseline : Player or array-like, optional
            Fills in the outcomes you don't set. The leftover probability
            (1 - sum of the rates you set) is shared among the unset
            outcomes in proportion to the baseline's rates for them.
            Example: baseline is league average and you set only HR=0.08;
            the other 8 outcomes keep their league-average *relative*
            sizes and are scaled down to make room for the extra homers.
            Required unless you set all 9 outcomes.
        normalize : bool, optional
            Only used when you set all 9 outcomes. If True, rescale them to
            sum to 1; if False (default), they must already sum to 1.

        Returns
        -------
        Player
        """
        aliases = {'single': '1B', 'double': '2B', 'triple': '3B',
                   'home_run': 'HR', 'walk': 'BB', 'hbp': 'HBP',
                   'strikeout': 'SO', 'ground_out': 'GO', 'fly_out': 'FO'}
        given = dict(rates or {})
        for key, value in kwargs.items():
            given[aliases.get(key, key)] = value

        unknown = set(given) - set(OUTCOMES)
        if unknown:
            raise ValueError(f"Unknown outcome(s) {sorted(unknown)}; use {OUTCOMES}")

        probs = np.zeros(len(OUTCOMES))
        for label, value in given.items():
            probs[OUTCOME_INDEX[label]] = value
        is_set = np.array([label in given for label in OUTCOMES])

        if is_set.all():
            if normalize:
                probs = probs / probs.sum()
            return cls(name=name, probs=probs)

        if baseline is None:
            missing = [o for o, s in zip(OUTCOMES, is_set) if not s]
            raise ValueError(f"Rates for {missing} are missing. Set them, or pass "
                             f"`baseline=` (e.g. Player.league_average(pa)) to fill them in.")

        base = baseline.probs if isinstance(baseline, Player) else np.asarray(baseline, dtype=float)
        leftover = 1.0 - probs[is_set].sum()
        if leftover < 0:
            raise ValueError(f"The rates you set add up to {1 - leftover:.4f}, which is more than 1")

        unset_base = base[~is_set]
        if unset_base.sum() == 0:
            raise ValueError("The baseline has zero probability for every outcome you left unset")
        probs[~is_set] = leftover * unset_base / unset_base.sum()

        return cls(name=name, probs=probs)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def as_series(self) -> pd.Series:
        """The 9 probabilities as a labeled pandas Series."""
        return pd.Series(self.probs, index=OUTCOMES, name=self.name)

    def __repr__(self):
        rates = ' '.join(f"{o}={p:.3f}" for o, p in zip(OUTCOMES, self.probs))
        extra = f", PA={self.n_pa:.0f}" if self.n_pa is not None else ""
        return f"Player({self.name!r}{extra}: {rates})"
