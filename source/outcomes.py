"""
Batter outcome probabilities: the paper's p_i^(b).

Section 2 of Beaudoin (2013) treats every PA as one draw from a
multinomial distribution over 9 outcomes:

    index  outcome  meaning
    -----  -------  ---------------------------------------------
      0     1B      single
      1     2B      double
      2     3B      triple
      3     HR      home run
      4     BB      walk (base on balls), incl. intentional walks
      5     HBP     hit by pitch
      6     SO      strikeout
      7     GO      ground out (incl. reached on error, fielder's
                    choice and sac bunts on ground balls)
      8     FO      fly out (fly balls, line drives and pop-ups;
                    incl. sac flies and errors on air balls)

The first six are "positive outcomes" (the batter reaches base) and the
last three are "negative outcomes".

For a batter B, p_i^(b) is the probability of outcome i when B faces a
league-average pitcher. Like the paper, we estimate it with B's observed
frequency, since a full season of PAs comes against a wide mix of pitchers.

"League average" is defined by SEC and ACC games only (both teams in one
of those conferences); see `LEAGUE_CONFERENCES` and `league_subset`. Real
batters still use all of their own PAs, whatever the opponent.

This first version ignores the pitcher: every batter is assumed to face a
league-average pitcher, so p_i = p_i^(b).
TODO (paper Section 2): add the logistic-regression adjustment
    log(PPO / (1 - PPO)) = b0 + b1 * PPOB + b2 * PPOP
to combine batter and pitcher rates for a specific matchup.

Balls in play with an unknown hit type
--------------------------------------
About 11% of PAs in the TruMedia file are batted-ball outs (or errors /
FCs / sacrifices) whose HitType is 'Undefined'. They are labeled
`BIP_UNKNOWN` by `source.dataloaders.classify_outcome`. Dropping them would
make every batter look better than he is (fewer outs), so instead each
batter's unknowns are split between GO and FO in the same proportion as his
known GO:FO split. A batter with no known GO or FO uses the league split.
"""

import numpy as np
import pandas as pd


# The 9 outcomes, in the paper's order. Every probability vector in this
# project uses this order.
OUTCOMES = ('1B', '2B', '3B', 'HR', 'BB', 'HBP', 'SO', 'GO', 'FO')

# Index of each outcome in OUTCOMES, e.g. OUTCOME_INDEX['HR'] == 3.
OUTCOME_INDEX = {name: i for i, name in enumerate(OUTCOMES)}

# Named indices, handy in the simulation code.
SINGLE, DOUBLE, TRIPLE, HOME_RUN, WALK, HIT_BY_PITCH, STRIKEOUT, GROUND_OUT, FLY_OUT = range(9)

# Outcomes where the batter reaches base (the paper's "positive outcomes").
POSITIVE_OUTCOMES = OUTCOMES[:6]

# Label for a ball-in-play out whose type (ground vs. air) is unknown.
BIP_UNKNOWN = 'BIP_UNK'

# Conferences that define "league average". The file covers all of D1, but
# league average is taken from SEC and ACC games only (see `league_subset`).
LEAGUE_CONFERENCES = ('D1 Southeastern Conference', 'D1 Atlantic Coast Conference')


def league_subset(pa: pd.DataFrame, conferences=LEAGUE_CONFERENCES) -> pd.DataFrame:
    """
    The PAs that define "league average".

    Keeps only PAs where **both** the batting team and the pitching team
    are in `conferences`: SEC/ACC hitters facing SEC/ACC pitchers. This is
    the same set of PAs as the raw file's `isACCSEC == 1` flag (~44k PAs).
    Requiring both sides keeps the average from being inflated by SEC/ACC
    hitters' early-season games against weaker non-conference pitching,
    which matters because the simulator assumes every batter faces a
    league-average pitcher.

    Parameters
    ----------
    pa : pd.DataFrame
        The PA table from `source.dataloaders.load_pa_table`.
    conferences : iterable of str or None
        Conference names exactly as in the data's 'HomeConference' column,
        e.g. 'D1 Big 12 Conference'. `None` returns `pa` unchanged (all D1).

    Returns
    -------
    pd.DataFrame
        The filtered PA table.
    """
    if conferences is None:
        return pa
    conferences = list(conferences)
    keep = pa['batter_conference'].isin(conferences) & pa['pitcher_conference'].isin(conferences)
    if not keep.any():
        raise ValueError(f"No PAs where both teams are in {conferences}. "
                         f"Check the spelling against pa['batter_conference'].unique().")
    return pa[keep]


def outcome_counts(pa: pd.DataFrame, by='batter_id') -> pd.DataFrame:
    """
    Count each batter's PAs by outcome, with unknown balls in play split.

    Parameters
    ----------
    pa : pd.DataFrame
        The PA table from `source.dataloaders.load_pa_table`.
    by : str or None
        Column to group by. The default gives one row per batter. Use
        `None` to get a single row with league totals.

    Returns
    -------
    pd.DataFrame
        Columns are the 9 `OUTCOMES` plus 'PA' (their sum). Counts can be
        fractional because of the GO/FO split of unknown balls in play.
        PAs with no usable outcome (interference etc.) are not counted.
    """
    pa = pa[pa['outcome'].isin(OUTCOMES + (BIP_UNKNOWN,))]

    if by is None:
        counts = pa['outcome'].value_counts().to_frame().T
        counts.index = ['league']
    else:
        counts = pd.crosstab(pa[by], pa['outcome'])

    counts = counts.reindex(columns=list(OUTCOMES) + [BIP_UNKNOWN], fill_value=0).astype(float)

    # Share of known batted-ball outs that were on the ground, per row.
    known = counts['GO'] + counts['FO']
    league_go_share = counts['GO'].sum() / known.sum()
    go_share = (counts['GO'] / known.where(known > 0)).fillna(league_go_share)

    # Hand the unknowns out to GO and FO.
    counts['GO'] += counts[BIP_UNKNOWN] * go_share
    counts['FO'] += counts[BIP_UNKNOWN] * (1 - go_share)
    counts = counts.drop(columns=BIP_UNKNOWN)

    counts['PA'] = counts[list(OUTCOMES)].sum(axis=1)
    return counts


def league_probabilities(pa: pd.DataFrame, conferences=LEAGUE_CONFERENCES) -> np.ndarray:
    """
    Outcome probabilities for a league-average batter.

    This pools every PA in the "league" (by default, SEC/ACC teams facing
    each other; see `league_subset`), so it is also what a league-average
    batter does against a league-average pitcher.

    Parameters
    ----------
    pa : pd.DataFrame
        The PA table.
    conferences : iterable of str or None, optional
        Which conferences make up the league. Defaults to SEC + ACC.
        `None` uses every PA in the data (all of D1).

    Returns
    -------
    np.ndarray, shape (9,)
        Probabilities in `OUTCOMES` order, summing to 1.
    """
    counts = outcome_counts(league_subset(pa, conferences), by=None).iloc[0]
    probs = counts[list(OUTCOMES)].to_numpy()
    return probs / probs.sum()


def batter_probabilities(pa: pd.DataFrame, min_pa=0, prior_pa=0) -> pd.DataFrame:
    """
    Outcome probabilities p_i^(b) for every batter in the data.

    Parameters
    ----------
    pa : pd.DataFrame
        The PA table.
    min_pa : int, optional
        Only return batters with at least this many PAs. The paper only
        reports batters with 200+ at-bats (Section 3).
    prior_pa : float, optional
        Shrink each batter toward league average by adding `prior_pa`
        "imaginary" league-average PAs to his record:

            p_i = (count_i + prior_pa * league_p_i) / (PA + prior_pa)

        The paper warns (Section 2) that small samples give silly rates,
        e.g. "a batter with 3 home runs in only 10 at-bats". With
        prior_pa = 0 (default) you get the raw observed rates. Something
        like 50-200 keeps regulars close to their raw rates while pulling
        part-time players toward the middle. "League average" here means
        the SEC/ACC rates from `league_probabilities`.

    Returns
    -------
    pd.DataFrame
        Indexed by 'batter_id', with columns:
          - 'batter' (name) and 'team' (most common team for that id)
          - the 9 `OUTCOMES` (probabilities summing to 1 across a row)
          - 'PA' (actual number of PAs, not including `prior_pa`)
    """
    counts = outcome_counts(pa, by='batter_id')
    counts = counts[counts['PA'] >= min_pa]

    league = league_probabilities(pa)
    cols = list(OUTCOMES)
    smoothed = counts[cols].to_numpy() + prior_pa * league
    probs = smoothed / smoothed.sum(axis=1, keepdims=True)

    out = pd.DataFrame(probs, index=counts.index, columns=cols)
    out['PA'] = counts['PA']

    names = batter_names(pa)
    out = names.join(out, how='inner')
    return out


def batter_names(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Most common name and team for each batter id.

    Parameters
    ----------
    pa : pd.DataFrame
        The PA table.

    Returns
    -------
    pd.DataFrame
        Indexed by 'batter_id' with columns 'batter' and 'team'.
    """
    def most_common(col):
        # Count (id, value) pairs, then keep the most frequent value per id.
        sizes = pa.groupby(['batter_id', col]).size().sort_values(ascending=False)
        return sizes.reset_index().drop_duplicates('batter_id').set_index('batter_id')[col]

    return pd.DataFrame({'batter': most_common('batter'), 'team': most_common('team')})
