"""
Data loading and preparation.

This module turns the raw, pitch-level TruMedia feather file into a
plate-appearance (PA) level table that the rest of the simulator uses.

Pipeline
--------
    TruMedia2026.feather  (1 row per pitch, ~2.5M rows, ~170 columns)
            |
            |  load_trumedia()      -> read only the columns we need, drop
            |                          true duplicates, convert booleans
            v
    pitch table           (1 row per pitch, ~22 columns)
            |
            |  build_pa_table()     -> keep the pitch that ended each PA,
            |                          classify it into one of the paper's
            |                          9 outcomes, and attach the base/out
            |                          state *after* the PA
            v
    PA table              (1 row per PA, ~680k rows)

`load_pa_table()` wraps both steps and caches the result to disk, because
re-reading and re-processing the 1.1 GB feather file every run is wasteful.

How this relates to the paper (Beaudoin 2013)
---------------------------------------------
The paper needs two things from data:

1. Each batter's outcome probabilities p_i^(b) for the 9 outcomes
   (Section 2). These are computed in `source/outcomes.py` from the
   `outcome` column built here.
2. Runner-advancement probabilities: where runners end up given the
   outcome, the runners on base, and the number of outs (Section 2,
   Tables 3-4). The paper hand-collected these from 18,000 plays. Here
   they are read from the data: `bases_before`/`outs_before` give the
   state at the PA-ending pitch, and `bases_after` is the state at the
   first pitch of the *next* PA in the same half-inning. See
   `source/baserunning.py` for how the table is used.

Base-state encoding
-------------------
Throughout the project, the runners on base are stored as one integer
(a 3-bit mask):

    bit 0 (value 1) -> runner on 1st
    bit 1 (value 2) -> runner on 2nd
    bit 2 (value 4) -> runner on 3rd

e.g. 0 = bases empty, 3 = runners on 1st and 2nd, 7 = bases loaded.
`source/baserunning.py` has helpers to convert this to readable strings.

See Docs/trumedia_data.md for notes on the raw file itself.
"""

import os

import numpy as np
import pandas as pd

from source.outcomes import OUTCOMES, BIP_UNKNOWN


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# The raw file has ~170 columns, but the simulator needs only these.
# Reading a subset is much faster and uses far less memory.
TRUMEDIA_COLUMNS = [
    # --- identify the game / half-inning / PA / pitch ---
    'TruMediaGameID',   # unique game id
    'Inning',           # inning number (1, 2, ...)
    'Top.Bottom',       # 'Top' or 'Bottom' half of the inning
    'PAofInning',       # 1-indexed PA number within the half-inning
    'PitchofPA',        # 1-indexed pitch number within the PA
    'Date',             # game date, e.g. '3/6/26'
    # --- who is batting ---
    'Batter',           # batter name, 'First Last'
    'BatterId',         # batter id (use this, names are not unique)
    'BatterTeamName',   # batter's school
    'PitcherTeamName',  # pitcher's school
    # --- teams / conferences in the game (to tell which team is in which conference) ---
    'HomeTeamName',
    'AwayTeamName',
    'HomeConference',   # e.g. 'D1 Southeastern Conference'
    'AwayConference',
    # --- game state at the time of the pitch ---
    'Outs',             # outs before the pitch (0, 1, 2)
    'ManOnFirst',       # 'true' / 'false'
    'ManOnSecond',      # 'true' / 'false'
    'ManOnThird',       # 'true' / 'false'
    # --- what happened on the pitch ---
    'isPA',             # 1.0 if this pitch ended the PA, else 0.0
    'PlayResult',       # e.g. 'Single', 'Out', 'Strikeout (K)', 'Ball'
    'HitType',          # 'GroundBall', 'FlyBall', 'LineDrive', 'Popup', 'Undefined'
    'KorBB',            # 'Strikeout', 'Walk', 'Undefined'
    'OutsOnPlay',       # outs recorded on this pitch
    'RunsScored',       # runs that scored on this pitch
]

# The columns that uniquely identify one pitch.
#
# NOTE: Do *not* deduplicate on 'PitchUID'. About 544k rows (~21%) have a
# missing PitchUID, and `drop_duplicates('PitchUID')` treats all those
# NaNs as equal, which would collapse them into a single row.
PITCH_KEY = ['TruMediaGameID', 'Inning', 'Top.Bottom', 'PAofInning', 'PitchofPA']

# The columns that identify one half-inning.
HALF_INNING_KEY = ['TruMediaGameID', 'Inning', 'Top.Bottom']

# Columns of the PA table made by `build_pa_table` (documented there).
# `load_pa_table` rebuilds its cache if any of these are missing.
PA_TABLE_COLUMNS = [
    'game_id', 'date', 'inning', 'half', 'pa_of_inning',
    'batter_id', 'batter', 'team', 'batter_conference', 'pitcher_team', 'pitcher_conference',
    'outs_before', 'bases_before', 'outcome', 'runs', 'outs_on_play',
    'inning_over', 'bases_after', 'transition_ok',
]

# PlayResult values on PA-ending pitches -> paper outcome.
# Balls in play that are outs (and errors / fielder's choices / sacrifices)
# are handled separately below because they depend on HitType.
_DIRECT_OUTCOMES = {
    'Single': '1B',
    'Double': '2B',
    'Triple': '3B',
    'HomeRun': 'HR',
    'Walk': 'BB',
    'Intentional Walk': 'BB',
    'HitByPitch': 'HBP',
    'Strikeout (K)': 'SO',      # swinging strikeout
    'Strikeout (ꓘ)': 'SO',      # looking strikeout (backwards K)
    'Strikeout': 'SO',
}

# Balls put in play where the batter is (usually) out. The paper groups
# all of these into GO or FO based on the type of batted ball:
#   - 'Error': the paper puts reaching on an error inside GO; the runner
#     model then decides that the batter reached base.
#   - 'FieldersChoice': batter reaches, a runner is (usually) put out.
#   - 'Sacrifice': sac bunts (ground balls) and sac flies (fly balls).
_BALL_IN_PLAY_RESULTS = ['Out', 'FieldersChoice', 'Error', 'Sacrifice']

# Batted-ball types that count as a "fly out" in the paper's sense.
_AIR_BALLS = ['FlyBall', 'LineDrive', 'Popup']


# --------------------------------------------------------------------------
# Step 1: pitch table
# --------------------------------------------------------------------------

def load_trumedia(path, columns=TRUMEDIA_COLUMNS, verbose=False) -> pd.DataFrame:
    """
    Read the raw TruMedia feather file and lightly clean it.

    Parameters
    ----------
    path : str
        Path to the feather file (e.g. 'TruMedia2026.feather').
    columns : list of str, optional
        Columns to read. Defaults to `TRUMEDIA_COLUMNS`, which is all the
        simulator needs. Pass `None` to read every column (slow, ~1 GB+).
    verbose : bool, optional
        If True, print how many rows were read and dropped.

    Returns
    -------
    pd.DataFrame
        One row per pitch, with:
          - true duplicate pitches removed (see `PITCH_KEY`)
          - 'ManOnFirst' / 'ManOnSecond' / 'ManOnThird' converted from the
            strings 'true'/'false' to real booleans
          - a 'bases' column holding the 3-bit base-state integer
          - rows sorted in game order
    """
    df = pd.read_feather(path, columns=columns)
    n_raw = len(df)

    # Remove pitches that appear twice (only a handful in the 2026 file).
    df = df.drop_duplicates(PITCH_KEY)
    if verbose:
        print(f"Read {n_raw:,} pitches, dropped {n_raw - len(df):,} duplicates")

    # The base-runner columns are stored as the strings 'true'/'false'.
    for col in ['ManOnFirst', 'ManOnSecond', 'ManOnThird']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().eq('true')

    if {'ManOnFirst', 'ManOnSecond', 'ManOnThird'} <= set(df.columns):
        df['bases'] = (
            df['ManOnFirst'].astype(int) * 1
            + df['ManOnSecond'].astype(int) * 2
            + df['ManOnThird'].astype(int) * 4
        )

    # Put the pitches in the order they happened.
    if set(PITCH_KEY) <= set(df.columns):
        df = df.sort_values(PITCH_KEY, kind='stable').reset_index(drop=True)

    return df


# --------------------------------------------------------------------------
# Step 2: PA table
# --------------------------------------------------------------------------

def classify_outcome(play_result: pd.Series, hit_type: pd.Series) -> pd.Series:
    """
    Map each PA-ending pitch to one of the paper's 9 outcomes.

    Parameters
    ----------
    play_result : pd.Series
        The 'PlayResult' column for PA-ending pitches.
    hit_type : pd.Series
        The 'HitType' column for the same pitches.

    Returns
    -------
    pd.Series of str (or NaN)
        One of '1B', '2B', '3B', 'HR', 'BB', 'HBP', 'SO', 'GO', 'FO', or:
          - `BIP_UNKNOWN` ('BIP_UNK') for a ball in play that was an out
            (or error / FC / sacrifice) but whose HitType is 'Undefined',
            so we cannot tell if it was a ground ball or a fly ball. About
            11% of PAs fall here. `source/outcomes.py` splits them between
            GO and FO; they are not used to fit runner advancement.
          - NaN for PAs we drop: catcher/batter interference, 'NoPlay', and
            a few data errors where a PA-ending pitch has a result like
            'Ball' or ''. These are about 0.04% of PAs.

    Notes
    -----
    Mapping (follows Section 2 of the paper):
        Single -> 1B, Double -> 2B, Triple -> 3B, HomeRun -> HR
        Walk / Intentional Walk -> BB, HitByPitch -> HBP
        Strikeout (K) / Strikeout (ꓘ) -> SO
        Out / FieldersChoice / Error / Sacrifice:
            HitType == GroundBall               -> GO
            HitType in FlyBall/LineDrive/Popup  -> FO
            HitType == Undefined                -> BIP_UNK

    The paper puts line drives and pop-ups in with fly outs because it has
    only one "air" out type. Errors on fly balls are put in FO here (the
    paper only discusses errors on ground balls).
    """
    play_result = play_result.astype(str).str.strip()
    hit_type = hit_type.astype(str).str.strip()

    outcome = play_result.map(_DIRECT_OUTCOMES)

    in_play = play_result.isin(_BALL_IN_PLAY_RESULTS)
    outcome = outcome.mask(in_play & (hit_type == 'GroundBall'), 'GO')
    outcome = outcome.mask(in_play & hit_type.isin(_AIR_BALLS), 'FO')
    outcome = outcome.mask(in_play & ~hit_type.isin(['GroundBall'] + _AIR_BALLS), BIP_UNKNOWN)

    return outcome


def build_pa_table(pitches: pd.DataFrame, verbose=False) -> pd.DataFrame:
    """
    Collapse the pitch table into one row per plate appearance.

    Parameters
    ----------
    pitches : pd.DataFrame
        Output of `load_trumedia()`.
    verbose : bool, optional
        If True, print a short summary of the data quality checks.

    Returns
    -------
    pd.DataFrame
        One row per PA with these columns:

        game_id, date, inning, half
            Where the PA happened.
        pa_of_inning
            1-indexed PA number within the half-inning.
        batter_id, batter, team
            Who batted.
        batter_conference, pitcher_team, pitcher_conference
            Conference of the batting team, and the pitching team and its
            conference, e.g. 'D1 Southeastern Conference'. Used to pick
            which PAs define "league average" (see
            `source.outcomes.league_subset`).
        outs_before : int
            Outs when the PA-ending pitch was thrown (0, 1 or 2).
        bases_before : int
            Base state (3-bit int) when the PA-ending pitch was thrown.
            Using the *last* pitch (not the first) means a runner who stole
            a base during the PA is already counted at his new base.
        outcome : str
            One of the 9 outcomes, 'BIP_UNK', or NaN (see `classify_outcome`).
        runs : int
            Runs that scored on the PA-ending pitch.
        outs_on_play : int
            Outs recorded on the PA-ending pitch.
        inning_over : bool
            True if the PA made the 3rd out.
        bases_after : int
            Base state after the PA, read from the first pitch of the next
            PA in the same half-inning. -1 if the inning ended. -2 if it is
            unknown (no next PA but fewer than 3 outs, e.g. a walk-off).
        transition_ok : bool
            True if this PA can be trusted for runner advancement. It
            requires a known outcome and that the numbers add up:
              * when the inning continues:
                    runners before + batter
                        == runners after + runs + outs on play
                and outs before + outs on play == outs at next PA
              * when the inning ends:
                    outs before + outs on play == 3
                and runs <= runners on base before the play
            Roughly 98% of PAs pass. The ones that fail are mostly caused
            by things that happen between PAs (pickoffs, steals with no
            pitch recorded, substitutions) or by tagging errors.
    """
    # The pitch that ended each PA. There can very rarely be two pitches
    # flagged as PA-ending in the same PA; keep the last.
    pa_key = HALF_INNING_KEY + ['PAofInning']
    pa = pitches[pitches['isPA'] == 1].drop_duplicates(pa_key, keep='last').copy()

    # The state at the *first* pitch of each PA. Shifting PAofInning down by
    # one lets us join "first pitch of PA k+1" onto "PA k".
    first = (
        pitches.sort_values(PITCH_KEY)
        .drop_duplicates(pa_key, keep='first')
        [pa_key + ['bases', 'Outs']]
        .rename(columns={'bases': 'next_bases', 'Outs': 'next_outs'})
    )
    first['PAofInning'] = first['PAofInning'] - 1
    pa = pa.merge(first, on=pa_key, how='left')

    out = pd.DataFrame({
        'game_id': pa['TruMediaGameID'],
        'date': pa['Date'],
        'inning': pa['Inning'].astype(int),
        'half': pa['Top.Bottom'],
        'pa_of_inning': pa['PAofInning'].astype(int),
        'batter_id': pa['BatterId'].astype(str),
        'batter': pa['Batter'],
        'team': pa['BatterTeamName'],
        # The file stores conferences per home/away team, so look up each
        # side's conference by matching its team name to the home team.
        'batter_conference': pa['HomeConference'].where(
            pa['BatterTeamName'] == pa['HomeTeamName'], pa['AwayConference']),
        'pitcher_team': pa['PitcherTeamName'],
        'pitcher_conference': pa['HomeConference'].where(
            pa['PitcherTeamName'] == pa['HomeTeamName'], pa['AwayConference']),
        'outs_before': pa['Outs'].fillna(0).astype(int),
        'bases_before': pa['bases'].astype(int),
        'outcome': classify_outcome(pa['PlayResult'], pa['HitType']),
        'runs': pa['RunsScored'].fillna(0).astype(int),
        'outs_on_play': pa['OutsOnPlay'].fillna(0).astype(int),
    })

    outs_after = out['outs_before'] + out['outs_on_play']
    has_next = pa['next_outs'].notna()
    out['inning_over'] = outs_after >= 3

    # bases_after: -1 = inning over, -2 = unknown
    bases_after = np.where(out['inning_over'], -1, -2)
    bases_after = np.where(~out['inning_over'] & has_next, pa['next_bases'].fillna(-2), bases_after)
    out['bases_after'] = bases_after.astype(int)

    # Data-quality check (see docstring).
    runners_before = _count_runners(out['bases_before'])
    runners_after = _count_runners(out['bases_after'].clip(lower=0))
    continues_ok = (
        ~out['inning_over'] & has_next
        & (runners_before + 1 == runners_after + out['runs'] + out['outs_on_play'])
        & (outs_after == pa['next_outs'])
    )
    ends_ok = (outs_after == 3) & (out['runs'] <= runners_before)
    known = out['outcome'].isin(OUTCOMES)
    out['transition_ok'] = known & (continues_ok | ends_ok)

    if verbose:
        n = len(out)
        print(f"PAs: {n:,}")
        print(f"  dropped (unclassifiable outcome): {out['outcome'].isna().sum():,}")
        print(f"  ball in play with unknown hit type: {(out['outcome'] == BIP_UNKNOWN).sum():,}")
        print(f"  usable for runner advancement: {out['transition_ok'].sum():,} "
              f"({out['transition_ok'].mean():.1%})")

    return out.reset_index(drop=True)


def _count_runners(bases) -> pd.Series:
    """Number of runners on base for a Series of 3-bit base-state ints."""
    b = pd.Series(bases).astype(int)
    return (b & 1) + ((b // 2) & 1) + ((b // 4) & 1)


# --------------------------------------------------------------------------
# Convenience wrapper with caching
# --------------------------------------------------------------------------

def load_pa_table(path='TruMedia2026.feather', cache_path='pa_table.feather',
                  rebuild=False, verbose=True) -> pd.DataFrame:
    """
    Load the PA table, building it from the raw file only when needed.

    The first call reads the large raw file
    and saves the much smaller PA table to `cache_path`. Later calls just
    read the cache.

    Parameters
    ----------
    path : str
        Path to the raw TruMedia feather file.
    cache_path : str or None
        Where to store the PA table. `None` disables caching.
        (*.feather files are gitignored, so the cache stays out of git.)
    rebuild : bool
        Force a rebuild from the raw file even if the cache exists. Do this
        after changing how `build_pa_table` or `classify_outcome` compute
        values. (A cache missing columns that the code now expects, e.g.
        one made by an older version, is rebuilt automatically.)
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        See `build_pa_table` for the columns.
    """
    if cache_path and os.path.exists(cache_path) and not rebuild:
        pa = pd.read_feather(cache_path)
        missing = set(PA_TABLE_COLUMNS) - set(pa.columns)
        if not missing:
            if verbose:
                print(f"Loading cached PA table from {cache_path}")
            return pa
        if verbose:
            print(f"Cached PA table is out of date (missing {sorted(missing)}), rebuilding")

    if verbose:
        print(f"Building PA table from {path}...")
    pitches = load_trumedia(path, verbose=verbose)
    pa = build_pa_table(pitches, verbose=verbose)

    if cache_path:
        pa.to_feather(cache_path)
        if verbose:
            print(f"Saved PA table to {cache_path}")
    return pa


def observed_runs_per_nine(pa: pd.DataFrame) -> float:
    """
    Actual runs scored per 9 innings in the data, for checking the simulator.

    Only half-innings that reached 3 outs are used, so walk-off and
    unfinished innings don't drag the number down. Only runs that scored on
    the PA-ending pitch are counted (no wild pitches, steals of home, etc.),
    which matches what the simulator can produce.

    Parameters
    ----------
    pa : pd.DataFrame
        Output of `build_pa_table` / `load_pa_table`.

    Returns
    -------
    float
        Average runs per half-inning, times 9.
    """
    halves = pa.groupby(['game_id', 'inning', 'half'])
    outs = halves['outs_on_play'].sum()
    runs = halves['runs'].sum()
    return 9 * runs[outs >= 3].mean()


# --------------------------------------------------------------------------
# Duke data (older file, kept for reference)
# --------------------------------------------------------------------------

def load_dukedata(path, verbose=False) -> pd.DataFrame:
    """
    Load the older Duke Trackman file and remove duplicate pitches.

    Rows with a missing 'PitchUID' are kept, since `drop_duplicates` would
    otherwise treat every missing id as the same pitch.
    """
    df = pd.read_feather(path)
    entries = len(df)

    has_uid = df['PitchUID'].notna()
    df = pd.concat([df[has_uid].drop_duplicates('PitchUID'), df[~has_uid]])

    if verbose:
        print(f"Dropped {entries - len(df)} duplicates")

    return df
