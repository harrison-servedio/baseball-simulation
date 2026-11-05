"""
Pitch-by-pitch half-inning simulator, for testing ABS challenge strategies.

PA-level vs. pitch-level
------------------------
The rest of the simulator is PA-level: `source.simulation.simulate_half_inning`
draws one outcome per PA and never sees individual pitches. That is all you
need for runs per game, NRGG or run expectancy (`source.challenges`).

A challenge *strategy* needs pitches. Whether to challenge depends on how
many challenges are left, and a challenge used now can't be used on a later
pitch in the inning. So this module plays out every pitch of every PA:

    module                      simulates          use it for
    -------------------------   ----------------   -------------------------------
    simulation, challenges      one draw per PA    runs per game, NRGG, RE tables
    challenge_sim (this one)    every pitch        challenge strategies

What a "pitch" is here: a ball, a called strike, another strike (swinging,
or a foul with fewer than 2 strikes), or the pitch that ends the PA (in
play or HBP). Fouls with 2 strikes don't change the count and are skipped.

How a PA is played out
----------------------
    1. The batter's outcome is drawn first, from his rates at the current
       count (`source.counts.at_count`), exactly as in the PA-level sim.
    2. Pitches are drawn one at a time from `source.counts.pitch_transitions`:
       what the next pitch does, given the count and that outcome.
    3. Every called pitch (ball or called strike) is a *close call* with the
       chance you set in `ChallengeRules.close_call_rate`. A close call can
       be challenged: the offense challenges called strikes and the defense
       challenges called balls. Your strategy function decides.
    4. If a challenge wins, the call is reversed. The count changes to the
       other branch and the outcome is drawn again from the batter's rates
       at the new count.

With no close calls this gives the same results as the PA-level
`simulate_half_inning`, just more slowly (about 3-4x).

A PA that ends on a close call (ball four, strike three, or a call
overturned into either) is played as a simple walk (forced runners move up)
or an out (runners hold), like the 'BB' / 'SO' counts in
`source.challenges`. Every other PA uses the runner model.

Usage
-----
    from source.challenge_sim import (PitchModel, ChallengeRules, PitchSituation,
                                      never_challenge, always_challenge,
                                      compare_strategies_by_pitch)

    pitch_model = PitchModel.fit(pa)
    rules = ChallengeRules(close_call_rate=0.05)

    def my_strategy(s: PitchSituation) -> bool:
        return s.confidence > 0.7 and s.outs < 2

    compare_strategies_by_pitch(avg, model, pitch_model, rules, {
        'never':  (never_challenge, never_challenge),
        'mine':   (my_strategy, never_challenge),
    }, n_innings=50_000, seed=1)

Not modeled: what a challenge kept for later innings is worth (challenges
left at the end are reported so you can weigh them yourself), and a
challenger who misjudges a call (`confidence` is exactly his chance of
winning).
"""

import bisect
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from source.baserunning import bases_to_str, str_to_bases
from source.challenges import _end_pa, challenge_table
from source.counts import (COUNTS, CALLS, PITCH_STEPS, WALK_COUNT, STRIKEOUT_COUNT, at_count,
                           count_after_call, count_probabilities, parse_count, pitch_transitions)
from source.outcomes import OUTCOMES, LEAGUE_CONFERENCES, WALK, STRIKEOUT
from source.players import Player
from source.simulation import MAX_PA_PER_HALF_INNING, SimulationResult, as_lineup


# --------------------------------------------------------------------------
# Pitch model
# --------------------------------------------------------------------------

class PitchModel:
    """
    Everything the pitch-by-pitch simulator needs to draw pitches.

    Holds the results matrix (`source.counts.count_probabilities`) and the
    pitch-level transitions (`source.counts.pitch_transitions`), in a form
    that is fast to draw from.

    Usage
    -----
        pitch_model = PitchModel.fit(pa)
        # or, from tables you already have:
        pitch_model = PitchModel(matrix, transitions)
    """

    def __init__(self, matrix, transitions):
        """
        Parameters
        ----------
        matrix : pd.DataFrame
            From `source.counts.count_probabilities`.
        transitions : pd.DataFrame
            From `source.counts.pitch_transitions`.
        """
        self.matrix = matrix
        self.transitions = transitions

        # steps[count_idx][outcome_idx] = cumulative chance of (ball,
        # called strike, other strike); anything above the last ends the PA.
        index = pd.MultiIndex.from_product([COUNTS, OUTCOMES])
        probs = transitions.loc[index, list(PITCH_STEPS)].to_numpy(dtype=float)
        cumulative = np.cumsum(probs.reshape(len(COUNTS), len(OUTCOMES), len(PITCH_STEPS)), axis=2)
        self.steps = cumulative[:, :, :3].tolist()

        self._players = {}

    @classmethod
    def fit(cls, pa: pd.DataFrame, conferences=LEAGUE_CONFERENCES):
        """
        Build both tables from the PA table.

        Parameters
        ----------
        pa : pd.DataFrame
            The PA table from `source.dataloaders.load_pa_table`.
        conferences : iterable of str or None, optional
            Which conferences make up the league (default SEC + ACC).
        """
        return cls(count_probabilities(pa, conferences), pitch_transitions(pa, conferences))

    def outcome_cumulative(self, player):
        """
        The player's cumulative outcome probabilities from each count on,
        as a list indexed like `COUNTS`. Cached per player.
        """
        cached = self._players.get(id(player))
        if cached is not None and cached[0] is player:
            return cached[1]
        table = [player.cumulative if count == '0-0' else at_count(player, self.matrix, count).cumulative
                 for count in COUNTS]
        self._players[id(player)] = (player, table)
        return table


# --------------------------------------------------------------------------
# What you set by hand, and what a strategy sees
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ChallengeRules:
    """
    The settings of a challenge experiment.

    It can't be changed after it's made. To try other settings, make a new
    one, or copy it with changes:
    `dataclasses.replace(rules, close_call_rate=0.1)`.

    Attributes
    ----------
    close_call_rate : float, dict or callable
        How often a *called* pitch (a ball, or a strike the batter took) is
        close enough to challenge. One of:
          - a float, used for every called pitch, e.g. 0.05
          - {'ball': x, 'strike': y}, separate rates for called balls and
            called strikes
          - a function `rate(count, call) -> float`, e.g. to make close
            calls more common at 3-2. `count` is the count before the pitch
            ('1-2'), `call` is 'ball' or 'strike'.
        Swinging strikes and fouls are never challengeable.
    overturn_prob : float, callable or None
        The chance that a close call is wrong, so a challenge of it wins.
        The challenger knows it: strategies see it as `confidence`.
          - None (default): drawn uniformly between 0 and 1 for each close
            call
          - a float: the same for every close call
          - a function `draw(rng) -> float`, using `rng` (a
            `random.Random`) for any randomness, e.g.
            `lambda rng: rng.betavariate(2, 2)`
    offense_challenges, defense_challenges : int
        Challenges each team has at the start of the half-inning (default 2).
    keep_on_success : bool
        If True (default, the MLB ABS rule), a challenge that wins isn't
        used up. Losing one always uses it up.
    """
    close_call_rate: object
    overturn_prob: object = None
    offense_challenges: int = 2
    defense_challenges: int = 2
    keep_on_success: bool = True
    # rates[count_idx][call] -> close-call rate. Filled in automatically.
    rates: list = field(init=False, repr=False)

    def __post_init__(self):
        rate = self.close_call_rate
        if callable(rate):
            get = rate
        elif isinstance(rate, dict):
            if set(rate) != set(CALLS):
                raise ValueError(f"close_call_rate dict needs exactly the keys {CALLS}, got {sorted(rate)}")
            get = lambda count, call: rate[call]
        else:
            get = lambda count, call: rate

        rates = []
        for count in COUNTS:
            row = {}
            for call in CALLS:
                value = float(get(count, call))
                if not 0 <= value <= 1:
                    raise ValueError(f"close_call_rate at {count}, {call} must be between 0 and 1, "
                                     f"got {value}")
                row[call] = value
            rates.append(row)
        object.__setattr__(self, 'rates', rates)    # the dataclass is frozen

        p = self.overturn_prob
        if p is not None and not callable(p) and not 0 <= p <= 1:
            raise ValueError(f"overturn_prob must be between 0 and 1, got {p}")

    def draw_confidence(self, rng):
        """The chance this close call is overturned, for one close call."""
        p = self.overturn_prob
        if p is None:
            return rng.random()
        if callable(p):
            value = float(p(rng))
            if not 0 <= value <= 1:
                raise ValueError(f"overturn_prob returned {value}; it must be between 0 and 1")
            return value
        return p


@dataclass(frozen=True)
class PitchSituation:
    """
    One close call, as a challenge strategy sees it.

    A strategy is any function `strategy(s: PitchSituation) -> bool` that
    returns True to challenge. It is only asked when the challenging team
    has a challenge left.

    Attributes
    ----------
    outs : int
        Outs when the pitch was thrown (0, 1, 2).
    bases : int
        Runners on base as a 3-bit int (see `base_state` for 'R1-R3' etc.).
    count : str
        Count when the pitch was thrown, e.g. '1-1' (before the call).
    call : str
        The umpire's call: 'strike' (the offense may challenge) or 'ball'
        (the defense may challenge).
    confidence : float
        The chance a challenge of this call wins (see
        `ChallengeRules.overturn_prob`).
    challenges_left : int
        Challenges the challenging team has left, including this one.
    opponent_challenges_left : int
        Challenges the other team has left.
    runs : int
        Runs the offense has scored so far this half-inning.
    batter : Player
        The batter at the plate.
    batter_idx : int
        His spot in the lineup (0-indexed).
    pa_of_inning : int
        Which PA of the half-inning this is (1 = first).
    pitch_of_pa : int
        Which pitch of the PA this is (1 = first; 2-strike fouls are not
        counted).

    Properties: `challenger` ('offense' / 'defense'), `base_state`,
    `count_if_stands`, `count_if_overturned` ('BB' / 'SO' if the call ends
    the PA), and `key`, which looks the call up in `challenge_gains`.
    """
    outs: int
    bases: int
    count: str
    call: str
    confidence: float
    challenges_left: int
    opponent_challenges_left: int
    runs: int
    batter: Player
    batter_idx: int
    pa_of_inning: int
    pitch_of_pa: int

    @property
    def challenger(self):
        return 'offense' if self.call == 'strike' else 'defense'

    @property
    def base_state(self):
        return bases_to_str(self.bases)

    @property
    def count_if_stands(self):
        return count_after_call(self.count, self.call)

    @property
    def count_if_overturned(self):
        return count_after_call(self.count, 'ball' if self.call == 'strike' else 'strike')

    @property
    def key(self):
        return (self.bases, self.outs, self.count, self.call)


@dataclass
class CloseCall:
    """
    What happened on one close call (from `simulate_half_inning_by_pitch`
    with `log=True`).

    Attributes
    ----------
    situation : PitchSituation
    challenged : bool
        Whether the challenging team challenged.
    call_was_wrong : bool
        Whether a challenge would have won (drawn whether or not anyone
        challenged).
    overturned : bool
        `challenged and call_was_wrong`.
    """
    situation: PitchSituation
    challenged: bool
    call_was_wrong: bool
    overturned: bool


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

def never_challenge(s: PitchSituation) -> bool:
    """Never challenge."""
    return False


def always_challenge(s: PitchSituation) -> bool:
    """Challenge every close call while challenges are left."""
    return True


def confidence_at_least(threshold):
    """
    A strategy that challenges when `s.confidence >= threshold`.

    >>> offense = confidence_at_least(0.6)
    """
    def strategy(s: PitchSituation) -> bool:
        return s.confidence >= threshold
    strategy.__name__ = f'confidence_at_least({threshold})'
    return strategy


def challenge_gains(re_table):
    """
    Run value of winning each challenge, for fast lookups in a strategy.

    Parameters
    ----------
    re_table : pd.DataFrame
        From `source.challenges.run_expectancy_table`.

    Returns
    -------
    dict
        {(bases, outs, count, call): runs gained by the challenging team if
        the call is overturned}. Look a pitch up with `gains[s.key]`. The
        values are `challenger_gain` from `source.challenges.challenge_table`
        with `p_overturn=1`, so multiply by `s.confidence` for the expected
        gain.
    """
    table = challenge_table(re_table, p_overturn=1.0)
    return {(row.bases, row.outs, row.count, row.call): row.challenger_gain
            for row in table.itertuples()}


def gain_at_least(gains, min_runs):
    """
    A strategy that challenges when the expected gain is big enough:
    `s.confidence * gains[s.key] >= min_runs`.

    `min_runs` is in effect the price of risking a challenge.

    Parameters
    ----------
    gains : dict
        From `challenge_gains`.
    min_runs : float
        Smallest expected gain, in runs, worth a challenge.
    """
    def strategy(s: PitchSituation) -> bool:
        return s.confidence * gains[s.key] >= min_runs
    strategy.__name__ = f'gain_at_least({min_runs})'
    return strategy


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

@dataclass
class HalfInningByPitch:
    """
    The result of one pitch-by-pitch half-inning.

    Attributes
    ----------
    runs : int
        Runs scored.
    next_batter_idx : int
        Who leads off the next inning.
    called_pitches : int
        Called balls and called strikes.
    close_calls : int
        Called pitches that were challengeable.
    offense_challenges, defense_challenges : int
        Challenges made by each team.
    offense_won, defense_won : int
        Challenges each team won.
    offense_left, defense_left : int
        Challenges each team had left at the end.
    close_call_log : list of CloseCall or None
        Every close call, in order (only with `log=True`).
    """
    runs: int
    next_batter_idx: int
    called_pitches: int = 0
    close_calls: int = 0
    offense_challenges: int = 0
    defense_challenges: int = 0
    offense_won: int = 0
    defense_won: int = 0
    offense_left: int = 0
    defense_left: int = 0
    close_call_log: list = None


def simulate_half_inning_by_pitch(lineup, runner_model, pitch_model, rules,
                                  offense=never_challenge, defense=never_challenge, rng=None,
                                  outs=0, bases=0, count='0-0', batter_idx=0, log=False):
    """
    **Pitch-by-pitch simulation** of one half-inning, with ABS challenges.

    Unlike `source.simulation.simulate_half_inning`, which draws one outcome
    per PA, this plays out every pitch of every PA: balls, called strikes,
    other strikes (swinging, or fouls with fewer than 2 strikes) and the
    pitch that ends the PA (in play or HBP). Fouls with 2 strikes are
    skipped. Called pitches can be close calls, and `offense` / `defense`
    decide whether to challenge them. See the module docstring for the
    details. With no close calls the results match the PA-level
    `simulate_half_inning`, but this runs about 3-4x slower.

    Parameters
    ----------
    lineup : Player or list of Player
        A single Player bats in every spot; a list is the batting order.
    runner_model : SimpleRules or EmpiricalTransitions
        Decides where runners go after each PA.
    pitch_model : PitchModel
        From `PitchModel.fit(pa)`.
    rules : ChallengeRules
        How often close calls happen, how often they are wrong, and how
        many challenges each team has.
    offense, defense : callable, optional
        Challenge strategies, `strategy(s: PitchSituation) -> bool`. The
        offense is asked about close called strikes and the defense about
        close called balls, only while they have a challenge left.
        Default: never challenge.
    rng : random.Random, optional
        Random number generator (a new unseeded one by default).
    outs : int, optional
        Outs at the start (default 0).
    bases : int or str, optional
        Runners on base at the start, e.g. 'R1' or 1 (default empty).
    count : str or (int, int), optional
        Count of the first batter at the start (default '0-0').
    batter_idx : int, optional
        Lineup spot of the first batter (0-indexed).
    log : bool, optional
        If True, record every close call in `close_call_log`.

    Returns
    -------
    HalfInningByPitch
    """
    lineup = as_lineup(lineup)
    rng = rng if rng is not None else random.Random()
    if isinstance(bases, str):
        bases = str_to_bases(bases)
    balls, strikes = parse_count(count)

    n = len(lineup)
    advance = runner_model.advance
    steps = pitch_model.steps
    rates = rules.rates
    left = {'offense': rules.offense_challenges, 'defense': rules.defense_challenges}
    made = {'offense': 0, 'defense': 0}
    won = {'offense': 0, 'defense': 0}
    result = HalfInningByPitch(runs=0, next_batter_idx=batter_idx,
                               close_call_log=[] if log else None)

    for pa_of_inning in range(1, MAX_PA_PER_HALF_INNING + 1):
        player = lineup[batter_idx]
        outcomes = pitch_model.outcome_cumulative(player)
        c = 3 * balls + strikes
        outcome = bisect.bisect_right(outcomes[c], rng.random())
        ended_on_close_call = None    # 'BB' / 'SO' if the PA ends on a close call
        pitch_of_pa = 0

        while True:
            pitch_of_pa += 1
            r = rng.random()
            ball_cut, called_cut, other_cut = steps[c][outcome]
            if r >= other_cut:
                break                                   # in play / HBP: PA over

            if r >= called_cut:
                # Swinging strike or foul: not challengeable.
                strikes += 1
                if strikes == 3:
                    outcome = STRIKEOUT
                    break
                c += 1
                continue

            # A called pitch. Is it a close call, and is it challenged?
            call = 'ball' if r < ball_cut else 'strike'
            result.called_pitches += 1
            close = False
            overturned = False
            rate = rates[c][call]
            if rate > 0 and rng.random() < rate:
                close = True
                result.close_calls += 1
                # Drawn whether or not anyone challenges, so every strategy
                # sees the same random numbers until their choices differ.
                confidence = rules.draw_confidence(rng)
                wrong = rng.random() < confidence

                challenger = 'offense' if call == 'strike' else 'defense'
                other = 'defense' if challenger == 'offense' else 'offense'
                situation = PitchSituation(
                    outs=outs, bases=bases, count=COUNTS[c], call=call, confidence=confidence,
                    challenges_left=left[challenger], opponent_challenges_left=left[other],
                    runs=result.runs, batter=player, batter_idx=batter_idx,
                    pa_of_inning=pa_of_inning, pitch_of_pa=pitch_of_pa)

                challenged = False
                if left[challenger] > 0:
                    strategy = offense if challenger == 'offense' else defense
                    challenged = bool(strategy(situation))
                if challenged:
                    made[challenger] += 1
                    if wrong:
                        won[challenger] += 1
                        if not rules.keep_on_success:
                            left[challenger] -= 1
                    else:
                        left[challenger] -= 1
                overturned = challenged and wrong
                if log:
                    result.close_call_log.append(CloseCall(situation, challenged, wrong, overturned))
                if overturned:
                    call = 'strike' if call == 'ball' else 'ball'

            if call == 'ball':
                balls += 1
            else:
                strikes += 1
            if balls == 4 or strikes == 3:
                if close:
                    ended_on_close_call = WALK_COUNT if balls == 4 else STRIKEOUT_COUNT
                else:
                    outcome = WALK if balls == 4 else STRIKEOUT
                break
            c = 3 * balls + strikes
            if overturned:
                # A new count: draw how the PA ends from here.
                outcome = bisect.bisect_right(outcomes[c], rng.random())

        # Play out the end of the PA.
        if ended_on_close_call is not None:
            bases, scored, outs = _end_pa(ended_on_close_call, bases, outs)
        else:
            bases, scored, outs_added = advance(outcome, bases, outs, rng)
            outs += outs_added
        result.runs += scored
        batter_idx = (batter_idx + 1) % n
        balls = strikes = 0

        if outs >= 3:
            result.next_batter_idx = batter_idx
            result.offense_challenges, result.defense_challenges = made['offense'], made['defense']
            result.offense_won, result.defense_won = won['offense'], won['defense']
            result.offense_left, result.defense_left = left['offense'], left['defense']
            return result

    raise RuntimeError(
        f"A half-inning went {MAX_PA_PER_HALF_INNING} PAs without 3 outs. "
        f"Check that the lineup's out probabilities (SO, GO, FO) aren't ~0.")


def compare_strategies_by_pitch(lineup, runner_model, pitch_model, rules, strategies,
                                n_innings=50_000, seed=None, outs=0, bases=0, count='0-0',
                                batter_idx=0, z=1.96) -> pd.DataFrame:
    """
    **Pitch-by-pitch simulation** of many half-innings under each challenge
    strategy, summarized side by side.

    Each half-inning is played by `simulate_half_inning_by_pitch` (every
    pitch of every PA, not one draw per PA). Half-inning i gets its own
    seed, the same for every strategy, so all strategies face the same
    pitches and close calls in it until their choices first differ. That
    makes the differences between strategies more precise than the run
    totals on their own ('vs_baseline_ci').

    Parameters
    ----------
    lineup : Player or list of Player
    runner_model : SimpleRules or EmpiricalTransitions
    pitch_model : PitchModel
    rules : ChallengeRules
    strategies : dict
        {name: (offense_strategy, defense_strategy)}. The first one is the
        baseline the others are compared to, e.g.
        {'never': (never_challenge, never_challenge),
         'offense always': (always_challenge, never_challenge)}.
    n_innings : int, optional
        Half-innings per strategy. Runs per half-inning have a standard
        deviation of about 1.3, so 50,000 gives a 95% CI of about +/-0.011
        on each mean.
    seed : int, optional
        Seed for reproducible results. If None, one is picked at random
        and still shared by every strategy.
    outs, bases, count, batter_idx : optional
        Starting state, as in `simulate_half_inning_by_pitch`.
    z : float, optional
        z-value for the confidence intervals (1.96 = 95%).

    Returns
    -------
    pd.DataFrame
        One row per strategy (index 'strategy') with:
          - 'runs', 'ci_low', 'ci_high', 'sd': runs per half-inning
          - 'vs_baseline', 'vs_baseline_ci': mean difference in runs from
            the first strategy, and the half-width of its paired CI
          - per half-inning averages: 'called_pitches', 'close_calls',
            'offense_challenges', 'offense_won', 'defense_challenges',
            'defense_won', 'offense_left', 'defense_left'
    """
    if not strategies:
        raise ValueError("Give at least one strategy")
    for name, pair in strategies.items():
        if not (isinstance(pair, tuple) and len(pair) == 2 and all(map(callable, pair))):
            raise TypeError(f"strategies[{name!r}] must be an (offense, defense) pair of functions")

    lineup = as_lineup(lineup)
    if seed is None:
        seed = random.randrange(2 ** 32)
    seeds = random.Random(seed)
    seeds = [seeds.getrandbits(64) for _ in range(n_innings)]
    counters = ['called_pitches', 'close_calls', 'offense_challenges', 'offense_won',
                'defense_challenges', 'defense_won', 'offense_left', 'defense_left']

    rows, baseline = [], None
    for name, (offense, defense) in strategies.items():
        rng = random.Random()
        results = []
        for half_inning_seed in seeds:
            rng.seed(half_inning_seed)
            results.append(simulate_half_inning_by_pitch(lineup, runner_model, pitch_model, rules,
                                                         offense, defense, rng, outs, bases, count,
                                                         batter_idx))
        runs = np.array([r.runs for r in results])
        summary = SimulationResult.from_runs(runs, z=z, per='half-inning')
        if baseline is None:
            baseline = runs
        diff = runs - baseline
        diff_sd = diff.std(ddof=1) if n_innings > 1 else 0.0

        row = {'strategy': name, 'runs': summary.mean, 'ci_low': summary.ci_low,
               'ci_high': summary.ci_high, 'sd': summary.sd,
               'vs_baseline': diff.mean(), 'vs_baseline_ci': z * diff_sd / math.sqrt(n_innings)}
        for key in counters:
            row[key] = np.mean([getattr(r, key) for r in results])
        rows.append(row)

    return pd.DataFrame(rows).set_index('strategy')
