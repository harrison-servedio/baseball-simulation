"""
The game simulator.

Each PA is simulated in two steps (paper Section 2, "steps executed by
the simulator for each at-bat"):

    1. Draw one of the 9 outcomes from the batter's probabilities
       (`Player.probs`).
    2. Move the batter and runners with a runner model
       (`source.baserunning.SimpleRules` or `EmpiricalTransitions`).

Repeat until 3 outs, keep the batting order going from inning to inning,
and add up the runs.

Only one team's offense is simulated: there is no opponent and no
pitcher yet, so a "game" is 9 innings of one lineup batting. That is all
that's needed for the paper's NRGG statistic (Section 3) and the batting
order application (Section 5).
TODO: two-team games, extra innings, in-game strategy (Section 4).

Example
-------
    import random
    from source.dataloaders import load_pa_table
    from source.players import Player
    from source.baserunning import EmpiricalTransitions
    from source.simulation import simulate_games, nrgg

    pa = load_pa_table()
    model = EmpiricalTransitions.fit(pa)
    avg = Player.league_average(pa)

    # One player in all 9 spots (the paper's NRGG)
    print(nrgg(avg, model, n_games=20_000, seed=1))

    # A 9-man lineup
    lineup = [Player.from_data(pa, name=n) for n in names]
    print(simulate_games(lineup, model, n_games=20_000, seed=1))

Randomness
----------
Python's `random.Random` is used instead of numpy because drawing one
number at a time is much faster with it. Pass `seed=` to get the same
results every run.
"""

import bisect
import math
import random
from dataclasses import dataclass

import numpy as np

from source.players import Player


# A half-inning that hasn't ended after this many PAs almost surely means
# a custom player with (nearly) zero chance of making an out.
MAX_PA_PER_HALF_INNING = 500


# --------------------------------------------------------------------------
# Lineups
# --------------------------------------------------------------------------

def as_lineup(lineup):
    """
    Turn a lineup argument into a list of Players.

    Parameters
    ----------
    lineup : Player or list of Player
        A single Player bats in every spot (9 copies), as in the paper's
        NRGG. A list is used as the batting order, first to last. It is
        normally 9 players, but any length works.

    Returns
    -------
    list of Player
    """
    if isinstance(lineup, Player):
        return [lineup] * 9
    lineup = list(lineup)
    if not lineup:
        raise ValueError("The lineup is empty")
    for p in lineup:
        if not isinstance(p, Player):
            raise TypeError(f"Lineup entries must be Player objects, got {type(p).__name__}")
    return lineup


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate_half_inning(lineup, batter_idx, runner_model, rng, outs=0, bases=0):
    """
    Simulate one half-inning.

    Parameters
    ----------
    lineup : list of Player
        Batting order (use `as_lineup` to build it).
    batter_idx : int
        Position in `lineup` of the first batter (0-indexed).
    runner_model : SimpleRules or EmpiricalTransitions
        Decides where runners go after each outcome.
    rng : random.Random
        Random number generator.
    outs : int, optional
        Outs at the start (default 0). Lets you start mid-inning.
    bases : int, optional
        Runners on base at the start, as a 3-bit int (default 0, empty).

    Returns
    -------
    runs : int
        Runs scored in the half-inning.
    next_batter_idx : int
        Who leads off the next inning.
    """
    runs = 0
    n = len(lineup)
    advance = runner_model.advance      # local name: a bit faster in the loop

    for _ in range(MAX_PA_PER_HALF_INNING):
        player = lineup[batter_idx]
        batter_idx = (batter_idx + 1) % n

        # Step 1: which of the 9 outcomes happens.
        outcome = bisect.bisect_right(player.cumulative, rng.random())

        # Step 2: move the runners.
        bases, scored, outs_added = advance(outcome, bases, outs, rng)
        runs += scored
        outs += outs_added
        if outs >= 3:
            return runs, batter_idx

    raise RuntimeError(
        f"A half-inning went {MAX_PA_PER_HALF_INNING} PAs without 3 outs. "
        f"Check that the lineup's out probabilities (SO, GO, FO) aren't ~0.")


def simulate_game(lineup, runner_model, rng, innings=9):
    """
    Simulate one team batting for a full game.

    The batting order carries over from one inning to the next.

    Parameters
    ----------
    lineup : list of Player
        Batting order (use `as_lineup` to build it).
    runner_model : SimpleRules or EmpiricalTransitions
    rng : random.Random
    innings : int, optional
        Number of innings to bat (default 9).

    Returns
    -------
    int
        Total runs scored.
    """
    runs, batter_idx = 0, 0
    for _ in range(innings):
        inning_runs, batter_idx = simulate_half_inning(lineup, batter_idx, runner_model, rng)
        runs += inning_runs
    return runs


@dataclass
class SimulationResult:
    """
    Summary of many simulated games.

    Attributes
    ----------
    runs : np.ndarray
        Runs scored in each simulated game.
    mean : float
        Average runs per game.
    sd : float
        Standard deviation of runs per game.
    n_games : int
    ci_low, ci_high : float
        95% confidence interval for the mean, `mean -/+ 1.96 * sd / sqrt(n)`
        (paper Section 3). Use these to decide whether two lineups or
        players really differ, as the paper does in Section 5.
    """
    runs: np.ndarray
    mean: float
    sd: float
    n_games: int
    ci_low: float
    ci_high: float

    @classmethod
    def from_runs(cls, runs, z=1.96):
        """Build the summary from an array of runs per game."""
        runs = np.asarray(runs)
        mean = runs.mean()
        sd = runs.std(ddof=1) if len(runs) > 1 else 0.0
        half_width = z * sd / math.sqrt(len(runs))
        return cls(runs, mean, sd, len(runs), mean - half_width, mean + half_width)

    def __repr__(self):
        return (f"SimulationResult(mean={self.mean:.3f} runs/game, "
                f"95% CI=({self.ci_low:.3f}, {self.ci_high:.3f}), "
                f"sd={self.sd:.3f}, n_games={self.n_games:,})")


def simulate_games(lineup, runner_model, n_games=10_000, seed=None, innings=9):
    """
    Simulate many games and summarize runs per game.

    Parameters
    ----------
    lineup : Player or list of Player
        A single Player fills all 9 spots; a list is the batting order.
    runner_model : SimpleRules or EmpiricalTransitions
    n_games : int, optional
        Number of games. The CI half-width shrinks like 1/sqrt(n_games);
        with sd near 4 runs, 10,000 games gives about +/-0.08 and 100,000
        about +/-0.025. (The paper used 2,500,000 for NRGG.)
    seed : int, optional
        Seed for reproducible results.
    innings : int, optional
        Innings per game (default 9).

    Returns
    -------
    SimulationResult
    """
    lineup = as_lineup(lineup)
    rng = random.Random(seed)
    runs = [simulate_game(lineup, runner_model, rng, innings) for _ in range(n_games)]
    return SimulationResult.from_runs(runs)


def nrgg(player, runner_model, n_games=10_000, seed=None):
    """
    The paper's NRGG ("number of runs generated per game", Section 3).

    The average runs per 9 innings scored by a lineup of nine copies of
    `player`. It turns everything about a batter into one number in
    easy-to-understand units.

    Parameters
    ----------
    player : Player
    runner_model : SimpleRules or EmpiricalTransitions
    n_games : int, optional
    seed : int, optional

    Returns
    -------
    SimulationResult
        `.mean` is the NRGG.
    """
    if not isinstance(player, Player):
        raise TypeError("nrgg() takes a single Player")
    return simulate_games(player, runner_model, n_games=n_games, seed=seed)
