# Baseball Simulation

A plate-appearance-by-plate-appearance baseball simulator for NCAA D1 baseball. It
recreates the approach in:

> David Beaudoin (2013). *Various applications to a more realistic baseball simulator.*
> Journal of Quantitative Analysis in Sports 9(3): 271–283.

It uses pitch-level TruMedia data from the 2026 season.

Each plate appearance ends in one of the paper's 9 outcomes (1B, 2B, 3B, HR, BB,
HBP, SO, GO, FO), drawn from the batter's own rates. Runners then move according
to probabilities learned from real games, which covers double plays, sac flies,
taking the extra base, errors and so on. Simulating many games gives stats like
the paper's **NRGG** ("number of runs generated per game"): how many runs a
lineup of nine copies of one player would score.

## Setup

You need Python 3.10+ (developed on 3.13).

```bash
git clone <this repo>
cd baseball-simulation
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

The data file is **not** in git because it is too large (1.1 GB). Get
`TruMedia2026.feather` from the project owner and put it in the repo root:

```
baseball-simulation/
├── TruMedia2026.feather   <- here
├── source/
└── ...
```

The first run reads the raw file and saves a much smaller per-plate-appearance
table to `pa_table.feather`. Later runs load that cache. Both files are
gitignored. To force a rebuild, call `load_pa_table(rebuild=True)`.

## Quick start

Run the walkthrough script from the repo root:

```bash
python examples/basic_simulation.py
```

Or use the pieces directly:

```python
from source.dataloaders import load_pa_table
from source.players import Player
from source.baserunning import SimpleRules, EmpiricalTransitions
from source.simulation import nrgg, simulate_games

pa = load_pa_table()                         # one row per plate appearance

# Runner advancement: learned from data, or the old fixed rules for comparison
model = EmpiricalTransitions.fit(pa)
# model = SimpleRules()

# Three ways to make a player
avg     = Player.league_average(pa)                       # SEC/ACC average
real    = Player.from_data(pa, name='Aaron Piasecki')      # or player_id='1750930555'
slugger = Player.custom('Slugger', baseline=avg, HR=0.07, SO=0.27)

# NRGG: one player in all 9 lineup spots
print(nrgg(real, model, n_games=50_000, seed=1))

# A real batting order (list of 9 Players, leadoff first)
print(simulate_games([real, avg, avg, slugger, avg, avg, avg, avg, avg],
                     model, n_games=50_000, seed=1))
```

Results come back as a `SimulationResult` with the mean runs per game and a 95%
confidence interval. If two intervals don't overlap, the lineups really differ.

## Players

| How | Code | Notes |
|---|---|---|
| League average | `Player.league_average(pa)` | Pools all plate appearances where **both teams are SEC or ACC**. Pass `conferences=None` for all of D1, or a list of other conference names. |
| Real player | `Player.from_data(pa, name=...)` / `player_id=...` | Uses the batter's observed rates. `prior_pa=100` shrinks small samples toward league average. If two players share a name, the error lists their IDs. |
| Custom | `Player.custom(name, baseline=avg, HR=0.07, ...)` | Set any outcomes. The rest are filled in from `baseline`, scaled so everything sums to 1. Use `rates={'1B': 0.2}` for labels that aren't valid Python names. |

For now every batter faces a **league-average pitcher**. Pitcher matchups (the
paper's logistic regression) are not implemented yet.

## Runner models

| Model | What it does |
|---|---|
| `EmpiricalTransitions.fit(pa)` | For each (outcome, runners on base, outs), draws the result (where runners end up, runs, outs) from how often it actually happened in the data. HR/BB/HBP use fixed rules. Situations with fewer than `min_obs` examples fall back to the simple rules. `model.describe('1B', 'R2', outs=2)` shows one situation's probabilities. |
| `SimpleRules()` | The D'Esopo & Lefkowitz (1977) rules used by earlier simulators: singles score a runner from 2nd, and runners never move on outs. Useful as a baseline. |

## Project layout

```
source/
  dataloaders.py   read TruMedia2026.feather -> one row per plate appearance (cached)
  outcomes.py      the 9 outcomes, batter and league-average probabilities
  players.py       Player class (league average / from data / custom)
  baserunning.py   runner advancement models
  simulation.py    simulate half-innings and games, NRGG
examples/
  basic_simulation.py   end-to-end walkthrough
Docs/
  trumedia_data.md      notes and gotchas for the raw data file
  trumedia_cols.txt     full column list of the TruMedia file
```

Every module, class and function has a docstring explaining what it does and how
it maps to the paper. Use `help(...)` or read the source.

## Sanity check

A league-average (SEC/ACC) lineup with `EmpiricalTransitions` scores about
**6.4 runs per 9 innings**. SEC/ACC teams actually scored **6.33** against each
other. The gap comes from things the simulator doesn't model yet (steals, wild
pitches, pickoffs). With `SimpleRules` the same lineup scores about 5.75,
because runners never advance on outs.

## Not yet implemented

- Pitcher-vs-batter matchups (paper Section 2 logistic regression)
- Testing whether the number of outs affects runner advancement, and pooling the
  cases where it doesn't (paper's ANOVA)
- Stolen bases, pickoff throwing errors
- Two-team games, extra innings, win probability and in-game strategy (Section 4)
- Optimal batting order search (Section 5)
