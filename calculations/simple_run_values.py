import os
import sys
import time

# Let `import source...` work when this file is run directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd

from source.dataloaders import load_pa_table
from source.players import Player
from source.baserunning import EmpiricalTransitions
from source.counts import count_probabilities
from source.challenges import run_expectancy

N_INNINGS = 100_000   # per situation; the 95% CI on each RE is roughly +/- 0.007 runs
SEED = 2026


# ---- Situtation ------
OUTS = 2
BASES = 'R1-R2-R3' # Examples: 'R1' for first base, 'R1-R2-R3' for loaded
COUNT = 'SO' # Count of the batter at the plate, e.g. '1-2'. 'BB' or 'SO' means a call just ended his PA with ball four or strike three, played as a simple walk (forced runners move up) or out (runners hold); the simulation then starts with the next batter.



pa = load_pa_table('TruMedia2026.feather')
empirical = EmpiricalTransitions.fit(pa, min_obs=20)
league_avg = Player.league_average(pa)

matrix = count_probabilities(pa)
# print("\nPA outcome probabilities by count (SEC/ACC):")
# print(matrix.round(3))



print("")
print(run_expectancy(league_avg, empirical, matrix, bases=BASES, outs=OUTS, count=COUNT, n_innings=N_INNINGS, seed=SEED))
print("")