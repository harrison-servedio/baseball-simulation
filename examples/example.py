import os
import sys

# Let `import source...` work when this file is run directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from source.dataloaders import load_pa_table, observed_runs_per_nine
from source.simulation import simulate_half_inning, nrgg, simulate_games
from source.baserunning import EmpiricalTransitions
from source.outcomes import league_subset
from source.players import Player

pa = load_pa_table('TruMedia2026.feather')

# Only PAs where both teams are in the ACC or SEC.
acc_sec = league_subset(pa)

league_avg = Player.league_average(acc_sec)

emp = EmpiricalTransitions.fit(acc_sec, 20)

# player = Player.from_data(acc_sec, name='')
# print(player.probs)

N_GAMES = 50_000

print("\nAverage player")
print(nrgg(league_avg, emp, N_GAMES))


print(f"\nActual average runs per 9: {observed_runs_per_nine(acc_sec)}")