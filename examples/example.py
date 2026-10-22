from source.dataloaders import load_pa_table, observed_runs_per_nine
from source.simulation import simulate_half_inning, nrgg, simulate_games
from source.baserunning import EmpiricalTransitions
from source.outcomes import league_subset
from source.players import Player

pa = load_pa_table('TruMedia2026.feather')

league_avg = Player.league_average(pa)

emp = EmpiricalTransitions.fit(pa, 20)


N_GAMES = 50_000   # per simulation; the 95% CI is roughly +/- 0.035 runs
SEED = 2026


print(nrgg(league_avg, emp, N_GAMES, SEED))


team = []
for _ in range(9):
    team.append(Player.league_average(pa))

print(simulate_games(team, emp, N_GAMES))

print(observed_runs_per_nine(league_subset(pa)))
