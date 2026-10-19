

class frame:
    def __init__(
            self, 
            team,                   # team - list of 9 players in the correct batting order, can also pass a singular player and he will take all at bats
            base_state = '000',     # base_state - 0 for no player, 1 for a player, goes 1st, 2nd, 3rd
            outs = 0,               # outs - the number of outs in the inning, does accept negatives to simulate longer innings
            at_bat = 1              # at_bat - player up to bat to start the simulation, this value is 1 indexed
    ):
        self.team = team
        self.first, self.second, self.third = [int(i) for i in list(base_state)]
        self.outs = outs
        self.batter = at_bat - 1

    def _simulate(self):
        outs = self.outs
        batter = self.batter
        runs = 0
        hits = 0
        at_bats = 0

        while outs < 3:
            pass

        return runs, hits, at_bats # pitches one day potentially but at bats will be sufficient for now

