import random as r

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
        first, second, third = self.first, self.second, self.third

        while outs < 3:
            at_bats += 1
            # Gets the person up to bat
            if isinstance(self.team, list):
                m = self.team[batter].result()
                batter = (batter+1)%len(self.team)
            else:
                m = self.team.result()

            idx = r.choices(range(len(m)), weights=m, k=1)[0]

            match idx:
                # Walk, hbp, catcher's interference
                case 0:
                    if first:
                        if second:
                            if third:
                                runs += 1
                            third = 1
                        second = 1
                    first = 1
                
                # Single - need to calculate the chance the runner at 2nd scores
                # Assumes not first to third - calulcuate that later too
                case 1:
                    hits += 1
                    if third:
                        third = 0
                        runs += 1
                    if second: # Assumes runner on second does not score unless 2 outs
                        second = 0
                        if outs == 2:
                            runs += 1
                        else:
                            third = 1
                    if first:
                        second = 1
                    first = 1

                # Double
                # Does not check if the runner at first scores
                case 2:
                    hits += 1
                    if third:
                        runs += 1
                        third = 0
                    if second:
                        runs += 1
                    if first:
                        third = 1
                        first = 0
                    second = 1
                
                # Triple
                case 3:
                    hits += 1
                    if third:
                        runs += 1
                    if second:
                        runs += 1
                        second = 0
                    if first:
                        runs += 1
                        first = 0
                    third = 1
                # HR
                case 4:
                    hits += 1
                    if first:
                        runs += 1
                    if second:
                        runs += 1
                    if third:
                        runs += 1
                    first, second, third = 0, 0, 0
                    runs += 1
                # Out
                case 5:
                    outs += 1

        return runs, hits, at_bats # pitches one day potentially but at bats will be sufficient for now

