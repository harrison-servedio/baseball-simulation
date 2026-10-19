# This file contains different player classes that can be used in simulations
import pandas as pd
from source.dataloaders import load_dukedata

class player:
    def __init__(self):
        self.result_matrix = [0.130,   0.152,  0.050,  0.006,  0.026, 0.636] # This a claude generated default that means nothing

    # Returns [walk + hbp, single, double, triple, hr, out]
    # Does not yet account for productive outs yet
    def result(self, count=None):
        return self.results_matrix


class DUKE_DATA_Average_Player(player):
    def __init__(self, path):
        df = load_dukedata(path)

        results = [
            ['Walk', 'HitByPitch', "Catcher's Interference"],
            ['Single'],
            ['Double'],
            ['Triple'],
            ['HomeRun'],
            ['Strikeout (K)', 'Strikeout (ꓘ)', 'Strikeout', 'Out',
            'FieldersChoice', 'Error', 'Sacrifice'],
        ]

        pr = df['PlayResult'].str.strip()
        out = [pr.isin(r).sum() for r in results]
        total = sum(out)
        out = [i/total for i in out]

        self._samplesize = total
        self.results_matrix = out