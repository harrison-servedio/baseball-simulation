"""
Baseball simulator based on Beaudoin (2013), "Various applications to a
more realistic baseball simulator", JQAS 9(3): 271-283.

Modules, in the order data flows through them:

    dataloaders  - read TruMedia2026.feather, build one row per PA
    outcomes     - the 9 PA outcomes and each batter's probabilities
    players      - Player objects (from data, custom, or league average)
    baserunning  - where runners go after each outcome (simple rules or
                   learned from data)
    simulation   - simulate innings / games, NRGG
    counts       - PA outcome probabilities by ball-strike count
    challenges   - run expectancy by base state / outs / count, and the
                   value of an ABS challenge

See examples/basic_simulation.py and examples/challenge_analysis.py for
walkthroughs.
"""
