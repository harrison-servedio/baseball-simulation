import pandas as pd


# Cleans the play result column
def _clean_PlayResult(df: pd.DataFrame):
    pass

# Loads the duke data file and cleans it
def load_dukedata(path, verbose = False) -> pd.DataFrame:
    df = pd.read_feather(path)

    entries = len(df)

    df.drop_duplicates('PitchUID')

    if verbose: print(f"Dropped {entries - len(df)} duplicates")

    return df