# TruMedia2026.feather — quick reference

Notes on the raw data file as used by the simulator. The full column list is in
[trumedia_cols.txt](trumedia_cols.txt). The code that reads the file is
[source/dataloaders.py](../source/dataloaders.py).

## What's in it

| | |
|---|---|
| Grain | one row per **pitch** |
| Rows | 2,561,383 (11 true duplicates) |
| Columns | 164 |
| Season | 2026 NCAA Division I, 2/13/26 – 6/7/26 |
| Games | 8,433 |
| Batters | 5,345 (1,495 with 200+ PA) |
| Plate appearances | ~678,000 |
| Size | 1.1 GB. Read only the columns you need with `pd.read_feather(path, columns=[...])`. |

## Columns the simulator uses

| Column | Meaning / values |
|---|---|
| `TruMediaGameID`, `Inning`, `Top.Bottom`, `PAofInning`, `PitchofPA` | Locate a pitch. Together they form the unique pitch key. |
| `Batter`, `BatterId`, `BatterTeamName` | Who is batting. Use `BatterId`, because a few names belong to two players. |
| `Outs` | Outs **before** the pitch (0–2). |
| `ManOnFirst`, `ManOnSecond`, `ManOnThird` | Runners on base before the pitch. These are the **strings** `'true'`/`'false'`, not booleans. |
| `isPA` | `1.0` on the pitch that ended the PA, `0.0` otherwise. |
| `PlayResult` | Result of the pitch. On PA-ending pitches it is one of: `Single`, `Double`, `Triple`, `HomeRun`, `Walk`, `HitByPitch`, `Strikeout (K)` (swinging), `Strikeout (ꓘ)` (looking), `Out`, `FieldersChoice`, `Error`, `Sacrifice`. A handful of rows have `BatterInterference`, `NoPlay`, `Ball`, `Strike ...` or `''`. |
| `HitType` | `GroundBall`, `FlyBall`, `LineDrive`, `Popup`, `Undefined` |
| `KorBB` | `Strikeout`, `Walk`, `Undefined` |
| `OutsOnPlay`, `RunsScored` | Outs made and runs scored on the pitch. |
| `PitcherTeamName`, `HomeTeamName`, `AwayTeamName`, `HomeConference`, `AwayConference` | The file has no "batter conference" column. Match `BatterTeamName` / `PitcherTeamName` to the home team to find each side's conference. |
| `isACCSEC` (not loaded) | `1` when **both** teams are SEC or ACC (~44k PAs). This is the same set the simulator uses for "league average". |

## Gotchas

- **Don't deduplicate on `PitchUID`.** 544,313 rows (21%) have a missing
  `PitchUID`. These are real pitches, and `drop_duplicates('PitchUID')` would
  collapse them into one row. Use
  `['TruMediaGameID', 'Inning', 'Top.Bottom', 'PAofInning', 'PitchofPA']` instead.
- **`BaseState` uses an odd code.** `1` = empty, `2` = 3rd, `3` = 2nd,
  `4` = 2nd & 3rd, `5` = 1st, `6` = 1st & 3rd, `7` = 1st & 2nd, `8` = loaded,
  and `000` also means empty. The `ManOn*` columns are easier to work with.
- **`ABResult` is a PA-level label repeated on every pitch of the PA.** For
  example, `Out (GroundBall)` appears on all of that PA's pitches. Counting it
  over all rows overcounts. Use `PlayResult` on `isPA == 1` rows.
- **11% of balls in play have `HitType == 'Undefined'`.** For outs this means
  we can't tell a ground out from a fly out. See
  [source/outcomes.py](../source/outcomes.py) for how these are handled.
- **The base state after a PA isn't stored.** It is read from the first pitch
  of the next PA in the same half-inning. About 98% of PAs satisfy
  "runners before + batter = runners after + runs + outs". The `transition_ok`
  column in the PA table flags them.
- `Year` is always `2026`. Every row has `is_PA == 1`, so that column means
  nothing. Use `isPA` instead.

## Reference numbers (for checking the simulator)

Runs per 9 innings count only completed half-innings and only runs that scored
on PA-ending pitches. GO and FO rates are after splitting the undefined balls in
play.

| | Runs per 9 | 1B | 2B | 3B | HR | BB | HBP | SO | GO | FO |
|---|---|---|---|---|---|---|---|---|---|---|
| SEC/ACC vs SEC/ACC (league average) | **6.33** | .147 | .042 | .003 | .037 | .106 | .027 | .233 | .205 | .199 |
| All D1 | 6.48 | .156 | .044 | .005 | .026 | .110 | .035 | .196 | .212 | .216 |
