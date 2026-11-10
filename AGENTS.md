# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Data-only starter for the **NFL Big Data Bowl – NYU Edition**, a 3-hour hackathon (brief: `NFL Big Data Bowl - NYU Edition.pdf`; full column dictionary: `DATA.md`; submission write-up: `README.md`). No code, build, or tests exist yet — anything written here is the submission.

**Task:** build, in Python, a visualization/comparison tool or a new metric that surfaces an original insight about **skill positions (TE, WR, RB)**, for a coach, scout, or broadcaster. Submission = code (notebook or .py) + output (viz/screenshot/metric demo) + a 3–5 sentence README covering findings, methods, assumptions, limitations. Judged on: does it run, clear insight, README; then Football Insight > Technical Execution > Originality (must not rehash the 2023 Big Data Bowl, which was pass-rush/pass-block focused) + Communication. Brief's advice: pick ONE question and go deep.

Allowed libraries per the brief: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, scipy. Free public supplemental data (e.g. nflverse) is allowed.

## Data (2021 season, Weeks 1–8, 122 games)

| File | Keys | Notes |
|---|---|---|
| `data/games.csv` | `gameId` | week, teams |
| `data/plays.csv` | `gameId`, `playId` | down/distance, formation, `pff_passCoverage`, `passResult` |
| `data/players.csv` | `nflId` | `officialPosition` (filter TE/WR/RB here), `displayName` |
| `data/pffScoutingData.csv` | `gameId`, `playId`, `nflId` | `pff_role` (`Pass Route`, `Coverage`, `Pass Block`, `Pass Rush`, `Pass`), `pff_positionLinedUp` |
| `data/tracking/tracking_<gameId>.csv` | `gameId`, `playId`, `nflId`, `frameId` | ~811 MB total, 10 Hz, one file per game |

Gotchas:
- `playId` is **not** unique across games — always join on `(gameId, playId)`.
- Tracking ball rows have `nflId` = NA and `team == "football"`.
- `event` is the string `"None"` when untagged, not NaN.
- Normalize direction: when `playDirection == "left"`, use `x = 120 - x`, `y = 53.3 - y`, and rotate `o`/`dir` by 180°.
- Pass-play window: `ball_snap` → first of {`pass_forward`, `qb_sack`, `qb_strip_sack`, `run`}.
- `gameId`/`playId` merge with nflverse play-by-play `old_game_id`/`play_id`.
- This is a pass-play dataset (dropbacks only) — no designed run plays.

## Working with tracking data

Don't load all 122 tracking files casually. Prototype on one game, then `pd.concat` over `glob("data/tracking/*.csv")` with `usecols` and filtered to the `nflId`s/plays you need. Cache intermediate results to parquet if iterating.

`.gitignore` excludes `*.pdf`.
