# Plan: Mismatch Market

**One question:** on man-coverage snaps, do defenses put linebackers on the tight ends and running backs who beat linebackers?

**Why it is original:** the usual separation leaderboard grades the receiver. This grades the defense's matchup decision.

Every number marked "checkpoint" below was measured on this repo's data with a working prototype. If your number is far off, you have a bug. Stop and fix it before moving on.

---

## 0. Ground rules

- One script: `mismatch_market.py` at the repo root. Running `python3 mismatch_market.py` must produce every output with no arguments.
- Libraries: pandas, numpy, scipy, matplotlib only. All are installed (pandas 2.2.2, scipy 1.14, matplotlib 3.9.1).
- **pyarrow is not installed.** Cache with `df.to_pickle(...)` / `pd.read_pickle(...)`, not parquet.
- The submission write-up is `README.md` (judges read it first). The data dictionary is `DATA.md`.
- Always join on `(gameId, playId)`. `playId` alone is not unique.
- There is **no coverage-assignment column** in this dataset. Who covers whom must be inferred from tracking (Step 2).
- Distances between players do not depend on play direction, so skip direction normalization everywhere except the play picture (Step 7).

Files to create:

```
mismatch_market.py      the whole pipeline
cache/matchups.pkl      output of Step 2 (add cache/ to .gitignore)
outputs/1_market.png    player scatter (main chart)
outputs/2_defenses.png  defense bar chart
outputs/3_adjust.png    did defenses adjust? scatter
outputs/4_play.png      one real play
outputs/players.csv     player table
README.md           3–5 sentence write-up
```

Time budget (3 hours): Steps 1–3 in 45 min, Steps 4–5 in 45 min, Steps 6–7 in 45 min, Step 9 in 15 min, 30 min spare. Step 8 is optional.

---

## 1. Load the small tables

```python
import glob, os
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

plays   = pd.read_csv('data/plays.csv')
scout   = pd.read_csv('data/pffScoutingData.csv')
players = pd.read_csv('data/players.csv')
games   = pd.read_csv('data/games.csv')

man = plays.loc[plays.pff_passCoverageType == 'Man', ['gameId', 'playId']]
roles = scout.merge(man, on=['gameId', 'playId'])
roles = roles[roles.pff_role.isin(['Pass Route', 'Coverage'])]
```

Checkpoints:
- `len(man) == 2481`
- Route snaps by TE or RB on man plays: `4437`

---

## 2. Infer who covers whom (the risky step)

For each man-coverage play:

1. Snap frame = first frame whose `event` is `ball_snap` or `autoevent_ballsnap`.
2. End frame = first frame after the snap whose `event` is one of `pass_forward`, `autoevent_passforward`, `qb_sack`, `qb_strip_sack`, `run`.
3. Skip the play if either frame is missing or the window is shorter than 10 frames (1 second).
4. Cap the end frame at snap + 40 frames (4.0 seconds). This removes scramble drills.
5. Receivers = every player with `pff_role == 'Pass Route'` (all positions, including WRs, so a corner is not wrongly handed to a TE). Defenders = every player with `pff_role == 'Coverage'`.
6. Cost matrix = average distance between each receiver and each defender over the window.
7. `linear_sum_assignment(cost)` gives one defender per receiver. It handles more defenders than receivers.
8. Record separation at the end frame: distance to the **nearest** coverage defender (`sep`).

Verified code:

```python
SNAP = {'ball_snap', 'autoevent_ballsnap'}
END  = {'pass_forward', 'autoevent_passforward', 'qb_sack', 'qb_strip_sack', 'run'}

def build_matchups():
    rows = []
    for f in sorted(glob.glob('data/tracking/*.csv')):
        t = pd.read_csv(f, usecols=['gameId', 'playId', 'nflId', 'frameId', 'x', 'y', 'event'])
        t = t.merge(man, on=['gameId', 'playId'])
        t = t[t.nflId.notna()]                      # drop the ball
        for (g, p), d in t.groupby(['gameId', 'playId']):
            ev = d[['frameId', 'event']].drop_duplicates()
            s = ev[ev.event.isin(SNAP)].frameId.min()
            if pd.isna(s):
                continue
            e = ev[ev.event.isin(END) & (ev.frameId > s)].frameId.min()
            if pd.isna(e) or e - s < 10:
                continue
            e = min(e, s + 40)
            r = roles[(roles.gameId == g) & (roles.playId == p)]
            w = d[(d.frameId >= s) & (d.frameId <= e)]
            X = w.pivot(index='frameId', columns='nflId', values='x')
            Y = w.pivot(index='frameId', columns='nflId', values='y')
            rec = [i for i in r[r.pff_role == 'Pass Route'].nflId if i in X.columns]
            dfn = [i for i in r[r.pff_role == 'Coverage'].nflId if i in X.columns]
            if not rec or not dfn:
                continue
            # D has shape (frames, receivers, defenders)
            D = np.sqrt((X[rec].values[:, :, None] - X[dfn].values[:, None, :]) ** 2 +
                        (Y[rec].values[:, :, None] - Y[dfn].values[:, None, :]) ** 2)
            cost = np.nanmean(D, axis=0)
            ri, ci = linear_sum_assignment(cost)
            for a, b in zip(ri, ci):
                rows.append((g, p, rec[a], dfn[b], cost[a, b], D[-1][a].min(), s, e))
    return pd.DataFrame(rows, columns=['gameId', 'playId', 'nflId', 'defId',
                                       'mean_dist', 'sep', 'snap_frame', 'end_frame'])

os.makedirs('cache', exist_ok=True)
if os.path.exists('cache/matchups.pkl'):
    m = pd.read_pickle('cache/matchups.pkl')
else:
    m = build_matchups()
    m.to_pickle('cache/matchups.pkl')
```

Checkpoints:
- Runs over all 122 games in under 30 seconds.
- About `11,309` matchup rows from `2,476` plays (5 plays skipped).
- Median window length is 2.7 seconds.

---

## 3. Label and filter

```python
pos  = players.set_index('nflId').officialPosition
name = players.set_index('nflId').displayName
m['pos'], m['dpos'], m['name'] = m.nflId.map(pos), m.defId.map(pos), m.nflId.map(name)

m = (m.merge(scout[['gameId', 'playId', 'nflId', 'pff_positionLinedUp']], on=['gameId', 'playId', 'nflId'])
       .merge(plays[['gameId', 'playId', 'defensiveTeam', 'passResult', 'playDescription']], on=['gameId', 'playId'])
       .merge(games[['gameId', 'week']], on='gameId'))

def defender_group(p):
    if p in ('ILB', 'OLB', 'MLB', 'LB'): return 'LB'
    if p in ('CB', 'FS', 'SS', 'DB'):    return 'DB'
    return 'other'

def alignment(a):
    if a.startswith(('HB', 'FB')): return 'backfield'
    if a.startswith('TE'):         return 'inline'
    if a in ('LWR', 'RWR'):        return 'wide'
    return 'slot'

m['dgrp']  = m.dpos.map(defender_group)
m['align'] = m.pff_positionLinedUp.map(alignment)

k = m[m.pos.isin(['TE', 'RB']) & (m.dgrp != 'other') & (m.mean_dist <= 8)].copy()
k['lb'] = (k.dgrp == 'LB').astype(int)
```

The `mean_dist <= 8` filter drops matchups where nobody was really covering him. It is a tuning knob: `<= 6` is stricter and keeps about 2,980 rows.

Checkpoints:
- `len(k)` is about `3,764`.
- Linebacker rate by position and alignment:

| pos | align | snaps | LB rate |
|---|---|---|---|
| RB | backfield | 952 | 0.89 |
| RB | slot | 117 | 0.62 |
| RB | wide | 238 | 0.52 |
| TE | inline | 927 | 0.38 |
| TE | slot | 1046 | 0.26 |
| TE | wide | 425 | 0.15 |

- Average separation: TE vs LB `2.26` yards, TE vs DB `1.84` yards. RB vs LB `3.05`, RB vs DB `2.99`.

That TE gap (about 0.4 yards more space against a linebacker) is the league-wide headline. Put it in the write-up.

---

## 4. The two numbers

Baselines are the average for the same position and alignment bucket:

```python
grp = k.groupby(['pos', 'align'])
k['exp_lb'] = grp.lb.transform('mean')
k['soe']    = k.sep - grp.sep.transform('mean')    # separation over expected
```

Per player:

- **Respect** = expected LB rate − actual LB rate. Positive means defenses avoid putting a linebacker on him.
- **Punishment** = average `soe` on his linebacker-covered snaps. Positive means he gets more space than a typical player from that alignment.

```python
g = k.groupby(['nflId', 'name', 'pos']).agg(
        n=('lb', 'size'), n_lb=('lb', 'sum'),
        lb_rate=('lb', 'mean'), exp_lb=('exp_lb', 'mean')).reset_index()
g = g.merge(k[k.lb == 1].groupby('nflId').soe.mean().rename('punish'), on='nflId', how='left')
g['respect'] = g.exp_lb - g.lb_rate
q = g[(g.n >= 20) & (g.n_lb >= 8)]                 # qualified players
g.to_csv('outputs/players.csv', index=False)
```

Checkpoints:
- `len(q) == 67` (38 TE, 29 RB).
- Correlation between `respect` and `punish` across qualified players is about `0.06`. That near-zero value is the main finding: who gets a linebacker is unrelated to who beats linebackers.
- Face-validity checks, as `assert`s at the bottom of the script:
  - Kyle Pitts: LB rate below 0.10, respect about +0.22.
  - Darren Waller: respect about +0.17.
  - Cordarrelle Patterson: respect about +0.30.
  - RB backfield LB rate above 0.80.
- Under-respected and winning (respect < 0, punish > 0) includes Mark Andrews (+0.97), Hunter Henry (+0.69), Najee Harris (+1.02), Joe Mixon (+0.42).

Known trap: Marcedes Lewis shows up as heavily under-respected with high Punishment. He is a blocking TE who leaks out late, and defenses ignore him on purpose. Say so in the limitations. Do not make him the headline player.

Players like Pitts have almost no linebacker snaps, so they have no Punishment value. That is the result, not a bug: list the top five by Respect in a small text box on the main chart ("defenses already won't put a linebacker on these players").

---

## 5. Chart 1: the market (`outputs/1_market.png`)

- Scatter of qualified players. X = Respect, Y = Punishment.
- Dashed lines at x = 0 and y = 0 make four corners. Label each corner in grey text:
  - left-top: "Under-respected and winning" (shade this corner lightly)
  - right-top: "Respected, still winning"
  - left-bottom: "Linebackers handle him"
  - right-bottom: "Over-respected"
- Colour by position (TE, RB). Dot size by `n_lb`.
- Label the names of every player in the left-top corner and the five largest |Respect| values. Leave the rest unlabelled.
- X-axis label: "Respect: linebacker rate below normal for his alignment". Y-axis label: "Punishment: extra yards of separation vs linebackers".
- Title states the finding, for example: "Defenses don't steer linebackers away from the players who beat them (r = 0.06)".
- `figsize=(11, 8)`, `dpi=150`, `bbox_inches='tight'`.

---

## 6. Charts 2 and 3: grading the defenses

### Chart 2: defense bar chart (`outputs/2_defenses.png`)

Per defense, on all TE/RB man snaps:

```python
games_played = k.groupby('defensiveTeam').gameId.nunique()
lbk = k[k.lb == 1]
d = pd.DataFrame({
    'n': k.groupby('defensiveTeam').size(),
    'lb_over_exp': k.groupby('defensiveTeam').apply(lambda x: (x.lb - x.exp_lb).mean()),
    'tax_per_game': lbk.groupby('defensiveTeam').soe.sum() / games_played,
})
```

- `tax_per_game` = extra yards of separation given up per game on linebacker matchups.
- Horizontal bars, 32 teams, sorted by `tax_per_game`. Positive bars one colour, negative another.
- Write `n` at the end of each bar. Sample sizes range from 42 to 245 snaps per team, so show them.

Checkpoint: per-team actual LB rate ranges from about 0.30 to 0.69, with mean 0.48.

### Chart 3: did defenses adjust? (`outputs/3_adjust.png`)

- Split `k` into early (weeks 1–4) and late (weeks 5–8).
- Per player and half: Respect, and (early only) Punishment.
- Keep players with at least 12 snaps in each half and at least 5 linebacker snaps in the early half. About 39 players have 12+ snaps in both halves; fewer survive the linebacker condition.
- Scatter: X = early Punishment, Y = late Respect − early Respect. Add a least-squares line (`np.polyfit`) and print the correlation in the title.
- An upward slope means defenses learned. A flat line means they did not. Report whichever it is.

This chart has a small sample. Say so on the chart in a caption.

---

## 7. Chart 4: one real play (`outputs/4_play.png`)

1. Pick a headline player from the left-top corner who is a real receiving threat (Mark Andrews or Hunter Henry, not Marcedes Lewis).
2. From his rows in `k` with `lb == 1` and `passResult == 'C'`, keep plays whose `playDescription` contains `"to "` + first initial + `"."` + last name (for example `to M.Andrews`). Take the one with the largest `sep`.
3. Load that game's tracking file and that play. If `playDirection == 'left'`, set `x = 120 - x` and `y = 53.3 - y`.
4. Draw the end frame: offense as blue dots, defense as grey dots, the receiver and his assigned linebacker as larger highlighted dots.
5. Draw both players' paths from `snap_frame` to `end_frame` as lines.
6. Draw a dashed line between them with the separation in yards as a text label.
7. Draw the line of scrimmage as a vertical line. Crop the x-axis to about 10 yards behind and 30 yards past it.
8. Title: the play description, week and teams.

Look at the picture. If the assigned linebacker is obviously not covering him, the matching is wrong on that play: pick the next play, and re-check Step 2.

---

## 8. Optional: put it in EPA

Only if Steps 1–7 and 9 are finished.

- Download `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.csv.gz` with `pd.read_csv(url, usecols=['old_game_id', 'play_id', 'epa', 'receiver_player_name'], low_memory=False)`.
- Cast `old_game_id` to int. Join to `k` on `gameId == old_game_id`, `playId == play_id`.
- Keep rows where the TE/RB was the target: `receiver_player_name` equals first initial + `"."` + last name.
- Report mean EPA per target when covered by a linebacker vs a defensive back. One sentence in the write-up, no chart.

---

## 9. `README.md` (3–5 sentences, fill the blanks from the outputs)

> **Finding.** On man-coverage dropbacks in Weeks 1–8 of 2021, tight ends got about 0.4 more yards of separation when a linebacker covered them than when a defensive back did, yet how often a player drew a linebacker was unrelated to how much he beat linebackers (r = __). Players such as __ and __ kept drawing linebackers at or above the normal rate for their alignment while gaining __ extra yards of space, and defenses such as __ gave up the most separation per game this way.
> **Method.** Man plays come from PFF's coverage tag; each route runner is matched to one coverage defender by minimum average distance from snap to throw (Hungarian assignment), and separation is measured at the throw, capped at 4 seconds, on every route whether or not he was targeted. Respect is the expected linebacker rate for his position and alignment minus his actual rate; Punishment is his separation over expected on linebacker-covered snaps.
> **Assumptions and limits.** Matchups are inferred from tracking, not charted, so pick plays, switches and misclassified zone snaps add noise; separation is not the same as being a threat (a blocking tight end who leaks out late looks "under-respected"); eight weeks gives 20–85 snaps per player, so small differences between players are not reliable.

Also add two lines: how to run (`python3 mismatch_market.py`) and the list of files in `outputs/`.

---

## 10. Before you submit

- Delete `cache/` and run `python3 mismatch_market.py` from scratch. It must finish without errors and write all four PNGs and `players.csv`.
- All `assert`s from Step 4 pass.
- Open each PNG and check that titles, axis labels and player names are readable and not overlapping.
- The numbers quoted in `README.md` match `outputs/players.csv`.

## If time runs short, cut in this order

1. Step 8 (EPA).
2. Chart 3 (did defenses adjust).
3. Chart 4 (the play picture).

Never cut Chart 1, Chart 2 or `README.md`.
