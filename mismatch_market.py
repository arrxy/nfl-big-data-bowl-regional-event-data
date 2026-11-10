"""Mismatch Market.

On man-coverage dropbacks (2021 Weeks 1-8), do defenses put linebackers on the
tight ends and running backs who beat linebackers?

Run from anywhere:  python3 mismatch_market.py
Writes outputs/1_market.png, 2_defenses.png, 3_adjust.png, 4_play.png,
outputs/players.csv and outputs/epa_by_matchup.csv. The slow step (matching every
route runner to a coverage defender across 122 tracking files) is cached in
cache/matchups.pkl; delete that file to rebuild it. EPA comes from
external/nflverse_epa_2021.csv, downloaded from nflverse only if it is missing.
"""
import glob
import os
import re
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.text import Text
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
CACHE = os.path.join(ROOT, 'cache', 'matchups.pkl')
OUT = os.path.join(ROOT, 'outputs')
# nflverse play-by-play (CC BY 4.0). Downloaded once and trimmed to this dataset's plays; the trimmed
# copy is committed so the script also runs offline.
EPA_URL = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.csv.gz'
EPA_FILE = os.path.join(ROOT, 'external', 'nflverse_epa_2021.csv')

SNAP = {'ball_snap', 'autoevent_ballsnap'}
END = {'pass_forward', 'autoevent_passforward', 'qb_sack', 'qb_strip_sack', 'run'}
MIN_FRAMES = 10                 # skip plays where the window is under 1.0 s
MAX_FRAMES = 40                 # cap the window at 4.0 s to drop scramble drills
MAX_MEAN_DIST = 8               # yards; farther than this, nobody was really covering him
MIN_SNAPS, MIN_LB_SNAPS = 20, 8  # qualification for the player chart
# Blocking tight end who leaks out late; defenses leave him alone on purpose, so his
# "under-respected" score is not a coverage mistake. Flagged on the chart, never the headline play.
KNOWN_BLOCKERS = {'Marcedes Lewis'}

COLORS = {'TE': '#1f77b4', 'RB': '#ff7f0e'}


# ---------------------------------------------------------------- Step 1: tables
def load_tables():
    plays = pd.read_csv(os.path.join(DATA, 'plays.csv'))
    scout = pd.read_csv(os.path.join(DATA, 'pffScoutingData.csv'))
    players = pd.read_csv(os.path.join(DATA, 'players.csv'))
    games = pd.read_csv(os.path.join(DATA, 'games.csv'))
    return plays, scout, players, games


# ------------------------------------------------- Step 2: who covers whom
def build_matchups(man, roles):
    """Match every route runner to one coverage defender on each man play.

    Cost = average receiver-defender distance from the snap to the throw (capped at
    4 s); the Hungarian algorithm gives each receiver a different defender.
    """
    key = ['gameId', 'playId']
    route = roles[roles.pff_role == 'Pass Route'].groupby(key).nflId.apply(list).to_dict()
    cover = roles[roles.pff_role == 'Coverage'].groupby(key).nflId.apply(list).to_dict()
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, 'tracking', 'tracking_*.csv'))):
        t = pd.read_csv(f, usecols=['gameId', 'playId', 'nflId', 'frameId', 'x', 'y', 'event'])
        t = t[t.nflId.notna()].merge(man, on=key)          # drop the ball, keep man plays
        t['nflId'] = t.nflId.astype(int)
        for (g, p), d in t.groupby(key):
            ev = d[['frameId', 'event']].drop_duplicates()
            s = ev[ev.event.isin(SNAP)].frameId.min()
            if pd.isna(s):
                continue
            e = ev[ev.event.isin(END) & (ev.frameId > s)].frameId.min()
            if pd.isna(e) or e - s < MIN_FRAMES:
                continue
            e = min(e, s + MAX_FRAMES)
            w = d[(d.frameId >= s) & (d.frameId <= e)]
            X = w.pivot(index='frameId', columns='nflId', values='x')
            Y = w.pivot(index='frameId', columns='nflId', values='y')
            rec = [i for i in route.get((g, p), []) if i in X.columns]
            dfn = [i for i in cover.get((g, p), []) if i in X.columns]
            if not rec or not dfn:
                continue
            # D[frame, receiver, defender] = distance in yards
            D = np.hypot(X[rec].values[:, :, None] - X[dfn].values[:, None, :],
                         Y[rec].values[:, :, None] - Y[dfn].values[:, None, :])
            cost = np.nanmean(D, axis=0)
            ri, ci = linear_sum_assignment(np.nan_to_num(cost, nan=1e6))
            for a, b in zip(ri, ci):
                rows.append((g, p, rec[a], dfn[b], cost[a, b], D[-1, a].min(), D[-1, a, b], s, e))
    return pd.DataFrame(rows, columns=['gameId', 'playId', 'nflId', 'defId', 'mean_dist',
                                       'sep', 'sep_assigned', 'snap_frame', 'end_frame'])


# ------------------------------------------------------ Step 3: label, filter
def defender_group(p):
    if p in ('ILB', 'OLB', 'MLB', 'LB'):
        return 'LB'
    if p in ('CB', 'FS', 'SS', 'DB'):
        return 'DB'
    return 'other'


def alignment(a):
    if a.startswith(('HB', 'FB')):
        return 'backfield'
    if a.startswith('TE'):
        return 'inline'
    if a in ('LWR', 'RWR'):
        return 'wide'
    return 'slot'


def label_matchups(m, plays, scout, players, games):
    pos = players.set_index('nflId').officialPosition
    name = players.set_index('nflId').displayName
    m = m.assign(pos=m.nflId.map(pos), dpos=m.defId.map(pos),
                 name=m.nflId.map(name), dname=m.defId.map(name))
    m = (m.merge(scout[['gameId', 'playId', 'nflId', 'pff_positionLinedUp']],
                 on=['gameId', 'playId', 'nflId'])
          .merge(plays[['gameId', 'playId', 'possessionTeam', 'defensiveTeam',
                        'passResult', 'playResult', 'playDescription']], on=['gameId', 'playId'])
          .merge(games[['gameId', 'week']], on='gameId'))
    m['dgrp'] = m.dpos.map(defender_group)
    m['align'] = m.pff_positionLinedUp.map(alignment)
    k = m[m.pos.isin(['TE', 'RB']) & (m.dgrp != 'other') & (m.mean_dist <= MAX_MEAN_DIST)].copy()
    k['lb'] = (k.dgrp == 'LB').astype(int)
    return k


# --------------------------------------------------- Step 4: the two numbers
def add_baselines(k):
    """Expected LB rate and separation for the same position + alignment bucket."""
    grp = k.groupby(['pos', 'align'])
    k['exp_lb'] = grp.lb.transform('mean')
    k['soe'] = k.sep - grp.sep.transform('mean')        # separation over expected
    return k


def player_table(k):
    g = k.groupby(['nflId', 'name', 'pos']).agg(
        n=('lb', 'size'), n_lb=('lb', 'sum'),
        lb_rate=('lb', 'mean'), exp_lb=('exp_lb', 'mean')).reset_index()
    g = g.merge(k[k.lb == 1].groupby('nflId').soe.mean().rename('punish'), on='nflId', how='left')
    g['respect'] = g.exp_lb - g.lb_rate
    g['qualified'] = (g.n >= MIN_SNAPS) & (g.n_lb >= MIN_LB_SNAPS)
    return g


# ------------------------------------------------------------- chart helpers
def spread_labels(ax, labels, steps=400):
    """Nudge annotation labels apart vertically until their text boxes stop overlapping."""
    renderer = ax.figure.canvas.get_renderer()

    def box(t):
        t.update_positions(renderer)
        return Text.get_window_extent(t, renderer).expanded(1.02, 1.1)

    for _ in range(steps):
        boxes = [box(t) for t in labels]
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if boxes[i].overlaps(boxes[j]):
                    hi, lo = (i, j) if boxes[i].y0 >= boxes[j].y0 else (j, i)
                    for idx, dy in ((hi, 1.0), (lo, -1.0)):
                        x, y = labels[idx].xyann
                        labels[idx].xyann = (x, y + dy)
                    moved = True
        if not moved:
            break


def annotate_players(ax, df, x, y, fontsize=8):
    halo = [patheffects.withStroke(linewidth=3, foreground='white')]
    texts = [ax.annotate(row['name'] + (' (blocking TE)' if row['name'] in KNOWN_BLOCKERS else ''),
                         (row[x], row[y]), xytext=(7, 0), textcoords='offset points',
                         fontsize=fontsize, va='center', path_effects=halo, zorder=6,
                         arrowprops=dict(arrowstyle='-', color='0.55', lw=0.5, shrinkA=0, shrinkB=4))
             for _, row in df.iterrows()]
    spread_labels(ax, texts)


# ------------------------------------------------------- Step 5: the market
def plot_market(q, g, r, r_no_blockers, path):
    fig, ax = plt.subplots(figsize=(11, 8))
    xl = q.respect.abs().max() * 1.2
    yl = q.punish.abs().max() * 1.15
    ax.set_xlim(-xl, xl)
    ax.set_ylim(-yl, yl)
    ax.fill_between([-xl, 0], 0, yl, color='#2ca02c', alpha=0.08, zorder=0)
    ax.axvline(0, color='0.5', ls='--', lw=1)
    ax.axhline(0, color='0.5', ls='--', lw=1)
    corner = dict(color='0.45', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.01, 0.99, 'Under-respected and winning', ha='left', va='top', **corner)
    ax.text(0.99, 0.99, 'Respected, still winning', ha='right', va='top', **corner)
    ax.text(0.01, 0.01, 'Linebackers handle him', ha='left', va='bottom', **corner)
    ax.text(0.99, 0.01, 'Over-respected', ha='right', va='bottom', **corner)

    for p, c in COLORS.items():
        s = q[q.pos == p]
        ax.scatter(s.respect, s.punish, s=s.n_lb * 6, color=c, alpha=0.7,
                   edgecolor='white', lw=0.8, label=f'{p} ({len(s)})', zorder=3)
    # name only the clear winners in the shaded corner; naming all of them piles labels on the dots
    corner_players = q[(q.respect < 0) & (q.punish >= 0.3)]
    extremes = q.loc[q.respect.abs().nlargest(5).index]
    fig.canvas.draw()
    annotate_players(ax, pd.concat([corner_players, extremes]).drop_duplicates('nflId'),
                     'respect', 'punish')

    pos_legend = ax.legend(title='Position (players)', loc='upper left',
                           bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.add_artist(pos_legend)
    sizes = [ax.scatter([], [], s=v * 6, color='0.6', alpha=0.7) for v in (10, 30, 60)]
    ax.legend(sizes, ['10', '30', '60'], title='Linebacker matchups', loc='upper left',
              bbox_to_anchor=(1.02, 0.83), frameon=False, labelspacing=1.3)

    top = g[g.n >= MIN_SNAPS].nlargest(5, 'respect')
    lines = [f'{row["name"]} ({row["pos"]}): LB on {row["lb_rate"]:.0%},\n    normal for his alignment {row["exp_lb"]:.0%}'
             for _, row in top.iterrows()]
    ax.text(1.03, 0.42, "Defenses already won't put a\nlinebacker on these players\n(most Respect, 20+ snaps):\n\n"
            + '\n'.join(lines), transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='0.75'))

    ax.set_xlabel('Respect: linebacker rate below normal for his alignment  (left = gets more linebackers)')
    ax.set_ylabel('Punishment: extra yards of separation vs linebackers')
    ax.set_title(f"Defenses don't steer linebackers away from the TEs and RBs who beat them (r = {r:.2f})",
                 fontsize=13, fontweight='bold', loc='left')
    ax.text(0, -0.1, f'Each dot is a TE or RB with {MIN_SNAPS}+ man-coverage routes and {MIN_LB_SNAPS}+ linebacker '
            'matchups, 2021 Weeks 1-8. Separation = yards to the nearest coverage defender at the throw,\n'
            'compared with the average for his position and alignment (inline, slot, wide, backfield). '
            f'Matchups inferred from tracking data. Without the blocking TE, r = {r_no_blockers:.2f}.',
            transform=ax.transAxes, fontsize=8, color='0.35', va='top')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------ Step 6: the defenses
def defense_table(k):
    games_played = k.groupby('defensiveTeam').gameId.nunique()
    lbk = k[k.lb == 1]
    return pd.DataFrame({
        'n': k.groupby('defensiveTeam').size(),
        'lb_rate': k.groupby('defensiveTeam').lb.mean(),
        'lb_over_exp': (k.lb - k.exp_lb).groupby(k.defensiveTeam).mean(),
        'tax_per_game': lbk.groupby('defensiveTeam').soe.sum() / games_played,
    }).fillna({'tax_per_game': 0.0}).sort_values('tax_per_game')


def plot_defenses(d, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = np.where(d.tax_per_game > 0, '#d62728', '#2ca02c')
    ax.barh(d.index, d.tax_per_game, color=colors, alpha=0.85)
    ax.axvline(0, color='0.3', lw=0.8)
    pad = d.tax_per_game.abs().max() * 0.02
    for i, (team, row) in enumerate(d.iterrows()):
        right = row.tax_per_game >= 0
        ax.text(row.tax_per_game + (pad if right else -pad), i,
                f'n={row.n:.0f}, LB {row.lb_rate:.0%}', va='center', ha='left' if right else 'right',
                fontsize=8, color='0.3')
    lim = d.tax_per_game.abs().max() * 1.45
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel('Yards of separation over expected given up per game on linebacker-vs-TE/RB matchups')
    ax.set_title('The linebacker tax: which defenses paid for putting linebackers on TEs and RBs',
                 fontsize=13, fontweight='bold', loc='left')
    ax.text(0, -0.06, 'Red = linebackers gave up more space than a typical defender against that alignment; '
            'green = they held up. This is a per-game total, not a rate,\nso defenses that play more man '
            'coverage pay more. n = TE/RB man-coverage routes faced; LB = share covered by a linebacker. '
            '2021 Weeks 1-8.', transform=ax.transAxes, fontsize=8, color='0.35', va='top')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def adjust_table(k, split=4):
    k = k.assign(half=np.where(k.week <= split, 'early', 'late'))
    h = k.groupby(['nflId', 'half']).agg(n=('lb', 'size'), n_lb=('lb', 'sum'),
                                         lb_rate=('lb', 'mean'), exp_lb=('exp_lb', 'mean'))
    h['respect'] = h.exp_lb - h.lb_rate
    w = h.unstack('half')
    a = pd.DataFrame({
        'n_early': w[('n', 'early')], 'n_late': w[('n', 'late')], 'n_lb_early': w[('n_lb', 'early')],
        'punish_early': k[(k.half == 'early') & (k.lb == 1)].groupby('nflId').soe.mean(),
        'respect_early': w[('respect', 'early')], 'respect_late': w[('respect', 'late')],
    })
    a['d_respect'] = a.respect_late - a.respect_early
    both_halves = ((a.n_early >= 12) & (a.n_late >= 12)).sum()
    a = a[(a.n_early >= 12) & (a.n_late >= 12) & (a.n_lb_early >= 5)].dropna()
    a = a.join(k.groupby('nflId')[['name', 'pos']].first())
    return a, both_halves


def adjust_stats(a, n_perm=5000):
    """Correlation, a permutation p-value, and the slope after controlling for early Respect.

    Players with extreme early Respect drift back toward zero on their own (regression to the
    mean), so the slope on early Punishment is also estimated with early Respect in the model.
    """
    x, y = a.punish_early.values, a.d_respect.values
    r = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(rng.permutation(x), y)[0, 1] for _ in range(n_perm)])
    p = (np.abs(perm) >= abs(r)).mean()
    X = np.column_stack([np.ones(len(a)), x, a.respect_early.values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / (len(a) - X.shape[1]))
    return r, p, beta[1], beta[1] / se[1]


def plot_adjust(a, other_splits, path):
    """`other_splits` maps a different last-early-week to its correlation, shown as a fragility check."""
    r, p_val, b_ctrl, t_ctrl = adjust_stats(a)
    slope, icpt = np.polyfit(a.punish_early, a.d_respect, 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    for ps, c in COLORS.items():
        s = a[a.pos == ps]
        ax.scatter(s.punish_early, s.d_respect, s=60, color=c, alpha=0.75, edgecolor='white', label=ps, zorder=3)
    xs = np.linspace(a.punish_early.min(), a.punish_early.max(), 50)
    ax.plot(xs, slope * xs + icpt, color='0.3', lw=1.5, label=f'least-squares fit (slope {slope:+.2f})')
    ax.axhline(0, color='0.5', ls='--', lw=1)
    ax.axvline(0, color='0.5', ls='--', lw=1)
    fig.canvas.draw()
    extreme = a.loc[(a.punish_early.abs() + a.d_respect.abs() * 4).nlargest(8).index]
    annotate_players(ax, extreme, 'punish_early', 'd_respect')
    if r > 0.3:
        title = ('A hint that defenses catch on: TEs and RBs who beat linebackers in Weeks 1-4\n'
                 f'drew fewer of them in Weeks 5-8 (r = {r:.2f}), but the result is fragile')
    elif r < -0.3:
        title = ('Defenses moved the wrong way: TEs and RBs who beat linebackers in Weeks 1-4\n'
                 f'drew more of them in Weeks 5-8 (r = {r:.2f})')
    else:
        title = ("Defenses didn't adjust: beating linebackers in Weeks 1-4 did not change\n"
                 f'who drew them in Weeks 5-8 (r = {r:.2f})')
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('Punishment in Weeks 1-4: extra yards of separation vs linebackers')
    ax.set_ylabel('Change in Respect, Weeks 5-8 minus Weeks 1-4\n(up = defenses used fewer linebackers on him)')
    ax.legend(loc='lower right', frameon=False)
    ax.text(0, -0.12, f'Small sample: {len(a)} players with 12+ man-coverage routes in each half and 5+ linebacker '
            f'matchups in Weeks 1-4. Permutation p = {p_val:.2f}. Controlling for Weeks 1-4 Respect\n'
            f'(regression to the mean), the slope is {b_ctrl:+.2f} (t = {t_ctrl:.1f}).\n'
            'Fragile: splitting the season elsewhere gives '
            + ', '.join(f'r = {v:+.2f} (Weeks 1-{s} vs {s + 1}-8)' for s, v in other_splits.items())
            + '. Treat this as a hint, not a finding.',
            transform=ax.transAxes, fontsize=8, color='0.35', va='top')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return r, p_val, b_ctrl, t_ctrl


# ------------------------------------------------------- Step 7: a real play
def short_name(name):
    """'Mark Andrews' -> 'M.Andrews', the form used in play descriptions and nflverse."""
    if ' ' not in name:
        return name
    first, last = name.split(' ', 1)
    last = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', '', last)   # outside the f-string: Python < 3.12
    return f'{first[0]}.{last}'


def pick_play(k, who):
    """Completed passes to `who` while a linebacker covered him, most separation first."""
    rows = k[(k.name == who) & (k.lb == 1) & (k.passResult == 'C')
             & k.playDescription.str.contains(f'to {short_name(who)}', regex=False)]
    return rows.sort_values('sep', ascending=False)


def choose_play(k, corner_names):
    """The longest completed gain by an under-respected-and-winning receiver against a linebacker.

    Only plays where the matched linebacker is also the nearest coverage defender at the throw
    count. On the others the matching is doubtful (checked by eye: Mark Andrews' and Hunter Henry's
    biggest-separation plays had the matched linebacker 8-10 yards away, not covering them).
    """
    cand = pd.concat([pick_play(k, who) for who in corner_names if who not in KNOWN_BLOCKERS])
    cand = cand[cand.sep_assigned - cand.sep < 0.3]
    return cand.sort_values(['playResult', 'sep'], ascending=False)


def plot_play(row, path):
    t = pd.read_csv(os.path.join(DATA, 'tracking', f'tracking_{row.gameId}.csv'))
    t = t[t.playId == row.playId].copy()
    if t.playDirection.iloc[0] == 'left':
        t['x'] = 120 - t.x
        t['y'] = 53.3 - t.y
    s, e = row.snap_frame, row.end_frame
    los = t[(t.team == 'football') & (t.frameId == s)].x.iloc[0]
    end = t[t.frameId == e]
    path_r = t[(t.nflId == row.nflId) & t.frameId.between(s, e)].sort_values('frameId')
    path_d = t[(t.nflId == row.defId) & t.frameId.between(s, e)].sort_values('frameId')
    rx, ry = path_r.x.iloc[-1], path_r.y.iloc[-1]
    dx, dy = path_d.x.iloc[-1], path_d.y.iloc[-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#eaf4ea')
    for x0 in (0, 110):                                       # end zones
        ax.axvspan(x0, x0 + 10, color='#c9dfc9', zorder=0)
    for yard in range(10, 111, 5):
        ax.axvline(yard, color='white', lw=2.5 if yard in (10, 110) else 1.5 if yard % 10 == 0 else 0.8, zorder=0)
    ax.axvline(los, color='#1f3b73', lw=2, zorder=1)
    ax.text(los, 0.99, ' line of scrimmage', color='#1f3b73', fontsize=9, va='top',
            transform=ax.get_xaxis_transform())

    off = end[end.team == row.possessionTeam]
    dfn = end[end.team == row.defensiveTeam]
    ball = end[end.team == 'football']
    ax.scatter(off.x, off.y, s=90, color='#1f77b4', edgecolor='white', zorder=3, label=f'{row.possessionTeam} offense')
    ax.scatter(dfn.x, dfn.y, s=90, color='0.55', edgecolor='white', zorder=3, label=f'{row.defensiveTeam} defense')
    ax.scatter(ball.x, ball.y, s=30, color='#8b4513', zorder=4, label='ball')
    for pth, c in ((path_r, '#ff7f0e'), (path_d, '#d62728')):
        ax.plot(pth.x, pth.y, color=c, lw=2.5, zorder=2)
        ax.scatter(pth.x.iloc[:1], pth.y.iloc[:1], s=70, facecolor='white', edgecolor=c, lw=2, zorder=3)
    ax.scatter([rx], [ry], s=260, color='#ff7f0e', edgecolor='black', lw=1.5, zorder=5,
               label=f'{row["name"]} ({row.pos}) route')
    ax.scatter([dx], [dy], s=260, color='#d62728', edgecolor='black', lw=1.5, zorder=5,
               label=f'{row.dname} ({row.dpos}) path')
    gap = np.hypot(rx - dx, ry - dy)
    ax.plot([rx, dx], [ry, dy], color='black', ls='--', lw=1.2, zorder=4)
    # put the label beside the dashed line (perpendicular offset) so it never hides either dot
    perp = np.array([-(ry - dy), rx - dx]) / max(gap, 1e-6)
    ax.annotate(f'{gap:.1f} yd', ((rx + dx) / 2, (ry + dy) / 2), xytext=tuple(perp * 30),
                textcoords='offset points', ha='center', va='center', fontsize=12, fontweight='bold',
                zorder=4, bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
    ax.set_xlim(max(0, los - 10), min(120, los + 30))
    on_field = end[end.team != 'football']          # crop the width to the players, not the whole field
    ax.set_ylim(max(0, on_field.y.min() - 6), min(53.3, on_field.y.max() + 6))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    desc = textwrap.fill(row.playDescription, 95)
    ax.set_title(f'Week {row.week}, {row.possessionTeam} vs {row.defensiveTeam}: '
                 f'{row["name"]} ({row.pos}) vs {row.dname} ({row.dpos}) in man coverage\n{desc}',
                 fontsize=10, loc='left')
    ax.text(0, -0.03, f'Positions at the throw ({(e - s) / 10:.1f} s after the snap). Lines show both paths from '
            f'the snap (hollow circles). Offense moves left to right.\nThe matched linebacker is also the nearest '
            f'coverage defender at the throw ({row.sep:.1f} yd).',
            transform=ax.transAxes, fontsize=8, color='0.35', va='top')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------- Step 8: EPA per target
def load_epa(plays):
    if not os.path.exists(EPA_FILE):
        try:
            pbp = pd.read_csv(EPA_URL, usecols=['old_game_id', 'play_id', 'epa', 'receiver_player_name'],
                              low_memory=False)
        except OSError as err:                     # offline and no committed copy: skip Step 8
            print(f'[8] skipped: could not download nflverse play-by-play ({err})')
            return None
        pbp['old_game_id'] = pbp.old_game_id.astype(int)
        pbp = pbp.merge(plays[['gameId', 'playId']], left_on=['old_game_id', 'play_id'],
                        right_on=['gameId', 'playId'])[['old_game_id', 'play_id', 'epa', 'receiver_player_name']]
        os.makedirs(os.path.dirname(EPA_FILE), exist_ok=True)
        pbp.to_csv(EPA_FILE, index=False)
    return pd.read_csv(EPA_FILE)


def epa_by_matchup(k, pbp):
    """EPA per target for TE/RB targets, split by who covered him (nflverse target = our receiver)."""
    t = k.merge(pbp, left_on=['gameId', 'playId'], right_on=['old_game_id', 'play_id'])
    t = t[t.receiver_player_name == t.name.map(short_name)]
    table = t.groupby(['pos', 'dgrp']).agg(targets=('epa', 'size'), epa_per_target=('epa', 'mean'),
                                           completion=('passResult', lambda v: (v == 'C').mean()),
                                           separation=('sep', 'mean'))
    rng = np.random.default_rng(0)
    for p in ('TE', 'RB'):
        a = t[(t.pos == p) & (t.dgrp == 'LB')].epa.values
        b = t[(t.pos == p) & (t.dgrp == 'DB')].epa.values
        boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        table.loc[(p, 'LB'), 'lb_minus_db'] = a.mean() - b.mean()
        table.loc[(p, 'LB'), 'ci_low'], table.loc[(p, 'LB'), 'ci_high'] = lo, hi
    return table


# ---------------------------------------------------------------- checks
def sanity_checks(g, k):
    by = g.set_index('name')
    assert by.loc['Kyle Pitts', 'lb_rate'] < 0.10, by.loc['Kyle Pitts']
    assert abs(by.loc['Kyle Pitts', 'respect'] - 0.22) < 0.05, by.loc['Kyle Pitts']
    assert abs(by.loc['Darren Waller', 'respect'] - 0.17) < 0.05, by.loc['Darren Waller']
    assert abs(by.loc['Cordarrelle Patterson', 'respect'] - 0.30) < 0.05, by.loc['Cordarrelle Patterson']
    # k['align'], not k.align: DataFrame.align is a pandas method
    assert k[(k['pos'] == 'RB') & (k['align'] == 'backfield')].lb.mean() > 0.80


def main():
    os.makedirs(OUT, exist_ok=True)
    plays, scout, players, games = load_tables()
    pos = players.set_index('nflId').officialPosition

    # Step 1: man-coverage plays and the players' roles on them
    man = plays.loc[plays.pff_passCoverageType == 'Man', ['gameId', 'playId']]
    roles = scout.merge(man, on=['gameId', 'playId'])
    roles = roles[roles.pff_role.isin(['Pass Route', 'Coverage'])]
    te_rb_routes = roles[roles.pff_role == 'Pass Route'].nflId.map(pos).isin(['TE', 'RB']).sum()
    print(f'[1] man plays: {len(man)}, TE/RB route snaps on them: {te_rb_routes}')

    # Step 2: infer who covers whom (cached)
    if os.path.exists(CACHE):
        m = pd.read_pickle(CACHE)
    else:
        m = build_matchups(man, roles)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        m.to_pickle(CACHE)
    n_plays = m[['gameId', 'playId']].drop_duplicates().shape[0]
    window = m.drop_duplicates(['gameId', 'playId'])
    print(f'[2] matchup rows: {len(m)} from {n_plays} plays; '
          f'median window {((window.end_frame - window.snap_frame) / 10).median():.1f} s')

    # Step 3: TE/RB routes covered by a linebacker or defensive back
    k = add_baselines(label_matchups(m, plays, scout, players, games))
    print(f'[3] TE/RB matchups kept: {len(k)}')
    print(k.groupby(['pos', 'align']).agg(snaps=('lb', 'size'), lb_rate=('lb', 'mean')).round(2).to_string())
    print(k.groupby(['pos', 'dgrp']).sep.mean().round(2).rename('avg separation').to_string())

    # Step 4: Respect and Punishment per player
    g = player_table(k)
    g.sort_values(['qualified', 'respect'], ascending=[False, True]).round(3).to_csv(
        os.path.join(OUT, 'players.csv'), index=False)
    q = g[g.qualified]
    r = q.respect.corr(q.punish)
    print(f'[4] qualified players: {len(q)} ({(q.pos == "TE").sum()} TE, {(q.pos == "RB").sum()} RB); '
          f'corr(respect, punish) = {r:.3f}')
    ur = q[(q.respect < 0) & (q.punish > 0)].sort_values('punish', ascending=False)
    print('    under-respected and winning:\n' +
          ur[['name', 'pos', 'n', 'n_lb', 'lb_rate', 'exp_lb', 'respect', 'punish']].round(2).to_string(index=False))
    print('    most respected (20+ snaps):\n' +
          g[g.n >= MIN_SNAPS].nlargest(5, 'respect')[['name', 'pos', 'n', 'lb_rate', 'exp_lb', 'respect']]
          .round(2).to_string(index=False))

    # Step 5: the market chart
    q_nb = q[~q.name.isin(KNOWN_BLOCKERS)]
    r_nb = q_nb.respect.corr(q_nb.punish)
    print(f'    corr without known blocking TEs = {r_nb:.3f}')
    plot_market(q, g, r, r_nb, os.path.join(OUT, '1_market.png'))

    # Step 6: grading the defenses
    d = defense_table(k)
    print(f'[6] team LB rate range {d.lb_rate.min():.2f}-{d.lb_rate.max():.2f}, mean {d.lb_rate.mean():.2f}; '
          f'snaps per team {d.n.min()}-{d.n.max()}')
    print('    biggest linebacker tax per game:\n' +
          d.sort_values('tax_per_game', ascending=False).head(5).round(2).to_string())
    plot_defenses(d, os.path.join(OUT, '2_defenses.png'))
    a, both_halves = adjust_table(k)
    other_splits = {}
    for s in (3, 5):                                # same test with the season split a week either side
        alt = adjust_table(k, split=s)[0]
        other_splits[s] = np.corrcoef(alt.punish_early, alt.d_respect)[0, 1]
    r_adj, p_adj, b_ctrl, t_ctrl = plot_adjust(a, other_splits, os.path.join(OUT, '3_adjust.png'))
    print(f'    adjust: {both_halves} players with 12+ snaps in both halves, {len(a)} kept; r = {r_adj:.3f}, '
          f'permutation p = {p_adj:.3f}; slope controlling for early Respect {b_ctrl:+.3f} (t = {t_ctrl:.1f}); '
          'other splits: ' + ', '.join(f'week {s}: r = {v:+.2f}' for s, v in other_splits.items()))

    # Step 7: one real play
    cand = choose_play(k, ur.name)
    if len(cand):
        row = cand.iloc[0]
        plot_play(row, os.path.join(OUT, '4_play.png'))
        print(f'[7] play: {row["name"]} vs {row.dname} ({row.dpos}), game {row.gameId} play {row.playId}, '
              f'{row.playResult} yds, sep {row.sep:.1f} yd')
    else:
        print('[7] no clean linebacker-covered completion found; 4_play.png not written')

    # Step 8: is the extra space worth points?
    pbp = load_epa(plays)
    if pbp is not None:
        epa = epa_by_matchup(k, pbp)
        epa.round(3).to_csv(os.path.join(OUT, 'epa_by_matchup.csv'))
        print('[8] EPA per target by matchup (bootstrap 95% interval for LB minus DB):\n'
              + epa.round(3).to_string())

    sanity_checks(g, k)
    print('Done. Wrote:', ', '.join(sorted(os.listdir(OUT))))


if __name__ == '__main__':
    main()
