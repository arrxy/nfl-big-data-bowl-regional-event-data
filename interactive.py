"""Mismatch Market, interactive explorer.

Run after mismatch_market.py (it builds the matchup cache):

    python3 interactive.py        ->  docs/index.html

One self-contained page (plotly loaded from its CDN) with three tools:
  1. the Respect vs Punishment market, hover any player for his scouting card;
  2. a defense picker: where does this defense put linebackers on TEs and RBs?
  3. the example play, animated, with a live separation readout.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import mismatch_market as mm

DOCS = os.path.join(mm.ROOT, 'docs')
BUCKETS = [('TE', 'inline'), ('TE', 'slot'), ('TE', 'wide'), ('RB', 'backfield')]
FONT = dict(family='-apple-system, Segoe UI, Helvetica, Arial, sans-serif', size=13)


def load():
    if not os.path.exists(mm.CACHE):
        raise SystemExit('Run "python3 mismatch_market.py" first: it builds cache/matchups.pkl.')
    plays, scout, players, games = mm.load_tables()
    k = mm.add_baselines(mm.label_matchups(pd.read_pickle(mm.CACHE), plays, scout, players, games))
    return k, mm.player_table(k)


# ------------------------------------------------------------ 1. the market
def player_card(row, k):
    """Hover text: the numbers a coach would ask for about one player."""
    d = k[k.nflId == row.nflId]
    team = d.possessionTeam.mode().iat[0]
    lines = [f'<b>{row["name"]}</b> ({row.pos}, {team})',
             f'{row.n} man-coverage routes, {row.n_lb} vs a linebacker',
             f'Linebacker on {row.lb_rate:.0%} of routes (normal for his alignment: {row.exp_lb:.0%})',
             f'<b>Respect {row.respect:+.2f}</b>   <b>Punishment {row.punish:+.2f} yd</b> (± {row.punish_se:.2f})',
             '', 'By alignment:']
    for al, a in d.groupby('align'):
        if len(a) >= 3:
            lb = a[a.lb == 1]
            sep = f', {lb.sep.mean():.1f} yd open vs LB' if len(lb) else ''
            lines.append(f'  {al}: {len(a)} routes, LB on {a.lb.mean():.0%}{sep}')
    best = mm.pick_play(k, row['name']).head(1)
    if len(best):
        b = best.iloc[0]
        desc = b.playDescription if len(b.playDescription) < 95 else b.playDescription[:92] + '...'
        lines += ['', f'Best catch vs a linebacker (Week {b.week}, {b.dname}):', f'  {desc}']
    return '<br>'.join(lines)


def market_figure(k, q, r):
    fig = go.Figure()
    xl, yl = q.respect.abs().max() * 1.2, q.punish.abs().max() * 1.2
    fig.add_shape(type='rect', x0=-xl, x1=0, y0=0, y1=yl, fillcolor='rgba(44,160,44,0.08)', line_width=0,
                  layer='below')
    fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
    fig.add_vline(x=0, line_dash='dash', line_color='gray', line_width=1)
    for (x, y, xa, ya, txt) in [(-xl, yl, 'left', 'top', 'Under-respected and winning'),
                                (xl, yl, 'right', 'top', 'Respected, still winning'),
                                (-xl, -yl, 'left', 'bottom', 'Linebackers handle him'),
                                (xl, -yl, 'right', 'bottom', 'Over-respected')]:
        fig.add_annotation(x=x, y=y, text=f'<b>{txt}</b>', showarrow=False, xanchor=xa, yanchor=ya,
                           font=dict(color='gray', size=13))
    for pos, color in mm.COLORS.items():
        s = q[q.pos == pos]
        named = (s.respect < 0) & (s.punish >= 0.5) | (s.respect.abs() >= 0.2)
        fig.add_trace(go.Scatter(
            x=s.respect, y=s.punish, mode='markers+text', name=f'{pos} ({len(s)})',
            text=np.where(named, s.name, ''), textposition='middle right', textfont=dict(size=10),
            marker=dict(size=np.sqrt(s.n_lb) * 4.5, color=color, opacity=0.75, line=dict(color='white', width=1)),
            error_y=dict(type='data', array=s.punish_se, thickness=1, width=0, color='rgba(120,120,120,0.28)'),
            hovertext=[player_card(row, k) for _, row in s.iterrows()], hoverinfo='text'))
    fig.update_layout(
        title=f"Defenses don't steer linebackers away from the TEs and RBs who beat them (r = {r:.2f})",
        xaxis=dict(title='Respect: linebacker rate below normal for his alignment (left = gets more linebackers)',
                   range=[-xl, xl], zeroline=False),
        yaxis=dict(title='Punishment: extra yards of separation vs linebackers', range=[-yl, yl], zeroline=False),
        height=640, font=FONT, plot_bgcolor='white', hoverlabel=dict(align='left', bgcolor='white'),
        legend=dict(title='Click to show/hide', orientation='h', y=-0.14), margin=dict(t=60, r=20))
    return fig


# ------------------------------------------------------- 2. defense picker
def defense_figure(k):
    tax = mm.defense_table(k).sort_values('tax_per_game', ascending=False)
    tax['rank'] = range(1, len(tax) + 1)
    labels = [f'{p} {a}' for p, a in BUCKETS]

    def rates(d):
        out = [d[(d['pos'] == p) & (d['align'] == a)].lb for p, a in BUCKETS]
        return [o.mean() if len(o) else np.nan for o in out], [len(o) for o in out]

    league, _ = rates(k)
    teams = sorted(k.defensiveTeam.unique())
    start = tax.index[0]
    fig = go.Figure(go.Bar(x=labels, y=league, name='League average', marker_color='lightgray',
                           text=[f'{v:.0%}' for v in league], textposition='outside'))
    titles = {}
    for t in teams:
        rate, n = rates(k[k.defensiveTeam == t])
        row = tax.loc[t]
        titles[t] = (f'<b>{t}</b>: linebacker tax {row.tax_per_game:+.1f} yd per game '
                     f'(rank {int(row["rank"])} of {len(tax)}, 1 = pays most)')
        fig.add_trace(go.Bar(
            x=labels, y=rate, name=t, marker_color='#d62728', visible=(t == start),
            text=[f'{v:.0%}<br>n={m}' if m else 'none' for v, m in zip(rate, n)], textposition='outside',
            hovertext=[f'{t} vs {lab}: linebacker on {v:.0%} of {m} routes (league {lg:.0%})'
                       for lab, v, m, lg in zip(labels, rate, n, league)], hoverinfo='text'))
    buttons = [dict(label=t, method='update',
                    args=[{'visible': [True] + [u == t for u in teams]}, {'title.text': titles[t]}])
               for t in teams]
    fig.update_layout(
        title=titles[start], barmode='group', height=480, font=FONT, plot_bgcolor='white',
        yaxis=dict(title='Share of routes covered by a linebacker', tickformat='.0%', range=[0, 1.19]),
        updatemenus=[dict(buttons=buttons, active=teams.index(start), x=1.0, xanchor='right', y=1.18,
                          yanchor='top', showactive=True)],
        annotations=[dict(text='<b>Pick a defense:</b>', x=0.9, xref='paper', y=1.155, yref='paper',
                          showarrow=False, xanchor='right')],
        legend=dict(orientation='h', y=-0.12), margin=dict(t=90, r=20))
    return fig


# --------------------------------------------------------- 3. animated play
def play_figure(row, lead=8, tail=18):
    t = pd.read_csv(os.path.join(mm.DATA, 'tracking', f'tracking_{row.gameId}.csv'))
    t = t[t.playId == row.playId].copy()
    if t.playDirection.iloc[0] == 'left':
        t['x'], t['y'] = 120 - t.x, 53.3 - t.y
    s, e = int(row.snap_frame), int(row.end_frame)
    first, last = max(t.frameId.min(), s - lead), min(t.frameId.max(), e + tail)
    t = t[t.frameId.between(first, last)]
    los = t[(t.team == 'football') & (t.frameId == s)].x.iloc[0]
    men = t[t.team != 'football']
    rec = t[t.nflId == row.nflId].set_index('frameId')
    dfn = t[t.nflId == row.defId].set_index('frameId')

    def traces(f):
        d = t[t.frameId == f]
        o, df_, b = d[d.team == row.possessionTeam], d[d.team == row.defensiveTeam], d[d.team == 'football']
        r, q = rec.loc[f], dfn.loc[f]
        since = max(f, s)
        dot = lambda xs, ys, c, size, name: go.Scatter(                              # noqa: E731
            x=xs, y=ys, mode='markers', name=name, hoverinfo='name',
            marker=dict(size=size, color=c, line=dict(color='white' if size < 16 else 'black', width=1.5)))
        return [
            go.Scatter(x=rec.loc[s:since].x, y=rec.loc[s:since].y, mode='lines', showlegend=False,
                       line=dict(color='#ff7f0e', width=3), hoverinfo='skip'),
            go.Scatter(x=dfn.loc[s:since].x, y=dfn.loc[s:since].y, mode='lines', showlegend=False,
                       line=dict(color='#d62728', width=3), hoverinfo='skip'),
            go.Scatter(x=[r.x, q.x], y=[r.y, q.y], mode='lines', showlegend=False,
                       line=dict(color='black', width=1.5, dash='dash'), hoverinfo='skip'),
            dot(o.x, o.y, '#1f77b4', 11, f'{row.possessionTeam} offense'),
            dot(df_.x, df_.y, '#8c8c8c', 11, f'{row.defensiveTeam} defense'),
            dot(b.x, b.y, '#8b4513', 7, 'ball'),
            dot([r.x], [r.y], '#ff7f0e', 18, f'{row["name"]} ({row.pos})'),
            dot([q.x], [q.y], '#d62728', 18, f'{row.dname} ({row.dpos})'),
        ]

    def readout(f):
        gap = np.hypot(rec.loc[f].x - dfn.loc[f].x, rec.loc[f].y - dfn.loc[f].y)
        phase = 'pre-snap' if f < s else f'{(f - s) / 10:.1f} s after snap' if f <= e else 'ball in the air'
        return f'<b>{phase}  |  separation {gap:.1f} yd</b>'

    frames = [go.Frame(data=traces(f), name=str(f), layout=dict(annotations=[dict(
        text=readout(f), x=0.5, xref='paper', y=1.0, yref='paper', showarrow=False, font=dict(size=16),
        bgcolor='white', bordercolor='gray', borderpad=5)])) for f in range(first, last + 1)]
    x0, x1 = max(0, los - 14), min(120, max(los + 26, men.x.max() + 3))
    y0, y1 = max(0, men.y.min() - 3), min(53.3, men.y.max() + 3)
    lines = [dict(type='line', x0=yd, x1=yd, y0=y0, y1=y1, line=dict(color='white', width=1.5), layer='below')
             for yd in range(10, 111, 5)]
    lines.append(dict(type='line', x0=los, x1=los, y0=y0, y1=y1, line=dict(color='#1f3b73', width=3), layer='below'))
    step = lambda f: dict(method='animate', label=f'{(f - s) / 10:+.1f}s', args=[                # noqa: E731
        [str(f)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=f'Week {row.week}, {row.possessionTeam} vs {row.defensiveTeam}: {row.dname} ({row.dpos}) on '
              f'{row["name"]} in man coverage<br><sup>{row.playDescription}</sup>',
        xaxis=dict(range=[x0, x1], visible=False), yaxis=dict(range=[y0, y1], visible=False, scaleanchor='x'),
        shapes=lines, annotations=frames[0].layout.annotations, plot_bgcolor='#eaf4ea', height=640, font=FONT,
        legend=dict(x=1.01, y=0.95), margin=dict(t=80, l=20, r=20),
        updatemenus=[dict(type='buttons', direction='left', x=0.0, y=-0.06, xanchor='left', yanchor='top',
                          buttons=[dict(label='▶ Play', method='animate', args=[None, dict(
                                        frame=dict(duration=100, redraw=True), transition=dict(duration=0),
                                        fromcurrent=True)]),
                                   dict(label='❚❚ Pause', method='animate', args=[[None], dict(
                                        mode='immediate', frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(steps=[step(f) for f in range(first, last + 1)], x=0.2, len=0.8, y=-0.04,
                      currentvalue=dict(prefix='Time from snap: '))])
    return fig


# ------------------------------------------------------------------- page
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mismatch Market</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; color: #1a1a1a; background: #fafafa;
         margin: 0; line-height: 1.5; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 60px; }}
  h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 44px; }}
  .lead {{ font-size: 1.15rem; border-left: 4px solid #d62728; padding: 4px 14px; background: #fff; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }}
  .stat {{ flex: 1 1 200px; background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 12px 14px; }}
  .stat b {{ display: block; font-size: 1.7rem; }}
  .stat span {{ color: #555; font-size: 0.9rem; }}
  .fig {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 6px; }}
  .note {{ color: #555; font-size: 0.9rem; }}
</style></head><body><main>
<h1>Mismatch Market</h1>
<p class="note">Are defenses putting linebackers on the wrong tight ends and running backs?
   NFL 2021, Weeks 1-8, man-coverage dropbacks. <a href="{repo}">Code and write-up</a>.</p>
<p class="lead">A linebacker on a tight end costs the defense {edge:.1f} yards of separation. Offenses can buy
that matchup by lining the tight end up inline, and defenses are not steering linebackers away from the players
who win it.</p>
<div class="stats">
  <div class="stat"><b>+{edge:.2f} yd</b><span>extra separation for a tight end vs a linebacker
       (95% interval {lo:.2f} to {hi:.2f})</span></div>
  <div class="stat"><b>{inline:.0%} vs {wide:.0%}</b><span>how often a tight end draws a linebacker lined up
       inline vs split wide</span></div>
  <div class="stat"><b>r = {r:.2f}</b><span>link between how often a player draws a linebacker and how much he
       beats one ({n} players)</span></div>
</div>

<h2>1. The market: hover any player for his scouting card</h2>
<p class="note">Left = defenses give him more linebackers than normal for where he lines up. Up = he gets more
space than normal when they do. The shaded corner is an offensive coordinator's target list and a defensive
coordinator's to-fix list. Bars are one standard error: one player's Punishment over eight weeks is noisy, so read
the corner as candidates, not a ranking.</p>
<div class="fig">{market}</div>

<h2>2. Scout a defense: where does it put linebackers?</h2>
<p class="note">Pick a defense to see how often it covers tight ends and running backs with a linebacker from each
alignment, against the league average. A red bar well above grey is a matchup that defense hands out more than
most. n = routes faced, so small bars are thin evidence.</p>
<div class="fig">{defense}</div>

<h2>3. What it looks like: press play</h2>
<p class="note">One completed pass from the shaded corner, with the matched defender highlighted and the separation
updating every tenth of a second. Drag the slider to step through it.</p>
<div class="fig">{play}</div>

<p class="note">Matchups are inferred from player tracking (each route runner is paired with one coverage defender
by smallest average distance from snap to throw), not charted. Data: NFL Next Gen Stats tracking and PFF scouting
from the NFL Big Data Bowl 2023 release.</p>
</main></body></html>
"""


def main():
    k, g = load()
    q = g[g.qualified]
    r = q.respect.corr(q.punish)
    al = mm.alignment_table(k)
    corner = q[(q.respect < 0) & (q.punish > 0)].sort_values('punish', ascending=False)
    row = mm.choose_play(k, corner.name).iloc[0]
    html = lambda fig, js: fig.to_html(full_html=False, include_plotlyjs=js, auto_play=False,  # noqa: E731
                                       config=dict(displaylogo=False, responsive=True))
    page = PAGE.format(
        repo='https://github.com/arrxy/nfl-big-data-bowl-regional-event-data',
        edge=al.loc['all', 'edge'], lo=al.loc['all', 'ci_low'], hi=al.loc['all', 'ci_high'],
        inline=al.loc['inline', 'lb_rate'], wide=al.loc['wide', 'lb_rate'], r=r, n=len(q),
        market=html(market_figure(k, q, r), 'cdn'), defense=html(defense_figure(k), False),
        play=html(play_figure(row), False))
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'Wrote {path} ({os.path.getsize(path) / 1024:.0f} KB): {len(q)} players, '
          f'{k.defensiveTeam.nunique()} defenses, play {row.gameId}/{row.playId}')


if __name__ == '__main__':
    main()
