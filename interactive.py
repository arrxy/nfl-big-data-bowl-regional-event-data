"""Mismatch Market, interactive explorer.

Run after mismatch_market.py (it builds the matchup cache):

    python3 interactive.py        ->  docs/index.html

Python computes every number and embeds it as JSON; the page itself is plain HTML, CSS
and SVG with no JavaScript libraries. Three tools:
  1. the Respect vs Punishment market, with a scouting card for the selected player;
  2. a defense picker: where does this defense put linebackers on TEs and RBs?
  3. the example play, animated, with a live separation readout.
"""
import json
import math
import os

import pandas as pd

import mismatch_market as mm

DOCS = os.path.join(mm.ROOT, 'docs')
REPO = 'https://github.com/arrxy/nfl-big-data-bowl-regional-event-data'
BUCKETS = [('TE', 'inline'), ('TE', 'slot'), ('TE', 'wide'), ('RB', 'backfield')]
DEFAULT_PLAYER = 'Mark Andrews'


def num(v, digits=3):
    """Float for JSON: rounded, and None instead of NaN."""
    return None if v is None or pd.isna(v) else round(float(v), digits)


def load():
    if not os.path.exists(mm.CACHE):
        raise SystemExit('Run "python3 mismatch_market.py" first: it builds cache/matchups.pkl.')
    plays, scout, players, games = mm.load_tables()
    k = mm.add_baselines(mm.label_matchups(pd.read_pickle(mm.CACHE), plays, scout, players, games))
    return k, mm.player_table(k)


def player_data(k, q):
    out = []
    for _, r in q.iterrows():
        d = k[k.nflId == r.nflId]
        by_align = [dict(a=a, n=len(x), lb=num(x.lb.mean()), sep=num(x[x.lb == 1].sep.mean(), 1))
                    for a, x in d.groupby('align') if len(x) >= 3]
        best = mm.pick_play(k, r['name']).head(1)
        out.append(dict(
            name=r['name'], pos=r.pos, team=d.possessionTeam.mode().iat[0], n=int(r.n), n_lb=int(r.n_lb),
            lb_rate=num(r.lb_rate), exp_lb=num(r.exp_lb), respect=num(r.respect), punish=num(r.punish),
            se=num(r.punish_se), align=by_align, blocker=r['name'] in mm.KNOWN_BLOCKERS,
            label=bool((r.respect < 0 and r.punish >= 0.5) or abs(r.respect) >= 0.2),
            best=None if best.empty else dict(week=int(best.iloc[0].week), dname=best.iloc[0].dname,
                                              desc=best.iloc[0].playDescription)))
    return out


def defense_data(k):
    def rates(d):
        cells = [d[(d['pos'] == p) & (d['align'] == a)].lb for p, a in BUCKETS]
        return [dict(label=f'{p} {a}', rate=num(c.mean()) if len(c) else None, n=len(c))
                for (p, a), c in zip(BUCKETS, cells)]

    tax = mm.defense_table(k).sort_values('tax_per_game', ascending=False)
    league = [r['rate'] for r in rates(k)]
    teams = {t: dict(tax=num(row.tax_per_game, 1), rank=i + 1, rows=rates(k[k.defensiveTeam == t]))
             for i, (t, row) in enumerate(tax.iterrows())}
    return dict(league=league, teams=teams, start=tax.index[0])


def play_data(row, lead=8, tail=18):
    t = pd.read_csv(os.path.join(mm.DATA, 'tracking', f'tracking_{row.gameId}.csv'))
    t = t[t.playId == row.playId].copy()
    if t.playDirection.iloc[0] == 'left':
        t['x'], t['y'] = 120 - t.x, 53.3 - t.y
    s, e = int(row.snap_frame), int(row.end_frame)
    first, last = max(t.frameId.min(), s - lead), min(t.frameId.max(), e + tail)
    t = t[t.frameId.between(first, last)]
    los = float(t[(t.team == 'football') & (t.frameId == s)].x.iloc[0])
    men = t[t.team != 'football']
    xy = lambda d: [[round(a, 2), round(b, 2)] for a, b in zip(d.x, d.y)]                    # noqa: E731
    frames = []
    for f, d in t.groupby('frameId'):
        r, q = d[d.nflId == row.nflId].iloc[0], d[d.nflId == row.defId].iloc[0]
        others = d[~d.nflId.isin([row.nflId, row.defId])]
        frames.append(dict(
            t=round((f - s) / 10, 1), phase='pre' if f < s else 'live' if f <= e else 'air',
            off=xy(others[others.team == row.possessionTeam]), dfn=xy(others[others.team == row.defensiveTeam]),
            ball=xy(d[d.team == 'football']), r=[round(r.x, 2), round(r.y, 2)], d=[round(q.x, 2), round(q.y, 2)],
            gap=round(math.hypot(r.x - q.x, r.y - q.y), 1)))
    x0 = max(0, 5 * math.floor((los - 13) / 5))
    return dict(
        frames=frames, los=round(los, 2), x0=x0, x1=min(120, max(x0 + 40, 5 * math.ceil((men.x.max() + 2) / 5))),
        y0=round(max(0, men.y.min() - 4), 1), y1=round(min(53.3, men.y.max() + 4), 1),
        title=f'Week {row.week}, {row.possessionTeam} vs {row.defensiveTeam}: {row.dname} ({row.dpos}) on '
              f'{row["name"]} in man coverage',
        desc=row.playDescription, off=row.possessionTeam, dfn=row.defensiveTeam,
        rec=f'{row["name"].split()[-1]} · {row.pos}', cov=f'{row.dname.split()[-1]} · {row.dpos}',
        rec_full=f'{row["name"]} ({row.pos})', cov_full=f'{row.dname} ({row.dpos})')


def main():
    k, g = load()
    q = g[g.qualified]
    al = mm.alignment_table(k)
    corner = q[(q.respect < 0) & (q.punish > 0)].sort_values('punish', ascending=False)
    row = mm.choose_play(k, corner.name).iloc[0]
    data = dict(
        players=player_data(k, q), defense=defense_data(k), play=play_data(row), start=DEFAULT_PLAYER, repo=REPO,
        stats=dict(edge=num(al.loc['all', 'edge'], 2), lo=num(al.loc['all', 'ci_low'], 2),
                   hi=num(al.loc['all', 'ci_high'], 2), inline=num(al.loc['inline', 'lb_rate'], 2),
                   wide=num(al.loc['wide', 'lb_rate'], 2), r=num(q.respect.corr(q.punish), 2), n=len(q),
                   n_def=k.defensiveTeam.nunique()))
    page = PAGE.replace('__DATA__', json.dumps(data, allow_nan=False).replace('</', '<\\/'))
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'Wrote {path} ({os.path.getsize(path) / 1024:.0f} KB): {len(q)} players, '
          f'{len(data["defense"]["teams"])} defenses, {len(data["play"]["frames"])} play frames')


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mismatch Market</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root { --navy:#0a2a5e; --red:#d0181f; --gold:#f0b429; --bg:#f3f5fa; --ink:#131722; --mute:#59616f; --line:#d8dde7;
        --serif:'Instrument Serif', Georgia, 'Times New Roman', serif; --mono:'IBM Plex Mono', ui-monospace, Menlo, monospace;
        --sans:Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans); font-size:15px; line-height:1.5; }
main { max-width:1120px; margin:0 auto; padding:28px 16px 64px; }
.cap { font-size:11px; letter-spacing:.09em; text-transform:uppercase; font-weight:600; color:var(--mute); }
.mono { font-family:var(--mono); font-size:12px; color:var(--mute); }
.hero { background:var(--navy); color:#fff; padding:44px 44px 48px; border-bottom:8px solid var(--red);
        background-image:repeating-linear-gradient(90deg, transparent 0 79px, rgba(255,255,255,.05) 79px 80px); }
.hero .top { display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; }
.hero .cap { color:#dfe6f3; } .tag { background:var(--red); color:#fff; padding:3px 8px; margin-right:10px; }
.hero a { color:#fff; font-size:11px; letter-spacing:.09em; font-weight:700; text-underline-offset:4px; }
h1 { font-family:var(--serif); font-weight:400; font-size:clamp(52px, 9vw, 92px); line-height:1; margin:26px 0 18px; letter-spacing:-.01em; }
.hero .q { font-family:var(--serif); font-style:italic; color:var(--gold); font-size:clamp(24px, 3.4vw, 34px); line-height:1.15; max-width:800px; margin:0 0 22px; }
.hero .lead { max-width:700px; margin:0; font-size:16px; color:#eef2f9; }
.stats { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:18px; margin:44px 0 0; }
.stat { background:#fff; border:1px solid var(--navy); border-top:6px solid var(--navy); padding:22px 22px 18px; }
.stat b { display:block; font-family:var(--serif); font-weight:400; font-size:58px; line-height:1.05; margin:6px 0 10px; }
.stat b i { font-size:.5em; } .stat.red b { color:var(--red); }
.stat p { margin:0 0 12px; font-size:13px; } .stat .mono { border-top:1px solid var(--line); padding-top:10px; }
section { margin-top:76px; border-top:3px solid var(--navy); padding-top:22px; }
.head { display:flex; gap:18px; align-items:flex-start; margin-bottom:22px; }
.no { background:var(--red); color:#fff; font-family:var(--mono); font-size:12px; padding:5px 8px; margin-top:10px; }
h2 { font-family:var(--serif); font-weight:400; font-size:46px; line-height:1.05; margin:0 0 8px; }
.head p { margin:0; max-width:740px; color:#333a47; font-size:14px; }
.card { background:#fff; border:1px solid var(--navy); }
.split { display:grid; grid-template-columns:minmax(0, 1fr) 320px; gap:24px; align-items:start; }
.note { color:var(--mute); font-size:13px; margin:14px 0 0; }
/* market */
.market { padding:16px; }
.tabs { display:inline-flex; border:2px solid var(--navy); }
.tabs button, .teams button, .play-btn { font:600 13px var(--sans); cursor:pointer; }
.tabs button { border:0; border-right:1px solid var(--navy); background:#fff; color:var(--navy); padding:8px 14px; }
.tabs button:last-child { border-right:0; } .tabs button[aria-pressed=true] { background:var(--navy); color:#fff; }
.legend { display:flex; flex-wrap:wrap; gap:16px; font-size:12px; color:#333a47; margin:14px 0 4px; align-items:center; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; border:1.5px solid var(--navy); vertical-align:-1px; }
svg { display:block; width:100%; height:auto; }
svg text { font-family:var(--sans); }
.pt { cursor:pointer; outline:none; } .pt circle { stroke:#fff; stroke-width:1.2; }
.pt.RB circle { stroke:var(--navy); } .pt.hide, .lbl.hide { display:none; }
.lbl { paint-order:stroke; stroke:#fff; stroke-width:3px; stroke-linejoin:round; pointer-events:none; } .lbl.sel { fill:var(--red); font-weight:700; }
.pt.sel circle, .pt:focus-visible circle { stroke:var(--red); stroke-width:3; }
.pt.sel .eb { stroke:var(--red); stroke-width:1.5; }
.scout { padding:24px 22px; border-top:6px solid var(--navy); box-shadow:6px 6px 0 var(--red); position:sticky; top:16px; }
.scout .row { display:flex; justify-content:space-between; }
.scout h3 { font-family:var(--serif); font-weight:400; font-size:40px; line-height:1.05; margin:18px 0 10px; }
.badge { display:inline-block; border:1.5px solid var(--red); color:var(--red); font-size:10.5px; font-weight:700; letter-spacing:.08em; padding:2px 7px; }
.badge.calm { border-color:var(--navy); color:var(--navy); }
.two { display:grid; grid-template-columns:1fr 1fr; margin:22px 0 18px; border-bottom:1px solid var(--line); padding-bottom:16px; }
.two > div + div { border-left:1px solid var(--line); padding-left:16px; }
.big { font-family:var(--mono); font-weight:500; font-size:24px; color:var(--ink); margin:4px 0 2px; }
.meter { position:relative; height:9px; background:#e7e9ee; margin:8px 0; } .meter i { position:absolute; inset:0 auto 0 0; background:var(--navy); }
.meter u { position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--red); }
.scout .small { font-size:12.5px; color:#333a47; } .scout .small b { color:var(--red); }
.scout ul { list-style:none; padding:0; margin:6px 0 0; font-family:var(--mono); font-size:11.5px; color:#333a47; }
.scout hr { border:0; border-top:1px solid var(--line); margin:18px 0 14px; }
.quote { font-family:var(--serif); font-style:italic; font-size:19px; line-height:1.25; margin:8px 0 0; }
/* defense */
.dsplit { display:grid; grid-template-columns:330px minmax(0, 1fr); gap:24px; align-items:start; }
.teams { display:grid; grid-template-columns:repeat(5, 1fr); gap:8px; margin-top:10px; }
.teams button { font-family:var(--mono); font-size:12px; padding:11px 0; background:#fff; color:var(--navy); border:1px solid var(--navy); }
.teams button[aria-pressed=true] { background:var(--navy); color:#fff; } .teams button:hover { background:#e6ebf5; }
.teams button[aria-pressed=true]:hover { background:var(--navy); }
.strip { background:#ecebe6; padding:16px 16px 12px; margin-top:16px; }
.dpanel { padding:24px 26px; } .dpanel .title { display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 18px; }
.dpanel .abbr { font-family:var(--serif); font-size:64px; line-height:1; }
.dpanel .say { font-family:var(--serif); font-style:italic; font-size:25px; } .dpanel .say b { color:var(--red); font-weight:400; }
.drow { display:grid; grid-template-columns:150px minmax(0, 1fr) 96px; gap:16px; align-items:center; padding:13px 0; border-top:1px solid var(--line); }
.drow:first-of-type { margin-top:18px; } .drow strong { display:block; font-size:15px; }
.dbar { position:relative; height:26px; background:#eceae4; } .dbar i { position:absolute; inset:0 auto 0 0; background:var(--navy); }
.dbar.hot i { background:var(--red); } .dbar u { position:absolute; top:-5px; bottom:-5px; width:2px; background:#555; }
.dval { text-align:right; } .dval strong { font-size:19px; }
.key { display:flex; flex-wrap:wrap; gap:18px; font-size:12px; color:#333a47; border-top:1px solid var(--line); padding-top:14px; }
.sw { display:inline-block; width:14px; height:9px; margin-right:6px; }
/* play */
.psplit { display:grid; grid-template-columns:minmax(0, 1fr) 290px; gap:24px; align-items:start; }
.ptitle { font-family:var(--serif); font-style:italic; font-size:27px; line-height:1.15; margin:0 0 4px; }
.field { border:1px solid var(--navy); padding:5px; background:#fff; margin-top:10px; }
.side { padding:22px; } .sepnum { font-family:var(--serif); font-size:92px; line-height:1; margin:4px 0 14px; } .sepnum small { font-size:.3em; }
.spark { display:flex; align-items:flex-end; gap:2px; height:74px; margin:12px 0 6px; } .spark i { flex:1; background:#d6d6d6; min-height:2px; }
.spark i.on { background:var(--red); } .spark i.past { background:#9aa3b2; }
.axis { display:flex; justify-content:space-between; }
input[type=range] { width:100%; accent-color:var(--red); margin:10px 0 12px; }
.play-btn { width:100%; background:var(--navy); color:#fff; border:0; padding:14px; font-size:15px; }
.play-btn:hover { background:#123a7d; }
button:focus-visible, .play-btn:focus-visible, input:focus-visible { outline:3px solid var(--gold); outline-offset:2px; }
footer { margin-top:70px; border-top:3px solid var(--ink); padding-top:18px; font-size:13px; color:#333a47; max-width:900px; }
footer a { font-family:var(--mono); font-size:12px; color:var(--navy); font-weight:500; }
@media (max-width: 920px) { .split, .dsplit, .psplit { grid-template-columns:1fr; } .scout { position:static; }
  .hero { padding:30px 22px 34px; } .drow { grid-template-columns:110px minmax(0, 1fr) 80px; } }
</style></head><body><main>

<header class="hero">
  <div class="top"><div class="cap"><span class="tag">Big Data Bowl</span>2021 season · Weeks 1–8 · Man-coverage dropbacks</div>
    <a id="repo" href="#">CODE AND WRITE-UP</a></div>
  <h1>Mismatch Market.</h1>
  <p class="q">Are defenses putting linebackers on the wrong tight ends and running backs?</p>
  <p class="lead">A linebacker on a tight end costs the defense <span id="lead-edge"></span> yards of separation. Offenses can buy that
  matchup by lining the tight end up inline, and defenses are not steering linebackers away from the players who win it.</p>
</header>

<div class="stats" id="stats"></div>

<section>
  <div class="head"><span class="no">01</span><div><h2>The market</h2>
    <p>Left means defenses give him more linebackers than normal for where he lines up. Up means he gets more space than normal
    when they do. The shaded corner is an offensive coordinator's target list, and a defensive coordinator's to-fix list.</p></div></div>
  <div class="split">
    <div class="card market">
      <div class="tabs" id="tabs" role="group" aria-label="Filter by position"></div>
      <div class="legend"><span><span class="dot" style="background:var(--navy)"></span>Tight end</span>
        <span><span class="dot" style="background:var(--gold)"></span>Running back</span>
        <span>│ ± 1 standard error</span><span>Dot size = linebacker matchups</span></div>
      <div id="scatter"></div>
    </div>
    <aside class="card scout" id="scout" aria-live="polite"></aside>
  </div>
  <p class="note">Bars are one standard error. One player's Punishment over eight weeks is noisy, so read the corner as candidates,
  not a ranking. Hover, tap or tab through the dots to swap players.</p>
</section>

<section>
  <div class="head"><span class="no">02</span><div><h2>Scout a defense</h2>
    <p>How often each defense covers tight ends and running backs with a linebacker, by alignment, against the league. A bar in red
    sits 15 or more points above average: a matchup that defense hands out more than most.</p></div></div>
  <div class="dsplit">
    <div><div class="cap">Pick a defense</div><div class="teams" id="teams"></div>
      <div class="strip"><div class="cap">Linebacker tax, all <span id="ndef"></span> · yd per game</div><div id="strip"></div></div></div>
    <div class="card dpanel" id="dpanel" aria-live="polite"></div>
  </div>
</section>

<section>
  <div class="head"><span class="no">03</span><div><h2>What it looks like</h2>
    <p>One completed pass from the shaded corner, with the matched defender highlighted and the separation updating every tenth
    of a second.</p></div></div>
  <p class="ptitle" id="ptitle"></p><div class="mono" id="pdesc"></div>
  <div class="psplit">
    <div><div class="field" id="field"></div><div class="legend" id="plegend"></div></div>
    <div class="card side">
      <div class="cap">Separation</div><div class="sepnum"><span id="sep"></span><small>yd</small></div>
      <div class="mono" id="phase"></div>
      <div class="spark" id="spark"></div><div class="axis mono"><span id="t0"></span><span>snap</span><span id="t1"></span></div>
      <div class="row" style="display:flex;justify-content:space-between;margin-top:26px"><span class="cap">Time from snap</span><span class="mono" id="tnow"></span></div>
      <input type="range" id="slider" min="0" value="0" aria-label="Time from snap">
      <button class="play-btn" id="playbtn">▶ Play</button>
    </div>
  </div>
</section>

<footer>Matchups are inferred from player tracking: each route runner is paired with one coverage defender by smallest average
distance from snap to throw, not charted. Data: NFL Next Gen Stats tracking and PFF scouting from the NFL Big Data Bowl 2023
release.<br><a id="repo2" href="#"></a></footer>
</main>

<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = v => Math.round(v * 100) + '%';
const sgn = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);

/* ---------- header and stat cards ---------- */
$('repo').href = $('repo2').href = D.repo; $('repo2').textContent = D.repo.replace('https://', '');
$('lead-edge').textContent = D.stats.edge.toFixed(1); $('ndef').textContent = D.stats.n_def;
$('stats').innerHTML = `
  <div class="stat red"><div class="cap">The linebacker tax</div><b>+${D.stats.edge.toFixed(2)} yd</b>
    <p>Extra separation for a tight end covered by a linebacker</p><div class="mono">95% interval ${D.stats.lo.toFixed(2)} – ${D.stats.hi.toFixed(2)}</div></div>
  <div class="stat"><div class="cap">How offenses buy it</div><b>${pct(D.stats.inline)} <i>vs</i> ${pct(D.stats.wide)}</b>
    <p>How often a tight end draws a linebacker lined up inline vs split wide</p><div class="mono">League average, all ${D.stats.n_def} defenses</div></div>
  <div class="stat"><div class="cap">Do defenses adjust?</div><b>r = ${D.stats.r.toFixed(2)}</b>
    <p>Link between how often a player draws a linebacker and how much he beats one</p><div class="mono">${D.stats.n} players · effectively none</div></div>`;

/* ---------- 1. market ---------- */
const W = 720, H = 560, M = {l: 58, r: 14, t: 16, b: 46}, XR = [-0.52, 0.42], YR = [-1.45, 1.75];
const sx = v => M.l + (v - XR[0]) / (XR[1] - XR[0]) * (W - M.l - M.r);
const sy = v => H - M.b - (v - YR[0]) / (YR[1] - YR[0]) * (H - M.t - M.b);
const COL = {TE: '#0a2a5e', RB: '#f0b429'};
function quadrant(p) {
  return p.respect < 0 ? (p.punish > 0 ? 'Under-respected and winning' : 'Linebackers handle him')
                       : (p.punish > 0 ? 'Respected, still winning' : 'Over-respected');
}
function drawScatter() {
  let g = `<rect x="${sx(XR[0])}" y="${sy(YR[1])}" width="${sx(0) - sx(XR[0])}" height="${sy(0) - sy(YR[1])}" fill="#fbeeee"/>`;
  for (const t of [-1, -0.5, 0.5, 1, 1.5]) g += `<line x1="${M.l}" x2="${W - M.r}" y1="${sy(t)}" y2="${sy(t)}" stroke="#e6e9ef"/>
    <text x="${M.l - 8}" y="${sy(t) + 4}" text-anchor="end" font-size="11" fill="#59616f" font-family="IBM Plex Mono, monospace">${sgn(t, 1)}</text>`;
  for (const t of [-0.4, -0.2, 0, 0.2, 0.4]) g += `<text x="${sx(t)}" y="${H - M.b + 18}" text-anchor="middle" font-size="11" fill="#59616f" font-family="IBM Plex Mono, monospace">${t === 0 ? '0' : sgn(t, 1)}</text>`;
  g += `<text x="${M.l - 8}" y="${sy(0) + 4}" text-anchor="end" font-size="11" fill="#59616f" font-family="IBM Plex Mono, monospace">0</text>
    <line x1="${M.l}" x2="${W - M.r}" y1="${sy(0)}" y2="${sy(0)}" stroke="#7b8494" stroke-dasharray="4 4"/>
    <line x1="${sx(0)}" x2="${sx(0)}" y1="${M.t}" y2="${H - M.b}" stroke="#7b8494" stroke-dasharray="4 4"/>
    <line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="#0a2a5e" stroke-width="1.5"/>
    <line x1="${M.l}" x2="${M.l}" y1="${M.t}" y2="${H - M.b}" stroke="#0a2a5e" stroke-width="1.5"/>`;
  const cap = (x, y, a, t, c) => `<text x="${x}" y="${y}" text-anchor="${a}" font-size="10.5" font-weight="700" letter-spacing="1" fill="${c}">${t}</text>`;
  g += cap(M.l + 8, M.t + 14, 'start', 'UNDER-RESPECTED, WINNING', '#d0181f') + cap(W - M.r - 8, M.t + 14, 'end', 'RESPECTED, STILL WINNING', '#59616f')
     + cap(M.l + 8, H - M.b - 9, 'start', 'LINEBACKERS HANDLE HIM', '#59616f') + cap(W - M.r - 8, H - M.b - 9, 'end', 'OVER-RESPECTED', '#59616f')
     + cap(M.l, H - 6, 'start', '← MORE LINEBACKERS THAN NORMAL', '#0a2a5e') + cap(W - M.r, H - 6, 'end', 'FEWER →', '#0a2a5e')
     + `<text transform="translate(13 ${(H - M.b + M.t) / 2}) rotate(-90)" text-anchor="middle" font-size="10.5" font-weight="700" letter-spacing="1" fill="#0a2a5e">PUNISHMENT · EXTRA YD OF SEPARATION VS LB</text>`;
  // name labels: nudge down when two would collide
  const placed = [];
  const lab = p => { const flip = p.respect > 0.2; let x = sx(p.respect) + (flip ? -11 : 11), y = sy(p.punish) + 4;
    while (placed.some(q => Math.abs(q[0] - x) < 95 && Math.abs(q[1] - y) < 12)) y += 12;
    placed.push([x, y]); return [x, y, flip ? 'end' : 'start']; };
  let labels = '';
  const order = D.players.map((p, i) => i).sort((a, b) => D.players[b].n_lb - D.players[a].n_lb);   // small dots on top
  for (const i of order) { const p = D.players[i], x = sx(p.respect), y = sy(p.punish), r = 3 + Math.sqrt(p.n_lb) * 0.95;
    const y1 = Math.max(M.t, sy(p.punish + p.se)), y2 = Math.min(H - M.b, sy(p.punish - p.se));
    if (p.label) { const [lx, ly, anchor] = lab(p); labels += `<text class="lbl ${p.pos}" data-i="${i}" x="${lx}" y="${ly}" text-anchor="${anchor}" font-size="10.5" fill="#131722">${esc(p.name)}</text>`; }
    g += `<g class="pt ${p.pos}" data-i="${i}" tabindex="0" role="button" aria-label="${esc(p.name)}, ${p.pos}, respect ${p.respect.toFixed(2)}, punishment ${p.punish.toFixed(2)} yards">
      <line class="eb" x1="${x}" x2="${x}" y1="${y1}" y2="${y2}" stroke="#b9bfca"/><circle cx="${x}" cy="${y}" r="${r.toFixed(1)}" fill="${COL[p.pos]}"/></g>`; }
  $('scatter').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="group" aria-label="Respect versus Punishment for ${D.players.length} players">${g}${labels}</svg>`;
  for (const el of document.querySelectorAll('.pt')) { const pick = () => selectPlayer(+el.dataset.i);
    el.addEventListener('mouseenter', pick); el.addEventListener('focus', pick); el.addEventListener('click', pick); }
}
function selectPlayer(i) {
  const p = D.players[i];
  for (const el of document.querySelectorAll('.pt, .lbl')) el.classList.toggle('sel', +el.dataset.i === i);
  const q = quadrant(p), hot = q === 'Under-respected and winning' && !p.blocker;
  const align = p.align.map(a => `<li>${a.a.padEnd(7)}${String(a.n).padStart(2)} rt · LB ${pct(a.lb).padStart(4)}${a.sep == null ? '' : ' · ' + a.sep.toFixed(1) + ' yd'}</li>`).join('');
  $('scout').innerHTML = `<div class="row"><span class="cap">Scouting card</span><span class="mono">${p.pos === 'TE' ? 'Tight end' : 'Running back'} · ${p.team}</span></div>
    <h3>${esc(p.name)}</h3><span class="badge ${hot ? '' : 'calm'}">${p.blocker ? 'BLOCKING TIGHT END · LEFT ALONE ON PURPOSE' : q.toUpperCase()}</span>
    <div class="two"><div><div class="cap">Respect</div><div class="big">${sgn(p.respect, 2)}</div></div>
      <div><div class="cap">Punishment</div><div class="big">${sgn(p.punish, 2)} yd</div><div class="mono">± ${p.se.toFixed(2)}</div></div></div>
    <div class="row small"><span><strong>Drew a linebacker</strong></span><strong>${pct(p.lb_rate)}</strong></div>
    <div class="meter"><i style="width:${p.lb_rate * 100}%"></i><u style="left:${p.exp_lb * 100}%"></u></div>
    <div class="small">Normal for his alignment: <b>${pct(p.exp_lb)}</b> · ${p.n} routes, ${p.n_lb} vs LB</div>
    <hr><div class="cap">By alignment · routes, linebacker rate, space vs LB</div><ul style="white-space:pre">${align}</ul>
    ${p.best ? `<hr><div class="cap">Best catch vs a linebacker</div><div class="mono" style="margin-top:6px">Week ${p.best.week} · vs ${esc(p.best.dname)}</div><p class="quote">${esc(p.best.desc)}</p>` : ''}`;
}
function setTab(pos) {
  for (const b of $('tabs').children) b.setAttribute('aria-pressed', b.dataset.pos === pos);
  for (const el of document.querySelectorAll('.pt, .lbl')) el.classList.toggle('hide', pos !== 'all' && !el.classList.contains(pos));
}
const count = pos => D.players.filter(p => p.pos === pos).length;
$('tabs').innerHTML = [['all', `All · ${D.players.length}`], ['TE', `Tight ends · ${count('TE')}`], ['RB', `Running backs · ${count('RB')}`]]
  .map(([k, t]) => `<button data-pos="${k}" aria-pressed="${k === 'all'}">${t}</button>`).join('');
for (const b of $('tabs').children) b.addEventListener('click', () => setTab(b.dataset.pos));
drawScatter();
selectPlayer(Math.max(0, D.players.findIndex(p => p.name === D.start)));

/* ---------- 2. defense ---------- */
const TEAMS = Object.keys(D.defense.teams).sort();
function drawStrip(sel) {
  const w = 300, x = v => 16 + (v + 8.5) / 17 * (w - 32);
  let g = `<line x1="16" x2="${w - 16}" y1="22" y2="22" stroke="#555"/>`;
  for (const t of TEAMS) if (t !== sel) g += `<circle cx="${x(D.defense.teams[t].tax)}" cy="22" r="4" fill="#8c8f96" stroke="#ecebe6"/>`;
  g += `<circle cx="${x(D.defense.teams[sel].tax)}" cy="22" r="7" fill="#d0181f"/>`;
  g += `<text x="16" y="50" font-size="10" fill="#59616f" font-family="IBM Plex Mono, monospace">−8 saves</text><text x="${x(0)}" y="50" text-anchor="middle" font-size="10" fill="#59616f" font-family="IBM Plex Mono, monospace">0</text><text x="${w - 16}" y="50" text-anchor="end" font-size="10" fill="#59616f" font-family="IBM Plex Mono, monospace">pays +8</text>`;
  $('strip').innerHTML = `<svg viewBox="0 0 ${w} 56" role="img" aria-label="${sel} linebacker tax ${D.defense.teams[sel].tax} yards per game">${g}</svg>`;
}
function selectTeam(t) {
  const d = D.defense.teams[t];
  for (const b of $('teams').children) b.setAttribute('aria-pressed', b.textContent === t);
  const rows = d.rows.map((r, i) => { const lg = D.defense.league[i];
    if (r.rate == null) return `<div class="drow"><div><strong>${r.label}</strong><span class="mono">no routes</span></div><div class="dbar"><u style="left:${lg * 100}%"></u></div><div class="dval mono">–</div></div>`;
    const diff = Math.round((r.rate - lg) * 100);
    return `<div class="drow"><div><strong>${r.label}</strong><span class="mono">n = ${r.n} routes</span></div>
      <div class="dbar ${diff >= 15 ? 'hot' : ''}"><i style="width:${r.rate * 100}%"></i><u style="left:${lg * 100}%"></u></div>
      <div class="dval"><strong>${pct(r.rate)}</strong><div class="mono">${diff >= 0 ? '+' : '−'}${Math.abs(diff)} vs avg</div></div></div>`; }).join('');
  $('dpanel').innerHTML = `<div class="title"><span class="abbr">${t}</span><span class="say">${d.tax >= 0 ? 'pays' : 'saves'} <b>${sgn(d.tax, 1)} yd</b> per game in linebacker tax</span></div>
    <div class="mono">Rank ${d.rank} of ${TEAMS.length} · 1 = pays most</div>${rows}
    <div class="key"><span><span class="sw" style="background:var(--navy)"></span>${t}</span><span><span class="sw" style="background:var(--red)"></span>15+ pts above league</span>
      <span>│ League average</span><span>n = routes faced; small n is thin evidence</span></div>`;
  drawStrip(t);
}
$('teams').innerHTML = TEAMS.map(t => `<button aria-pressed="false">${t}</button>`).join('');
for (const b of $('teams').children) b.addEventListener('click', () => selectTeam(b.textContent));
selectTeam(D.defense.start);

/* ---------- 3. play ---------- */
const P = D.play, S = 14, FW = (P.x1 - P.x0) * S, FH = (P.y1 - P.y0) * S;
const fx = x => (x - P.x0) * S, fy = y => (P.y1 - y) * S;
(function drawField() {
  let g = '';
  for (let x = P.x0; x < P.x1; x += 5) g += `<rect x="${fx(x)}" y="0" width="${5 * S}" height="${FH}" fill="${(x / 5) % 2 ? '#2f7d32' : '#388e3c'}"/>`;
  for (let x = P.x0; x <= P.x1; x += 5) { g += `<line x1="${fx(x)}" x2="${fx(x)}" y1="0" y2="${FH}" stroke="#fff" stroke-opacity=".75" stroke-width="${x % 10 ? 1 : 2}"/>`;
    if (x % 10 === 0 && x > P.x0 && x < P.x1 && x >= 20 && x <= 100) { const n = Math.min(x - 10, 110 - x);
      g += `<text x="${fx(x)}" y="${FH - 26}" text-anchor="middle" font-size="26" font-weight="700" fill="#fff" fill-opacity=".7" letter-spacing="6">${n}</text>
            <text transform="translate(${fx(x)} 44) rotate(180)" text-anchor="middle" font-size="26" font-weight="700" fill="#fff" fill-opacity=".7" letter-spacing="6">${n}</text>`; } }
  for (const h of [23.58, 29.72]) if (h > P.y0 && h < P.y1) for (let x = P.x0 + 1; x < P.x1; x++) g += `<line x1="${fx(x)}" x2="${fx(x)}" y1="${fy(h) - 4}" y2="${fy(h) + 4}" stroke="#fff" stroke-opacity=".6"/>`;
  g += `<line x1="${fx(P.los)}" x2="${fx(P.los)}" y1="0" y2="${FH}" stroke="#2f6bff" stroke-width="3"/>
        <rect x="${fx(P.los) + 6}" y="96" width="128" height="18" fill="#2f6bff"/><text x="${fx(P.los) + 12}" y="109" font-size="10.5" font-weight="700" fill="#fff" letter-spacing="1">LINE OF SCRIMMAGE</text>`;
  $('field').innerHTML = `<svg viewBox="0 0 ${FW} ${FH}" role="img" aria-label="Animated player positions for the example play">${g}<g id="live"></g></svg>`;
  $('ptitle').textContent = P.title; $('pdesc').textContent = P.desc;
  $('plegend').innerHTML = `<span><span class="dot" style="background:#e4572e;border-color:#3b1206"></span>${P.off} offense</span>
    <span><span class="dot" style="background:#fff;border-color:#1565c0"></span>${P.dfn} defense</span><span><span class="dot" style="background:#8b4513;border-color:#3b1206;width:8px;height:8px"></span>Ball</span>
    <span><span class="dot" style="background:#ff6b35;border-color:#fff;outline:2px solid #111"></span>${esc(P.rec_full)}</span><span><span class="dot" style="background:#1565c0;border-color:#d0181f"></span>${esc(P.cov_full)}</span>`;
})();
const maxGap = Math.max(...P.frames.map(f => f.gap));
$('spark').innerHTML = P.frames.map(f => `<i style="height:${Math.max(3, f.gap / maxGap * 100)}%"></i>`).join('');
$('slider').max = P.frames.length - 1; $('t0').textContent = sgn(P.frames[0].t, 1) + ' s'; $('t1').textContent = sgn(P.frames[P.frames.length - 1].t, 1) + ' s';
function tagAt(x, y, text, fill, dx) { const w = text.length * 6.6 + 14;
  return `<rect x="${x + dx - (dx < 0 ? w : 0)}" y="${y + 12}" width="${w}" height="18" fill="${fill}"/><text x="${x + dx - (dx < 0 ? w : 0) + 7}" y="${y + 25}" font-size="11" font-weight="700" fill="#fff">${esc(text)}</text>`; }
function drawFrame(i) {
  const f = P.frames[i], dots = (a, fill, stroke) => a.map(([x, y]) => `<circle cx="${fx(x)}" cy="${fy(y)}" r="5.5" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`).join('');
  const trail = k => P.frames.slice(0, i + 1).filter(g => g.phase !== 'pre').map(g => `${fx(g[k][0])},${fy(g[k][1])}`).join(' ');
  const [rx, ry] = [fx(f.r[0]), fy(f.r[1])], [dx, dy] = [fx(f.d[0]), fy(f.d[1])], left = rx <= dx;
  $('live').innerHTML = `<polyline points="${trail('r')}" fill="none" stroke="#ffb199" stroke-width="3" stroke-linecap="round"/>
    <polyline points="${trail('d')}" fill="none" stroke="#9cc4f5" stroke-width="3" stroke-linecap="round"/>
    ${dots(f.off, '#e4572e', '#3b1206')}${dots(f.dfn, '#fff', '#1565c0')}
    <line x1="${rx}" y1="${ry}" x2="${dx}" y2="${dy}" stroke="#fff" stroke-width="2" stroke-dasharray="6 5"/>
    ${f.ball.map(([x, y]) => `<ellipse cx="${fx(x)}" cy="${fy(y)}" rx="5.5" ry="3.5" fill="#8b4513" stroke="#3b1206"/>`).join('')}
    <circle cx="${rx}" cy="${ry}" r="10" fill="#ff6b35" stroke="#fff" stroke-width="2.5"/><circle cx="${rx}" cy="${ry}" r="12" fill="none" stroke="#111" stroke-width="1.5"/>
    <circle cx="${dx}" cy="${dy}" r="10" fill="#1565c0" stroke="#d0181f" stroke-width="3"/>
    ${tagAt(rx, ry, P.rec, '#111', left ? -4 : 4)}${tagAt(dx, dy, P.cov, '#d0181f', left ? 4 : -4)}`;
  $('sep').textContent = f.gap.toFixed(1);
  $('phase').textContent = f.phase === 'pre' ? 'pre-snap' : f.phase === 'live' ? f.t.toFixed(1) + ' s after snap' : 'ball in the air';
  $('tnow').textContent = sgn(f.t, 1) + ' s'; $('slider').value = i;
  [...$('spark').children].forEach((el, j) => { el.className = j === i ? 'on' : j < i ? 'past' : ''; });
}
let cur = 0, timer = null;
function stop() { clearInterval(timer); timer = null; $('playbtn').textContent = '▶ Play'; }
$('playbtn').addEventListener('click', () => {
  if (timer) return stop();
  if (cur >= P.frames.length - 1) cur = 0;
  $('playbtn').textContent = '❚❚ Pause';
  timer = setInterval(() => { cur++; drawFrame(cur); if (cur >= P.frames.length - 1) stop(); }, 100);
});
$('slider').addEventListener('input', e => { stop(); cur = +e.target.value; drawFrame(cur); });
drawFrame(0);
</script>
</body></html>
'''


if __name__ == '__main__':
    main()
