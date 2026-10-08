#!/usr/bin/env python3
"""Diagnose suite R errors of a method variant (default base_py = documented v6+mel x2 gate baseline).

DIAGNOSTICS ONLY: re-derives each truth event's kind (onset / note change) from the CSD f0 files with
build_rab.f0_events, and estimates truth noise with an independent local f0-contour cross-correlation
(from the same human-corrected f0). Nothing here feeds any method.
usage: method_diag.py [variant]   -> prints tables, saves results/method_diag_<variant>.npz
"""
import os, sys, json, concurrent.futures as cf
import numpy as np
_H = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(_H), 'runners'), os.path.join(os.path.dirname(_H), 'rab')]
import method_lib as ML, rabpaths

CHORAL = os.environ.get('CSD_DIR', 'datasets/ChoralSingingDataset')   # unzipped Choral Singing Dataset


def csd_f0(piece, sec, k):
    d = np.loadtxt(f'{CHORAL}/CSD_{piece}_{sec}_{k}.f0')
    t, f = d[:, 0], d[:, 1]
    st = np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1e-9) / 440), np.nan)
    return t, st


def xcorr_lag(tg, sg, td, sd, t0, t1, maxlag=0.25):
    """Best lag L (double = guide + L) of the guide's semitone contour in [t0, t1] against the double's,
    on a 5 ms grid, by mean |diff| over frames voiced in both (needs >= 60% overlap)."""
    grid = np.arange(t0, t1, 0.005)
    a = np.interp(grid, tg, np.nan_to_num(sg, nan=-999)); va = np.interp(grid, tg, np.isfinite(sg).astype(float)) > 0.99
    best, bl = None, None
    for L in np.arange(-maxlag, maxlag + 1e-9, 0.005):
        b = np.interp(grid + L, td, np.nan_to_num(sd, nan=-999)); vb = np.interp(grid + L, td, np.isfinite(sd).astype(float)) > 0.99
        ok = va & vb
        if ok.mean() < 0.6: continue
        # voicing agreement counts as well: mismatched voicing costs 1 st per frame
        cost = (np.minimum(np.abs(a[ok] - b[ok]), 3).sum() + (va ^ vb).sum()) / len(grid)
        if best is None or cost < best: best, bl = cost, L
    return bl, best


def work(args):
    cid, var = args
    import method_variants as MV, build_rab as BR
    z = ML.load(cid)
    g, d = MV.VARIANTS[var](z)
    pred = np.interp(z['tg'], g, d)
    _, piece, sec, gd = cid.split('_')
    eg = BR.f0_events(f'{CHORAL}/CSD_{piece}_{sec}_{gd[0]}.f0')
    kinds = []
    for t in z['tg']:
        k = min(eg, key=lambda e: abs(e[1] - t)); kinds.append(0 if k[0] == 'on' else 1)
    # phrase position: time since the latest guide onset event (any onset, matched or not)
    ons = np.array([e[1] for e in eg if e[0] == 'on'])
    since = np.array([t - ons[ons <= t + 1e-6].max() if (ons <= t + 1e-6).any() else 9.9 for t in z['tg']])
    tg_, sg = csd_f0(piece, sec, gd[0]); td_, sd = csd_f0(piece, sec, gd[1])
    xl, xc = [], []
    for t, tdd in zip(z['tg'], z['td']):
        L, c = xcorr_lag(tg_, sg, td_, sd, t - 0.15, t + 0.15)
        xl.append(np.nan if L is None else L); xc.append(np.nan if c is None else c)
    return cid, dict(tg=z['tg'], td=z['td'], pred=pred, kind=np.array(kinds), since=since,
                     xlag=np.array(xl), xcost=np.array(xc), unison=z.get('unison'), interval=z.get('interval'))


def main():
    var = sys.argv[1] if len(sys.argv) > 1 else 'base_py'
    ids = ML.manifest('R')
    with cf.ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(work, [(c, var) for c in ids]))
    np.savez(f'{rabpaths.RESULTS}/method_diag_{var.replace('/', '_')}.npz', **{f'{c}|{k}': np.asarray(v) for c, r in res.items() for k, v in r.items() if v is not None})
    cat = lambda k: np.concatenate([res[c][k] for c in ids])
    tg, td, pred, kind, since, xlag, xcost = (cat(k) for k in ('tg', 'td', 'pred', 'kind', 'since', 'xlag', 'xcost'))
    err = (pred - td) * 1000; ae = np.abs(err); raw = (tg - td) * 1000; araw = np.abs(raw)
    P = lambda a: f'mean {a.mean():5.1f} med {np.median(a):5.1f} <=20 {100 * (a <= 20).mean():4.1f}% n {len(a)}'
    print(f'variant {var}: all events {P(ae)} | raw {P(araw)}')
    print(f'signed error mean {err.mean():+.1f} ms, median {np.median(err):+.1f} ms (pred - truth; + = aligner says later)')
    for k, nm in ((0, 'onsets'), (1, 'note changes')):
        s = kind == k; print(f'  {nm:13s} {P(ae[s])} | raw {P(araw[s])} | signed med {np.median(err[s]):+.1f}')
    print('by |raw offset| (truth double - guide):')
    for a, b in ((0, 20), (20, 50), (50, 100), (100, 150), (150, 260)):
        s = (araw >= a) & (araw < b)
        print(f'  [{a:3d},{b:3d}) ms  {P(ae[s])}  share of total abs err {100 * ae[s].sum() / ae.sum():4.1f}%')
    print('by time since latest guide onset:')
    for a, b in ((0, 0.05), (0.05, 0.3), (0.3, 1), (1, 3), (3, 99)):
        s = (since >= a) & (since < b); print(f'  [{a:4.2f},{b:4.2f}) s {P(ae[s])}')
    print('error tail: share of total abs error from events with err > 100 ms:',
          f'{100 * ae[ae > 100].sum() / ae.sum():.1f}% ({(ae > 100).mean() * 100:.1f}% of events);',
          f'> 50 ms: {100 * ae[ae > 50].sum() / ae.sum():.1f}% ({(ae > 50).mean() * 100:.1f}% of events)')
    # truth noise: independent local f0-contour lag vs the event-time truth
    ok = np.isfinite(xlag) & (xcost < 0.5)
    dx = np.abs(xlag[ok] * 1000 - raw[ok])
    print(f'truth check (local +-150 ms f0-contour xcorr lag vs truth offset), {ok.sum()} events with a clean contour:')
    print(f'  |xcorr lag - truth offset|: mean {dx.mean():.1f} med {np.median(dx):.1f} p90 {np.percentile(dx, 90):.1f} ms; >50 ms on {100 * (dx > 50).mean():.1f}%')
    for k, nm in ((0, 'onsets'), (1, 'note changes')):
        s = kind[ok] == k; print(f'  {nm:13s} med {np.median(dx[s]):.1f} mean {dx[s].mean():.1f} >50ms {100 * (dx[s] > 50).mean():.1f}%')
    e_x = np.abs(pred[ok] - (tg[ok] + xlag[ok])) * 1000
    print(f'  aligner error vs xcorr-truth: {P(e_x)}  (vs event truth on same events: {P(ae[ok])})')
    big = dx > 50
    print(f'  events where truth and xcorr disagree >50 ms: aligner err vs event truth mean {ae[ok][big].mean():.1f}, vs xcorr {e_x[big].mean():.1f}')
    print('per case (sorted by err - raw):')
    rows = []
    for c in ids:
        r = res[c]; e = np.abs(r['pred'] - r['td']) * 1000; rw = np.abs(r['tg'] - r['td']) * 1000
        rows.append((e.mean() - rw.mean(), c, e.mean(), np.median(e), rw.mean(), (e > 100).mean() * 100, len(e), r['interval']))
    for d_, c, m, md, rw, big, n, iv in sorted(rows, reverse=True):
        print(f'  {c:18s} err {m:5.1f} med {md:5.1f} raw {rw:5.1f} d {d_:+6.1f} | >100ms {big:4.1f}% n {n} int {iv:+.2f}')


if __name__ == '__main__':
    main()
