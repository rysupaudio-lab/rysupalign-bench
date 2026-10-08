#!/usr/bin/env python3
"""Audit suite R truth (diagnostic only; the bench scoring is NOT changed).

For every truth pair (tg, td) re-derive an independent event time in the double from the same
human-corrected CSD f0 (no audio, no aligner):
  note change: the guide's step a -> b (medians of 9 frames before/after tg); the double's
               crossing time = the time nearest tg + (td - tg)... no: nearest to tg, within +-250 ms,
               where the double's contour crosses the midpoint (a+b)/2 in the step's direction and stays
               on the new side for >= 40 ms.
  onset:       the double's nearest voicing onset (any length) to tg within +-250 ms.
Reports how often the independent estimate disagrees with td, and lists the large-offset events.
Also checks for duplicate f0 annotations across singers (identical contours).
"""
import os, sys, json
import numpy as np
_H = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(_H), 'runners'), os.path.join(os.path.dirname(_H), 'rab')]
import method_lib as ML, rabpaths
import build_rab as BR

CHORAL = os.environ.get('CSD_DIR', 'datasets/ChoralSingingDataset')   # unzipped Choral Singing Dataset


def f0(piece, sec, k):
    d = np.loadtxt(f'{CHORAL}/CSD_{piece}_{sec}_{k}.f0'); t, f = d[:, 0], d[:, 1]
    return t, np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1e-9) / 440), np.nan)


def crossing(t, s, tg, a, b, win=0.25):
    up = b > a; mid = (a + b) / 2
    lo, hi = np.searchsorted(t, tg - win), np.searchsorted(t, tg + win)
    side = (s > mid) if up else (s < mid)
    cands = []
    for i in range(max(lo, 1), hi):
        if side[i] and not side[i - 1] and np.isfinite(s[i]):
            k = i + 7   # >= 40 ms on the new side (5.8 ms hop)
            if k < len(s) and side[i:k].all(): cands.append(t[i])
    return min(cands, key=lambda x: abs(x - tg)) if cands else None


def onset(t, s, tg, win=0.25):
    v = np.isfinite(s); idx = np.where(v[1:] & ~v[:-1])[0] + 1
    c = [t[i] for i in idx if abs(t[i] - tg) <= win]
    return min(c, key=lambda x: abs(x - tg)) if c else None


def main():
    rows = []; dup = []
    for cid in ML.manifest('R'):
        _, piece, sec, gd = cid.split('_')
        tg_, sg = f0(piece, sec, gd[0]); td_, sd = f0(piece, sec, gd[1])
        n = min(len(sg), len(sd)); both = np.isfinite(sg[:n]) & np.isfinite(sd[:n])
        same = np.mean(np.abs(sg[:n][both] - sd[:n][both]) < 1e-3) if both.any() else 0
        if same > 0.5: dup.append((cid, same))
        eg = BR.f0_events(f'{CHORAL}/CSD_{piece}_{sec}_{gd[0]}.f0')
        pairs = np.asarray(json.load(open(f'{rabpaths.CASES}/{cid}/truth.json'))['pairs'])
        for tg, td in pairs:
            k = min(eg, key=lambda e: abs(e[1] - tg))
            if k[0] == 'note':
                i = np.searchsorted(tg_, tg)
                a, b = np.nanmedian(sg[max(0, i - 9):i]), np.nanmedian(sg[i:i + 9])
                cg = crossing(tg_, sg, tg, a, b, 0.06)          # same estimator on the guide side
                cd = crossing(td_, sd, tg, a, b)
                est = None if cg is None or cd is None else tg + (cd - cg)
            else:
                est = onset(td_, sd, tg)
            rows.append((cid, k[0], tg, td, est))
    print('duplicate-annotation check (share of co-voiced frames with identical f0):', dup or 'none > 50%')
    kind = np.array([r[1] for r in rows]); off = np.array([(r[3] - r[2]) * 1000 for r in rows])
    est = np.array([np.nan if r[4] is None else (r[4] - r[2]) * 1000 for r in rows])
    ok = np.isfinite(est); dx = np.abs(est - off)
    print(f'events {len(rows)}; independent estimate found for {ok.mean() * 100:.1f}%')
    for k in ('on', 'note'):
        s = ok & (kind == k)
        print(f'  {k:4s}: |indep - truth| median {np.median(dx[s]):.1f} ms, mean {dx[s].mean():.1f}, '
              f'<=10 ms {100 * (dx[s] <= 10).mean():.1f}%, >50 ms {100 * (dx[s] > 50).mean():.1f}%')
    for a, b in ((0, 50), (50, 100), (100, 150), (150, 260)):
        s = ok & (np.abs(off) >= a) & (np.abs(off) < b)
        print(f'  truth |offset| [{a},{b}) ms: n {s.sum()}  indep agrees within 20 ms {100 * (dx[s] <= 20).mean():.1f}%, '
              f'indep |offset| median {np.median(np.abs(est[s])):.1f} ms')
    # aligner (base_py, from method_diag) against the event truth vs the crossing truth on note events
    if os.path.exists(f'{rabpaths.RESULTS}/method_diag_base_py.npz'):
        Z = np.load(f'{rabpaths.RESULTS}/method_diag_base_py.npz')
        pred = np.concatenate([Z[f'{c}|pred'] for c in ML.manifest('R')])
        s = ok & (kind == 'note')
        e1 = np.abs(pred[s] - np.array([r[3] for r in rows])[s]) * 1000
        e2 = np.abs(pred[s] - (np.array([r[2] for r in rows])[s] + est[s] / 1000)) * 1000
        print(f'  base_py on note events: vs event truth mean {e1.mean():.1f} med {np.median(e1):.1f} <=20 {100*(e1<=20).mean():.1f}% | '
              f'vs crossing truth mean {e2.mean():.1f} med {np.median(e2):.1f} <=20 {100*(e2<=20).mean():.1f}%')
        r1 = np.abs(off[s]); r2 = np.abs(est[s])
        print(f'  raw on note events: vs event truth mean {r1.mean():.1f} med {np.median(r1):.1f} | vs crossing truth mean {r2.mean():.1f} med {np.median(r2):.1f}')
    np.savez(f'{rabpaths.RESULTS}/method_truth_audit.npz', cid=np.array([r[0] for r in rows]), kind=kind,
             tg=np.array([r[2] for r in rows]), td=np.array([r[3] for r in rows]), est=est)


if __name__ == '__main__':
    main()
