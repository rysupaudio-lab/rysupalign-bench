#!/usr/bin/env python3
"""Method research harness (2026-10-07): in-process RAB evaluation of DTW *method* variants on
fixed features (learned v6 + mel, gated), so levers can be compared quickly and identically.

Per-case features are cached in $RAB_WORK/method_cache/<cid>.npz (10 ms grid):
  Al, Bl  learned v6 block (unit-norm, centre-offset placed, as learned_dtw.py)
  Am, Bm  fdtw log-mel block (unit-norm)
  fg, fd  yin f0 Hz (nan unvoiced, F0MAX 800 like the v6 runner), rg, rd frame rms
Never reads the CSD .f0 truth files (only truth.json, for scoring).
"""
import json, os, sys
import numpy as np
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rabpaths
CASES = rabpaths.CASES
MC = os.path.join(rabpaths.WORK, 'method_cache')
SR = 44100; HOP = 441; FS = 100.0


def manifest(suite):
    return [c['id'] for c in json.load(open(rabpaths.MANIFEST)) if c['suite'] == suite]


def build(cid):
    f = f'{MC}/{cid}.npz'
    if os.path.exists(f): return
    import fdtw, librosa, embed_onnx
    os.makedirs(MC, exist_ok=True)
    out = {}
    for side, nm in (('g', 'guide'), ('d', 'double')):
        S = 'A' if side == 'g' else 'B'
        p = f'{CASES}/{cid}/{nm}.wav'; x = fdtw.load(p)
        Lb = embed_onnx.learned_block(p, x, SR, HOP)
        Mb = fdtw.block('mel', p, x, HOP)
        n = min(Lb.shape[1], Mb.shape[1])
        out[S + 'l'] = Lb[:, :n].T.astype(np.float32)
        out[S + 'm'] = Mb[:, :n].T.astype(np.float32)
        y = librosa.resample(x, orig_sr=SR, target_sr=8000)
        fq = librosa.yin(y, fmin=70, fmax=800, sr=8000, frame_length=512, hop_length=80)
        r = librosa.feature.rms(y=y, frame_length=512, hop_length=80)[0][:len(fq)]
        fq[r < max(1e-4, np.percentile(r, 90) * 0.05)] = np.nan
        out['f' + side] = fq.astype(np.float32); out['r' + side] = r.astype(np.float32)
    np.savez(f + '.tmp.npz', **out); os.replace(f + '.tmp.npz', f)


def build_f0hi(cid):
    """yin f0 with a 1600 Hz ceiling (v7 gate fix: octave-up guides above 800 Hz), cached separately."""
    f = f'{MC}/{cid}_f0hi.npz'
    if os.path.exists(f): return
    import fdtw, librosa
    os.makedirs(MC, exist_ok=True)
    out = {}
    for side, nm in (('g', 'guide'), ('d', 'double')):
        x = fdtw.load(f'{CASES}/{cid}/{nm}.wav')
        y = librosa.resample(x, orig_sr=SR, target_sr=8000)
        fq = librosa.yin(y, fmin=70, fmax=1600, sr=8000, frame_length=512, hop_length=80)
        r = librosa.feature.rms(y=y, frame_length=512, hop_length=80)[0][:len(fq)]
        fq[r < max(1e-4, np.percentile(r, 90) * 0.05)] = np.nan
        out['h' + side] = fq.astype(np.float32)
    np.savez(f + '.tmp.npz', **out); os.replace(f + '.tmp.npz', f)


def load(cid):
    z = dict(np.load(f'{MC}/{cid}.npz'))
    if os.path.exists(f'{MC}/{cid}_f0hi.npz'): z.update(dict(np.load(f'{MC}/{cid}_f0hi.npz')))
    t = json.load(open(f'{CASES}/{cid}/truth.json'))
    if t['kind'] == 'events':
        p = np.asarray(t['pairs']); z['tg'], z['td'] = p[:, 0], p[:, 1]
    else:
        tw = np.asarray(t['warp'])
        g = np.concatenate([np.arange(a, b, 0.01) for a, b in t['voiced']]) if t['voiced'] else np.array([])
        z['tg'], z['td'] = g, np.interp(g, tw[:, 0], tw[:, 1])
    z['cid'] = cid
    return z


# ------------------------------------------------------------------ DTW core
@njit(cache=True)
def band_cost(A, B, band):
    """C[i, k] = 1 - A[i].B[j], j = i + k - band (inf outside B)."""
    n, m = A.shape[0], B.shape[0]
    C = np.full((n, 2 * band + 1), np.inf, np.float32)
    for i in range(n):
        for k in range(2 * band + 1):
            j = i + k - band
            if 0 <= j < m:
                s = 0.0
                for q in range(A.shape[1]): s += A[i, q] * B[j, q]
                C[i, k] = 1.0 - s
    return C


@njit(cache=True)
def dtw_band_py(C, m, wd):
    """fdtw.band_dtw's exact recursion (predecessor chosen on D alone, then + w*c)."""
    n, W = C.shape; band = (W - 1) // 2
    INF = 1e18
    D = np.full((n + 1, W + 2), INF)
    P = np.zeros((n + 1, W + 2), np.int8)
    D[0, band + 1] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band); hi = min(m, i + band)
        for j in range(lo, hi + 1):
            k = j - i + band + 1
            c = C[i - 1, k - 1]
            best = D[i - 1, k]; arg = 0
            if D[i, k - 1] < best: best = D[i, k - 1]; arg = 1
            if k + 1 < W + 2 and D[i - 1, k + 1] < best: best = D[i - 1, k + 1]; arg = 2
            D[i, k] = best + c * (wd if arg == 0 else 1.0); P[i, k] = arg
    i, j = n, m
    pi = np.empty(n + m, np.int64); pj = np.empty(n + m, np.int64); L = 0
    while i > 0 and j > 0:
        pi[L] = i - 1; pj[L] = j - 1; L += 1
        k = j - i + band + 1
        a = P[i, k]
        if a == 0: i -= 1; j -= 1
        elif a == 1: j -= 1
        else: i -= 1
    return pi[:L][::-1].copy(), pj[:L][::-1].copy()


@njit(cache=True)
def dtw_band(C, m, wd, wh, wv):
    """Same recursion as fdtw.band_dtw on a precomputed banded cost (wd=2,wh=wv=1 reproduces it)."""
    n, W = C.shape; band = (W - 1) // 2
    INF = 1e18
    D = np.full((n + 1, W + 2), INF)
    P = np.zeros((n + 1, W + 2), np.int8)
    D[0, band + 1] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band); hi = min(m, i + band)
        for j in range(lo, hi + 1):
            k = j - i + band + 1
            c = C[i - 1, k - 1]
            best = D[i - 1, k] + wd * c; arg = 0
            v = D[i, k - 1] + wh * c
            if v < best: best = v; arg = 1
            if k + 1 < W + 2:
                v = D[i - 1, k + 1] + wv * c
                if v < best: best = v; arg = 2
            D[i, k] = best; P[i, k] = arg
    i, j = n, m
    pi = np.empty(n + m, np.int64); pj = np.empty(n + m, np.int64); L = 0
    while i > 0 and j > 0:
        pi[L] = i - 1; pj[L] = j - 1; L += 1
        k = j - i + band + 1
        a = P[i, k]
        if a == 0: i -= 1; j -= 1
        elif a == 1: j -= 1
        else: i -= 1
    return pi[:L][::-1].copy(), pj[:L][::-1].copy()


def readout(pi, pj):
    cnt = np.bincount(pi).astype(float)
    dm = np.bincount(pi, pj.astype(float)) / np.maximum(cnt, 1)
    return np.arange(len(dm)), dm


def unit(F): return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def combo(z, melw, side):
    L, Mm = z[side + 'l'], z[side + 'm']
    return np.ascontiguousarray(np.concatenate([L, melw * Mm], 1) / np.sqrt(1 + melw * melw), np.float32)


def interval(z, dm):
    fg, fd = (z['hg'], z['hd']) if 'hg' in z else (z['fg'], z['fd'])
    di = np.clip(np.round(dm).astype(int), 0, len(fd) - 1); gg = np.clip(np.arange(len(dm)), 0, len(fg) - 1)
    ok = np.isfinite(fg[gg]) & np.isfinite(fd[di])
    return float(np.median(12 * np.log2(fd[di][ok] / fg[gg][ok]))) if ok.sum() > 50 else 0.0


def gate(z, band=100):
    """learned-only pass -> median f0 interval (proto_v5 / learned_dtw GATE=1 rule)."""
    if 'unison' in z: return z['unison']
    Cl = band_cost(z['Al'], z['Bl'], band); z['_Cl'] = Cl
    pi, pj = dtw_band(Cl, z['Bl'].shape[0], 2.0, 1.0, 1.0)
    _, dm = readout(pi, pj)
    z['interval'] = interval(z, dm); z['unison'] = abs(z['interval']) <= 0.5
    return z['unison']


def baseline_cost(z, band=100, melw=2.0):
    """Gated cost exactly as learned_dtw GATE=1 MELW=2."""
    key = f'_C{band}_{melw}'
    if key in z: return z[key]
    if gate(z) or melw == 0:
        C = band_cost(combo(z, melw, 'A'), combo(z, melw, 'B'), band) if melw else band_cost(z['Al'], z['Bl'], band)
    else:
        C = z['_Cl'] if band == 100 else band_cost(z['Al'], z['Bl'], band)
    z[key] = C
    return C


def errors(z, gtimes, dtimes):
    pred = np.interp(z['tg'], gtimes, dtimes)
    return np.abs(pred - z['td']) * 1000, np.abs(z['tg'] - z['td']) * 1000


def summ(errs):
    a = np.concatenate(errs) if len(errs) else np.zeros(1)
    return dict(mean=a.mean(), median=np.median(a), p90=np.percentile(a, 90), w20=100 * (a <= 20).mean(), n=len(a))


if __name__ == '__main__':
    import concurrent.futures as cf
    ids = manifest('R') + manifest('S')
    with cf.ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        list(ex.map(build, ids)); list(ex.map(build_f0hi, ids))
    print('built', len(ids))


# ------------------------------------------------------------------ general corridor DTW (2026-10-07 levers)
@njit(cache=True)
def corr_cost(A, B, cen, band):
    """C[i, k] = 1 - A[i].B[j] with j = cen[i] + k - band (inf outside B)."""
    n, m = A.shape[0], B.shape[0]
    C = np.full((n, 2 * band + 1), np.inf, np.float32)
    for i in range(n):
        for k in range(2 * band + 1):
            j = cen[i] + k - band
            if 0 <= j < m:
                s = 0.0
                for q in range(A.shape[1]): s += A[i, q] * B[j, q]
                C[i, k] = 1.0 - s
    return C


@njit(cache=True)
def corr_pitch(sg, sd, cen, band, iv, cap, vm):
    """Pitch cost on the same layout: both voiced -> min(|sg - (sd - iv)| mod-octave, cap) / cap,
    one voiced -> vm, none -> 0. sg/sd semitones with nan = unvoiced."""
    n, m = sg.shape[0], sd.shape[0]
    P = np.zeros((n, 2 * band + 1), np.float32)
    for i in range(n):
        a = sg[i]; va = not np.isnan(a)
        for k in range(2 * band + 1):
            j = cen[i] + k - band
            if j < 0 or j >= m: continue
            b = sd[j]; vb = not np.isnan(b)
            if va and vb:
                d = abs(a - (b - iv)); d = d % 12.0; d = min(d, 12.0 - d)
                P[i, k] = min(d, cap) / cap
            elif va != vb:
                P[i, k] = vm
    return P


@njit(cache=True)
def corr_absdiff(x, y, cen, band, cap):
    """|x_i - y_j| / cap clipped to 1 (nan -> 0) on the corridor layout."""
    n, m = x.shape[0], y.shape[0]
    P = np.zeros((n, 2 * band + 1), np.float32)
    for i in range(n):
        for k in range(2 * band + 1):
            j = cen[i] + k - band
            if j < 0 or j >= m: continue
            d = abs(x[i] - y[j])
            if not np.isnan(d): P[i, k] = min(d, cap) / cap
    return P


@njit(cache=True)
def dtw_corr(C, cen, m, wd, wh, wv, py):
    """DTW on a corridor cost (row i covers j = cen[i]-band .. cen[i]+band), path (0,0)->(n-1,m-1).
    py=1: fdtw's recursion (predecessor by D alone, then + w*c); py=0: textbook (min of D + w*c)."""
    n, W = C.shape; band = (W - 1) // 2
    INF = 1e18
    D = np.full((n, W), INF)
    P = np.zeros((n, W), np.int8)
    for i in range(n):
        for k in range(W):
            j = cen[i] + k - band
            if j < 0 or j >= m: continue
            c = C[i, k]
            if not np.isfinite(c): continue
            if i == 0 and j == 0:
                D[i, k] = c * wd; P[i, k] = 3; continue
            dd = INF; dh = INF; dv = INF
            if i > 0 and j > 0:
                kk = j - 1 - cen[i - 1] + band
                if 0 <= kk < W: dd = D[i - 1, kk]
            if k > 0: dh = D[i, k - 1]                         # (i, j-1)
            if i > 0:
                kk = j - cen[i - 1] + band
                if 0 <= kk < W: dv = D[i - 1, kk]              # (i-1, j)
            if py == 1:
                best = dd; arg = 0
                if dh < best: best = dh; arg = 1
                if dv < best: best = dv; arg = 2
                if best >= INF: continue
                D[i, k] = best + c * (wd if arg == 0 else (wh if arg == 1 else wv)); P[i, k] = arg
            else:
                best = dd + wd * c; arg = 0
                v = dh + wh * c
                if v < best: best = v; arg = 1
                v = dv + wv * c
                if v < best: best = v; arg = 2
                if best >= INF: continue
                D[i, k] = best; P[i, k] = arg
    i = n - 1; j = m - 1
    pi = np.empty(n + m, np.int64); pj = np.empty(n + m, np.int64); L = 0
    k = j - cen[i] + band
    if k < 0 or k >= W or D[i, k] >= INF:      # corner unreachable: end at the best cell of the last row
        best = INF; bk = 0
        for kk in range(W):
            if D[i, kk] < best: best = D[i, kk]; bk = kk
        j = cen[i] + bk - band
    while i >= 0 and j >= 0:
        pi[L] = i; pj[L] = j; L += 1
        k = j - cen[i] + band
        a = P[i, k]
        if a == 3: break
        if a == 0: i -= 1; j -= 1
        elif a == 1: j -= 1
        else: i -= 1
    return pi[:L][::-1].copy(), pj[:L][::-1].copy()
