#!/usr/bin/env python3
"""Reference feature-DTW aligner for RAB (research code, not the plug-in).

usage (as a rab_score RUNNER): fdtw.py MODEL guide.wav double.wav   (MODEL is ignored)
env FEAT = mel | mfcc | chroma | flux | <ssl>@<layer> | combinations such as hubert@6+mel
           ('+' concatenates unit-normalised blocks; <ssl> names are listed in ssl_feats.py)
    HOP_MS (default 10), BAND_S (default 1.0)
    GATE=1: interval gate for '...+mel' features: align on the features without mel first, read the
           median yin f0 interval; mel is added only for unison doubles (|interval| <= 0.5 st), as in
           learned_onnx.py (mel compares harmonic patterns and breaks harmony / octave doubles)
Prints {"warp": [[g, d], ...]} following the full-resolution DTW path (no anchor filtering),
so it measures what a feature can do, not what our engine's safety gates allow.
"""
import json, os, sys
import numpy as np, soundfile as sf, librosa
from numba import njit

SR = 44100


def load(p):
    x, sr = sf.read(p, dtype='float32')
    x = x.mean(1) if x.ndim > 1 else x
    return librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x


def block(name, p, x, hop):
    if name == 'mel':
        F = np.log(librosa.feature.melspectrogram(y=x, sr=SR, n_fft=2048, hop_length=hop, n_mels=80, fmax=12000) + 1e-6)
    elif name == 'mfcc':
        F = librosa.feature.mfcc(y=x, sr=SR, n_fft=2048, hop_length=hop, n_mfcc=20)[1:]
    elif name == 'chroma':
        F = librosa.feature.chroma_cqt(y=x, sr=SR, hop_length=hop)
    elif name == 'flux':   # onset-type feature: positive log-mel differences (pitch-robust)
        M = np.log(librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=hop, n_mels=40) + 1e-6)
        F = np.maximum(0, np.diff(M, axis=1, prepend=M[:, :1]))
    elif '@' in name:                                   # frontier model embeddings (ssl_feats.py)
        import ssl_feats
        mdl, layer = name.split('@')
        E, fps = ssl_feats.embed(p, x, SR, mdl, int(layer))
        E = E - E.mean(0, keepdims=True)                   # remove the take's constant offset
        n = 1 + len(x) // hop
        src = np.linspace(0, len(E) - 1, n)
        F = np.stack([np.interp(src, np.arange(len(E)), c) for c in E.T])
    else:
        raise SystemExit('unknown feature ' + name)
    F = F - F.mean(0, keepdims=True) if name in ('mel', 'mfcc') else F
    return F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)


def features(p, x, feat, hop):
    blocks = [block(b, p, x, hop) for b in feat.split('+')]
    n = min(b.shape[1] for b in blocks)
    F = np.concatenate([b[:, :n] for b in blocks]) / np.sqrt(len(blocks))
    return np.ascontiguousarray(F.T.astype(np.float32))


@njit(cache=True)
def band_dtw(A, B, band):
    n, m = A.shape[0], B.shape[0]
    INF = 1e18
    D = np.full((n + 1, 2 * band + 3), INF)        # D[i, j - i + band + 1]
    P = np.zeros((n + 1, 2 * band + 3), np.int8)
    D[0, band + 1] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band); hi = min(m, i + band)
        for j in range(lo, hi + 1):
            c = 1.0 - np.dot(A[i - 1], B[j - 1])
            k = j - i + band + 1
            best = D[i - 1, k]; arg = 0                 # (i-1, j-1)
            if k - 1 >= 0 and D[i, k - 1] + c * 0.0 < INF and D[i, k - 1] < best: best = D[i, k - 1]; arg = 1   # (i, j-1)
            if k + 1 < D.shape[1] and D[i - 1, k + 1] < best: best = D[i - 1, k + 1]; arg = 2                     # (i-1, j)
            D[i, k] = best + c * (2.0 if arg == 0 else 1.0)
            P[i, k] = arg
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = j - i + band + 1
        a = P[i, k]
        if a == 0: i -= 1; j -= 1
        elif a == 1: j -= 1
        else: i -= 1
    return path[::-1]


def main():
    _, guide, dbl = sys.argv[1:4]
    feat = os.environ.get('FEAT', 'mel'); hop = int(round(float(os.environ.get('HOP_MS', '10')) / 1000 * SR))
    band = int(float(os.environ.get('BAND_S', '1.0')) * SR / hop)
    xg, xd = load(guide), load(dbl)
    if os.environ.get('GATE', '0') == '1' and feat.endswith('+mel') and hop == 441:
        import learned_onnx as LO
        base = feat[:-len('+mel')]
        A, B = features(guide, xg, base, hop), features(dbl, xd, base, hop)
        p = np.asarray(band_dtw(A, B, band)); cnt = np.bincount(p[:, 0]).astype(float)
        dm = np.bincount(p[:, 0], p[:, 1].astype(float)) / np.maximum(cnt, 1)
        fg, fd = LO.f0(xg), LO.f0(xd)
        di = np.clip(np.round(dm).astype(int), 0, len(fd) - 1); gg = np.clip(np.arange(len(dm)), 0, len(fg) - 1)
        ok = np.isfinite(fg[gg]) & np.isfinite(fd[di])
        iv = float(np.median(12 * np.log2(fd[di][ok] / fg[gg][ok]))) if ok.sum() > 50 else 0.0
        if abs(iv) > 0.5: feat = base
    A, B = features(guide, xg, feat, hop), features(dbl, xd, feat, hop)
    path = band_dtw(A, B, band)
    fs = SR / hop
    # one knot per guide frame: mean double frame of the path at that guide frame
    g = np.array([p[0] for p in path]); d = np.array([p[1] for p in path], float)
    gu, inv = np.unique(g, return_inverse=True)
    dm = np.bincount(inv, d) / np.bincount(inv)
    warp = [[float(a / fs), float(b / fs)] for a, b in zip(gu, dm)]
    print(json.dumps({'ok': True, 'moved': True, 'warp': warp}))


if __name__ == '__main__':
    main()
