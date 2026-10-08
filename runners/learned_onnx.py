#!/usr/bin/env python3
"""RAB runner: learned v6 embeddings (released ONNX) + log-mel x2 with the interval gate, dense band DTW.

This is the "learned v6 + mel" row of the README. It is the PyTorch runner learned/learned_dtw.py with
CKPT=align_v6_final, FEAT=learned+mel, MELW=2, GATE=1, re-expressed on the released ONNX graph.

usage (as a rab_score RUNNER): learned_onnx.py MODEL guide.wav double.wav
  MODEL: path to rysupalign_learned_v6[.int8].onnx; '-' = env RYSUPALIGN_ONNX / <repo>/models/...
env MELW (2.0) weight of the log-mel block, INT_ST (0.5) unison threshold in semitones,
    GATE (1) 0 = always use learned+mel, BAND_S (1.0) DTW band.

Interval gate: align on the learned block alone, read yin f0 of guide and double at corresponding
frames; if the median interval is more than INT_ST from unison (harmony / octave double), the log-mel
block (which compares harmonic patterns) is left out and the learned path is used as is.
"""
import json, os, sys
import numpy as np, librosa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fdtw, embed_onnx

SR, HOP = fdtw.SR, 441


def f0(x, fmax=800):
    y = librosa.resample(x, orig_sr=SR, target_sr=8000)
    f = librosa.yin(y, fmin=70, fmax=fmax, sr=8000, frame_length=512, hop_length=80)   # 10 ms
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=80)[0][:len(f)]
    f[rms < max(1e-4, np.percentile(rms, 90) * 0.05)] = np.nan
    return f


def feats(path, x, melw):
    L = embed_onnx.learned_block(path, x, SR, HOP)
    if melw == 0: return np.ascontiguousarray(L.T, np.float32)
    Mb = fdtw.block('mel', path, x, HOP)
    n = min(L.shape[1], Mb.shape[1])
    F = np.concatenate([L[:, :n], melw * Mb[:, :n]]) / np.sqrt(1 + melw * melw)
    return np.ascontiguousarray(F.T, np.float32)


def readout(path):
    p = np.asarray(path); cnt = np.bincount(p[:, 0]).astype(float)
    return np.bincount(p[:, 0], p[:, 1].astype(float)) / np.maximum(cnt, 1)


def main():
    model, guide, dbl = sys.argv[1:4]
    if model not in ('-', '') and model.endswith('.onnx'): embed_onnx.MODEL = model
    band = int(float(os.environ.get('BAND_S', '1.0')) * SR / HOP)
    melw = float(os.environ.get('MELW', '2.0'))
    xg, xd = fdtw.load(guide), fdtw.load(dbl)
    interval, cand = None, 'learned+mel'
    if os.environ.get('GATE', '1') == '1' and melw > 0:
        A, B = feats(guide, xg, 0), feats(dbl, xd, 0)
        dm = readout(fdtw.band_dtw(A, B, band))
        fg, fd = f0(xg), f0(xd)
        di = np.clip(np.round(dm).astype(int), 0, len(fd) - 1); gg = np.clip(np.arange(len(dm)), 0, len(fg) - 1)
        ok = np.isfinite(fg[gg]) & np.isfinite(fd[di])
        interval = float(np.median(12 * np.log2(fd[di][ok] / fg[gg][ok]))) if ok.sum() > 50 else 0.0
        if abs(interval) > float(os.environ.get('INT_ST', '0.5')): cand = 'learned'
    if cand == 'learned' and interval is not None:
        path = fdtw.band_dtw(A, B, band)
    else:
        path = fdtw.band_dtw(feats(guide, xg, melw), feats(dbl, xd, melw), band)
    p = np.asarray(path); g = p[:, 0]; d = p[:, 1].astype(float)
    gu, inv = np.unique(g, return_inverse=True); dm = np.bincount(inv, d) / np.bincount(inv)
    fs = SR / HOP
    print(json.dumps({'ok': True, 'moved': True, 'cand': cand, 'interval': interval,
                      'warp': [[float(a / fs), float(b / fs)] for a, b in zip(gu, dm)]}))


if __name__ == '__main__':
    main()
