#!/usr/bin/env python3
"""RAB runner for method M1 (the "learned v6 + mel + M1 DTW" row of the README).

Features: learned v6 (released ONNX) + log-mel x2 with the interval gate (as learned_onnx.py), yin pitch.
Method M1 = textbook DTW recursion with diagonal weight 1.5, and in unison mode two pitch terms added to
the local cost: 0.25 * pitch distance (|semitone difference| octave-folded, capped at 2 st, /2; 0.5 for a
voicing mismatch) and 0.5 * pitch-slope distance (difference of d st/dt over +-3 frames, capped at
0.3 st/frame). See analysis/METHOD_RESULTS.md.

usage (as a rab_score RUNNER): m1_dtw.py MODEL guide.wav double.wav
  MODEL: path to the ONNX model or '-' (env RYSUPALIGN_ONNX / <repo>/models/...)
env METHOD: a method_variants name or 'k=v/k=v' config (default M1 below; 'base_py' = plain v6 + mel).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M1 = 'rec=std/wd=1.5/pw=0.25/pcap=2/puni=1/dw=0.5/dcap=0.3/duni=1'


def yin(x, fmax):
    import librosa
    y = librosa.resample(x, orig_sr=44100, target_sr=8000)
    f = librosa.yin(y, fmin=70, fmax=fmax, sr=8000, frame_length=512, hop_length=80)
    r = librosa.feature.rms(y=y, frame_length=512, hop_length=80)[0][:len(f)]
    f[r < max(1e-4, np.percentile(r, 90) * 0.05)] = np.nan
    return f.astype(np.float32), r.astype(np.float32)


def main():
    model, guide, dbl = sys.argv[1:4]
    import fdtw, embed_onnx, method_lib as ML, method_variants as MV
    if model not in ('-', '') and model.endswith('.onnx'): embed_onnx.MODEL = model
    z = {}
    for S, s, p in (('A', 'g', guide), ('B', 'd', dbl)):
        x = fdtw.load(p)
        Lb = embed_onnx.learned_block(p, x, fdtw.SR, ML.HOP); Mb = fdtw.block('mel', p, x, ML.HOP)
        n = min(Lb.shape[1], Mb.shape[1])
        z[S + 'l'] = np.ascontiguousarray(Lb[:, :n].T, np.float32); z[S + 'm'] = np.ascontiguousarray(Mb[:, :n].T, np.float32)
        z['f' + s], z['r' + s] = yin(x, 800)
        z['h' + s], _ = yin(x, 1600)
    name = os.environ.get('METHOD', M1)
    g, d = MV.VARIANTS[name](z)
    print(json.dumps({'ok': True, 'moved': True, 'cand': 'unison' if z.get('unison') else 'learned',
                      'interval': z.get('interval'), 'warp': [[float(a), float(b)] for a, b in zip(g, d)]}))


if __name__ == '__main__':
    main()
