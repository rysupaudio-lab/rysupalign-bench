#!/opt/homebrew/bin/python3.13
"""Collect owned training vocals -> mono float32 44.1 kHz phrase chunks in $RAB_WORK/training/src.

Sources: your own vocal recordings in $OWN_VOCALS_DIR ('Recording Track #*.wav', as exported by a DAW).
The published model used ~28 min of Rys Up Audio's own unreleased studio vocal recordings, which are not
distributed. The fury takes (a suite-S source) and all RAB / Choral Singing Dataset audio were excluded.
Output: training/src/<n>.npy chunks (3-12 s, mostly voiced) + training/src/index.json
"""
import json, os, re
import numpy as np, soundfile as sf, librosa

LOGIC = os.environ.get('OWN_VOCALS_DIR', 'own_vocals')
OUT = os.path.join(os.environ.get('RAB_WORK', 'work'), 'training', 'src')
SR = 44100


def energy_db(x, hop):
    n = len(x) // hop
    return 10 * np.log10(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12)


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = []; tot = 0.0; k = 0
    files = sorted(f for f in os.listdir(LOGIC) if re.match(r'^Recording Track #\d+\.wav$', f))
    for f in files:
        x, sr = sf.read(f'{LOGIC}/{f}', dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
        if sr != SR: x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        if np.abs(x).max() < 1e-3: continue
        hop = int(0.01 * SR); e = energy_db(x, hop)
        thr = max(-55.0, np.percentile(e, 98) - 35.0); act = e >= thr
        if act.sum() < 100 or np.percentile(e, 98) < -50: continue
        # chunk: active regions with <=0.6 s gaps merged, then split to <=12 s with 0.25 s pad
        segs = []; i = 0; n = len(act)
        while i < n:
            if not act[i]: i += 1; continue
            s = i; last = i
            while i < n and (act[i] or i - last < 60):
                if act[i]: last = i
                i += 1
            segs.append((max(0, s - 25), min(n, last + 26)))
        for a, b in segs:
            while b - a >= 300:
                e_ = min(b, a + 1200)
                if b - e_ < 300: e_ = b if b - a <= 1500 else a + 900
                seg = x[a * hop:e_ * hop]
                if act[a:e_].mean() > 0.35:
                    np.save(f'{OUT}/{k:05d}.npy', seg); idx.append({'id': k, 'file': f, 't0': a * 0.01, 'dur': len(seg) / SR})
                    tot += len(seg) / SR; k += 1
                a = e_
    json.dump(idx, open(f'{OUT}/index.json', 'w'), indent=0)
    print(len(idx), 'chunks', tot / 60, 'min', len({c['file'] for c in idx}), 'files')


if __name__ == '__main__':
    main()
