#!/opt/homebrew/bin/python3.13
"""VocalSet 1.2 (Wilkins et al. 2018, CC BY 4.0: 20 singers, techniques/vowels/excerpts) ->
training chunks in training/src_vs/ (same format as prep_sources.py: mono float32 44.1 kHz,
3-12 s mostly-voiced chunks + index.json with singer/context/technique).

Held-out split for validation: singers female9, male11 (never used for training).
"""
import json, os
import numpy as np, soundfile as sf, librosa
import prep_sources as P

VS = os.environ.get('VOCALSET_DIR', 'datasets/VocalSet')
OUT = os.path.join(os.environ.get('RAB_WORK', 'work'), 'training', 'src_vs')
VAL_SINGERS = {'female9', 'male11'}


def wavs():
    for root, dirs, files in os.walk(VS, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__MACOSX')) and not os.path.islink(os.path.join(root, d))]
        for f in files:
            if f.endswith('.wav') and not f.startswith('._'): yield os.path.join(root, f)


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = []; k = 0; tot = 0.0; SR = P.SR; hop = int(0.01 * SR)
    for p in sorted(wavs()):
        rel = os.path.relpath(p, VS); parts = rel.split(os.sep)
        singer = next((s for s in parts if s.rstrip('0123456789') in ('female', 'male') and s[-1].isdigit()), None)
        if singer is None: continue
        x, sr = sf.read(p, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
        if sr != SR: x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        if np.abs(x).max() < 1e-3: continue
        e = P.energy_db(x, hop); thr = max(-55.0, np.percentile(e, 98) - 35.0); act = e >= thr
        if act.sum() < 100: continue
        a = int(np.argmax(act)); b = len(act) - int(np.argmax(act[::-1]))
        a = max(0, a - 25); b = min(len(act), b + 25)
        while b - a >= 300:
            e_ = min(b, a + 1200)
            if b - e_ < 300: e_ = b if b - a <= 1500 else a + 900
            if act[a:e_].mean() > 0.35:
                np.save(f'{OUT}/{k:05d}.npy', x[a * hop:e_ * hop])
                idx.append({'id': k, 'file': rel, 'singer': singer, 't0': a * 0.01, 'dur': (e_ - a) * 0.01,
                            'val': singer in VAL_SINGERS}); tot += (e_ - a) * 0.01; k += 1
            a = e_
    json.dump(idx, open(f'{OUT}/index.json', 'w'), indent=0)
    print(len(idx), 'chunks', tot / 60, 'min', len({c['singer'] for c in idx}), 'singers')


if __name__ == '__main__':
    main()
