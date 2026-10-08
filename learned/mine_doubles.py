#!/opt/homebrew/bin/python3.13
"""Find real double takes (same line sung again) among the owned Logic recordings and build
PSEUDO-labelled pairs (marked separately: truth comes from DTW on hubert@6+mel, not exact).

Candidates: recordings within 8 file numbers of each other, duration ratio 0.8-1.25, >= 2.5 s.
Score: mean DTW path cost vs the cost distribution of random unrelated pairs; keep pairs whose
cost z-score is very low AND whose path is smooth (few long horizontal/vertical runs).
Output training/pseudo/pairs.json with [file_a, file_b, cost, z] and per-pair warps (npz).
"""
import json, os, re, sys
import numpy as np, soundfile as sf, librosa
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import fdtw

LOGIC = os.environ.get('OWN_VOCALS_DIR', 'own_vocals')
OUT = os.path.join(os.environ.get('RAB_WORK', 'work'), 'training', 'pseudo')


def path_stats(A, B, band):
    p = np.array(fdtw.band_dtw(A, B, band))
    c = 1 - np.einsum('ij,ij->i', A[p[:, 0]], B[p[:, 1]])
    st = np.diff(p, axis=0); flat = (st[:, 0] == 0) | (st[:, 1] == 0)
    run = 0; mx = 0
    for f in flat:
        run = run + 1 if f else 0; mx = max(mx, run)
    return p, float(c.mean()), mx


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted((int(re.findall(r'\d+', f)[0]), f) for f in os.listdir(LOGIC) if re.match(r'^Recording Track #\d+\.wav$', f))
    info = {}
    for n, f in files:
        d = sf.info(f'{LOGIC}/{f}').duration
        if d >= 2.5: info[n] = (f, d)
    hop = int(0.01 * fdtw.SR); feats = {}

    def feat(n):
        if n not in feats:
            p = f'{LOGIC}/{info[n][0]}'; x = fdtw.load(p)
            feats[n] = fdtw.features(p, x, 'hubert@6+mel', hop)
        return feats[n]
    keys = sorted(info)
    cand = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:] if b - a <= 8 and 0.8 <= info[a][1] / info[b][1] <= 1.25]
    rng = np.random.default_rng(0)
    rand = []
    while len(rand) < 150:
        a, b = rng.choice(keys, 2, replace=False)
        if abs(a - b) > 40 and 0.8 <= info[a][1] / info[b][1] <= 1.25: rand.append((a, b))
    res = {}
    for a, b in cand + rand:
        A, B = feat(a), feat(b)
        band = int(max(100, abs(len(A) - len(B)) + 50))
        p, c, mx = path_stats(A, B, band)
        res[(a, b)] = (c, mx, p)
    rc = np.array([res[k][0] for k in rand]); mu, sd = rc.mean(), rc.std()
    print(f'random pairs cost {mu:.3f} +- {sd:.3f}; candidates {len(cand)}')
    keep = []
    for a, b in cand:
        c, mx, p = res[(a, b)]; z = (c - mu) / sd
        if z < -3.0 and mx < 40:
            keep.append(dict(a=info[a][0], b=info[b][0], cost=c, z=z))
            np.save(f'{OUT}/{a}_{b}.npy', p.astype(np.int32))
            keep[-1]['path'] = f'{a}_{b}.npy'
    print(len(keep), 'pseudo doubles kept')
    for k in sorted(keep, key=lambda k: k['z'])[:40]: print(f"{k['a']:28s} {k['b']:28s} cost {k['cost']:.3f} z {k['z']:.1f}")
    json.dump(dict(random_mu=mu, random_sd=sd, pairs=keep), open(f'{OUT}/pairs.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
