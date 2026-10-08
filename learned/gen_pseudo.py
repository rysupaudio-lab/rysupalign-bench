#!/opt/homebrew/bin/python3.13
"""PSEUDO-labelled real-double pairs (mine_doubles.py): two of the owner's own takes of the same line.
Truth = a DTW path (hubert@6+mel by default, or LABEL=<ckpt> to relabel with a learned model),
so it is approximate; train.py uses a soft Gaussian target (meta 'sigma') for these.
Both sides may get an extra unison revoice (pitch shift shared, formant shift independent) and colour.

usage: gen_pseudo.py OUTNAME NPAIRS [--split train|val] [--seed S] [--jobs 10] [--set logic|vs|gang]
(v7) --set gang: real lead/gang pairs (pseudo_gang, mine_gang.py); meta pitch = [pa, pa + interval, fa, fb]
so octave/harmony gang pairs are not counted as unison by train.py.
Validation split = pairs whose first file is in gen_pairs' held-out file set.
"""
import argparse, json, os, sys, tempfile, multiprocessing as mp
import numpy as np, soundfile as sf, librosa
from scipy.ndimage import median_filter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import gen_pairs as G

LOGIC = os.environ.get('OWN_VOCALS_DIR', 'own_vocals')
PS = f'{G.TR}/pseudo'


def load(f):
    x, sr = sf.read(f if f.startswith('/') else f'{LOGIC}/{f}', dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    return librosa.resample(x, orig_sr=sr, target_sr=G.SR) if sr != G.SR else x


def one(args):
    seed, pair = args
    rng = np.random.default_rng(seed)
    xa, xb = load(pair['a']), load(pair['b'])
    iv = float(pair.get('interval', 0.0))                    # gang pairs: double's interval to the guide (st)
    if 'npz' in pair:                                       # VocalSet cross-singer / gang: agreed labels + mask
        z = np.load(f"{G.TR}/{pair.get('dir', 'pseudo_vs')}/{pair['npz']}"); dm = np.maximum.accumulate(z['dm'].astype(float)); okm = z[pair.get('okkey', 'ok')]
    else:
        p = np.load(f"{PS}/{pair['path']}")                # 10 ms DTW path (a frame, b frame)
        na = p[:, 0].max() + 1
        cnt = np.bincount(p[:, 0], minlength=na); dm = np.bincount(p[:, 0], p[:, 1].astype(float), minlength=na) / np.maximum(cnt, 1)
        dm = median_filter(dm, 5, mode='nearest')            # a time (10 ms) -> b time (10 ms)
        okm = np.ones(len(dm), bool)
    if rng.random() < 0.5:                                  # swap roles
        xa, xb = xb, xa
        inv = np.interp(np.arange(len(xa) // 441 + 1), np.maximum.accumulate(dm), np.arange(len(dm)))
        okm = okm[np.clip(np.round(inv).astype(int), 0, len(okm) - 1)]; dm = inv; iv = -iv
    L = int(G.CROP * G.SR)
    s = rng.integers(0, max(1, len(xa) - L + 1)); t0 = s / G.SR
    tb0 = np.interp(t0 * 100, np.arange(len(dm)), dm) / 100 - rng.uniform(-0.1, 0.1)
    sb = int(max(0, tb0 * G.SR))
    a = np.pad(xa[s:s + L], (0, max(0, L - len(xa[s:s + L])))); b = np.pad(xb[sb:sb + L], (0, max(0, L - len(xb[sb:sb + L]))))
    a = a + rng.standard_normal(L).astype(np.float32) * 1e-5; b = b + rng.standard_normal(L).astype(np.float32) * 1e-5
    pa = 0.0; fa = fb = 0.0
    if rng.random() < (0.6 if 'path' in pair else 0.3 if 'interval' not in pair else 0.25):
        pa = rng.uniform(-6, 9); fa = np.clip(0.3 * pa + rng.normal(0, 1.5), -4, 4); fb = np.clip(0.3 * pa + rng.normal(0, 1.5), -4, 4)
    with tempfile.TemporaryDirectory() as tmp:
        A = G.colour(G.revoice(a, pa, fa, tmp) if pa else a / (np.abs(a).max() + 1e-9) * 0.5, rng)
        B = G.colour(G.revoice(b, pa + rng.normal(0, 0.15), fb, tmp) if pa else b / (np.abs(b).max() + 1e-9) * 0.5, rng)
    A16 = librosa.resample(A, orig_sr=G.SR, target_sr=16000)[:int(G.CROP * 16000)]
    B16 = librosa.resample(B, orig_sr=G.SR, target_sr=16000)[:int(G.CROP * 16000)]
    tA = 0.02 * np.arange(G.NF) + 0.0125
    tB = np.interp((t0 + tA) * 100, np.arange(len(dm)), dm) / 100 - sb / G.SR
    jf = (tB - 0.0125) / 0.02
    hop = int(0.02 * G.SR); ex = G.energy_db(np.pad(a, (int(0.0025 * G.SR), hop)), hop)[:G.NF]
    ka = np.clip(np.round((t0 + tA) * 100).astype(int), 0, len(okm) - 1)
    valid = (ex > np.percentile(ex, 98) - 35) & (jf >= 0) & (jf <= G.NF - 1) & ((t0 + tA) * 100 < len(dm) - 1) & okm[ka]
    return A16.astype(np.float16), B16.astype(np.float16), jf.astype(np.float32), valid, np.array([pa, pa + iv, fa, fb], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('n', type=int)
    ap.add_argument('--split', default='train'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=10); ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--set', default='logic')       # logic | vs (VocalSet cross-singer) | gang (mine_gang.py)
    ap.add_argument('--minagree', type=float, default=0.3); ap.add_argument('--strict', type=int, default=0)   # gang: ok3 mask
    a = ap.parse_args()
    if a.set == 'gang':                             # real lead/gang pairs, weighted by labelled seconds
        J = json.load(open(f'{G.TR}/pseudo_gang/pairs.json'))
        ck = 'cover3_s' if a.strict else 'cover_s'
        J = [dict(p, dir='pseudo_gang', okkey='ok3' if a.strict else 'ok') for p in J
             if p['split'] == a.split and p['agree'] >= a.minagree and p[ck] >= 1.5]
        pairs = sum(([p] * max(1, int(round(p[ck] / 2))) for p in J), [])
        print('gang pairs used', len(J), {k: sum(p['kind'] == k for p in J) for k in ('unison', 'octave', 'harmony')})
    elif a.set == 'vs':
        J = json.load(open(f'{G.TR}/pseudo_vs/pairs.json'))
        pairs = [p for p in J if p['split'] == a.split]
        pairs = sum(([p] * (3 if p['kind'] == 'excerpt' else 1) for p in pairs), [])
    else:
        J = json.load(open(f'{PS}/pairs.json'))['pairs']
        idx = json.load(open(f'{G.TR}/src/index.json'))
        files = sorted({c['file'] for c in idx}); val_files = set(files[::10])
        pairs = [p for p in J if ((p['a'] in val_files) or (p['b'] in val_files)) == (a.split == 'val')]
    print(len(pairs), 'pseudo pairs in split', a.split, flush=True)
    rng = np.random.default_rng(a.seed)
    S = int(G.CROP * 16000)
    XA = np.lib.format.open_memmap(f'{G.TR}/pairs/{a.name}_A.npy', 'w+', np.float16, (a.n, S))
    XB = np.lib.format.open_memmap(f'{G.TR}/pairs/{a.name}_B.npy', 'w+', np.float16, (a.n, S))
    JF = np.zeros((a.n, G.NF), np.float32); V = np.zeros((a.n, G.NF), bool); P = np.zeros((a.n, 4), np.float32)
    jobs = [(a.seed * 7919 + i, pairs[rng.integers(len(pairs))]) for i in range(a.n)]
    with mp.Pool(a.jobs) as pool:
        for i, (A, B, jf, v, p) in enumerate(pool.imap(one, jobs, chunksize=4)):
            XA[i] = A; XB[i] = B; JF[i] = jf; V[i] = v; P[i] = p
    XA.flush(); XB.flush()
    np.savez(f'{G.TR}/pairs/{a.name}_meta.npz', jf=JF, valid=V, pitch=P, sigma=np.float32(a.sigma))
    print('done', a.n, 'valid frac', V.mean())


if __name__ == '__main__':
    main()
