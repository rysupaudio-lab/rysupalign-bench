#!/opt/homebrew/bin/python3.13
"""Different-singer PSEUDO pairs from VocalSet: two singers performing the same excerpt / exercise
(same file name after the singer id, e.g. f1_caro_straight vs m3_caro_straight).

Confident-DTW labelling: two independent aligners must agree. Labels = mean of
  (1) the learned model (CKPT, default v1ms4k) + fdtw log-mel (MELW 1.5)  and
  (2) hubert@6 + log-mel (fdtw),
and a guide frame is valid only where the two paths agree within AGREE_MS (20 ms) and the
guide is audible. Pairs with < 50 % agreeing audible frames are dropped.
Held-out singers female9 / male11 -> split 'val'. EXCERPTS_ONLY=1 (default) mines only the
lyric excerpts (caro / dona / row); exercises are slow to mine and rarely pass the agreement test.

Output training/pseudo_vs/<k>.npz {dm: guide 10 ms frame -> double frame (float), ok: bool}
       + pairs.json [{a, b, npz, agree, kind, split}]
"""
import itertools, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault('CKPT', 'v1ms4k'); os.environ.setdefault('MELW', '1.5'); os.environ['FEAT'] = 'learned+mel'
import fdtw, learned_dtw as LD

VS = os.path.join(os.environ.get('VOCALSET_DIR', 'datasets/VocalSet'), 'data_by_singer')
OUT = os.path.join(os.environ.get('RAB_WORK', 'work'), 'training', 'pseudo_vs')
VAL = {'female9', 'male11'}
AGREE = 2.0          # frames (10 ms) -> 20 ms


def dense(A, B, band):
    p = np.array(fdtw.band_dtw(A, B, band))
    cnt = np.bincount(p[:, 0], minlength=len(A)).astype(float)
    return np.bincount(p[:, 0], p[:, 1].astype(float), minlength=len(A)) / np.maximum(cnt, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    groups = {}
    for singer in sorted(os.listdir(VS)):
        if not re.match(r'^(fe)?male\d+$', singer): continue
        for root, dirs, files in os.walk(f'{VS}/{singer}', followlinks=False):
            if '/long_tones' in root: continue
            for f in files:
                if not f.endswith('.wav') or f.startswith('._'): continue
                key = re.sub(r'^[fm]\d+_', '', f).strip()
                groups.setdefault(key, []).append((singer, os.path.join(root, f)))
    rng = np.random.default_rng(0)
    hop = int(0.01 * fdtw.SR); out = []; k = 0; feats = {}

    def F(p):
        if p not in feats:
            x = fdtw.load(p)
            feats[p] = (LD.features(p, x, hop), fdtw.features(p, x, 'hubert@6+mel', hop), x)
        return feats[p]
    for key, mem in sorted(groups.items()):
        if len(mem) < 2: continue
        exc = key.startswith(('row', 'caro', 'dona'))
        if not exc and os.environ.get('EXCERPTS_ONLY', '1') == '1': continue
        pairs = list(itertools.combinations(mem, 2))
        rng.shuffle(pairs); pairs = pairs[:(60 if exc else 6)]
        for (sa, pa), (sb, pb) in pairs:
            if 'spoken' in key: continue
            A1, A2, xa = F(pa); B1, B2, xb = F(pb)
            if not 0.6 <= len(A1) / len(B1) <= 1.67: continue
            band = int(abs(len(A1) - len(B1)) + 150)
            d1 = dense(A1, B1, band); d2 = dense(A2, B2, band)
            n = min(len(d1), len(d2), len(A1))
            xp = np.pad(xa, (0, max(0, n * hop - len(xa)))); xe = xp[:n * hop].reshape(n, hop); ed = 10 * np.log10((xe ** 2).mean(1) + 1e-12)
            aud = ed > np.percentile(ed, 98) - 35
            ok = aud & (np.abs(d1[:n] - d2[:n]) <= AGREE)
            agree = ok.sum() / max(1, aud.sum())
            if agree < 0.5: continue
            split = 'val' if (sa in VAL or sb in VAL) else 'train'
            np.savez(f'{OUT}/{k}.npz', dm=((d1[:n] + d2[:n]) / 2).astype(np.float32), ok=ok)
            out.append(dict(a=pa, b=pb, npz=f'{k}.npz', agree=float(agree), kind='excerpt' if exc else 'exercise', split=split))
            k += 1
        print(key, len(mem), 'kept so far', k, flush=True)
    json.dump(out, open(f'{OUT}/pairs.json', 'w'), indent=0)
    ag = np.array([o['agree'] for o in out])
    print('kept', len(out), 'excerpt', sum(o['kind'] == 'excerpt' for o in out), 'val', sum(o['split'] == 'val' for o in out),
          'agree median', np.median(ag))


if __name__ == '__main__':
    main()
