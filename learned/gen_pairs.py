#!/opt/homebrew/bin/python3.13
"""Self-supervised alignment pairs from owned vocals (training/src chunks, see prep_sources.py).

Each pair = (A, B) at 16 kHz, both CROP seconds long:
  A = aug_A(x)                        the 'guide' singer: pitch/formant shift + EQ/noise/reverb/drive
  B = aug_B(wsola(x, warp))           the 'double': same line with realistic per-phrase / per-syllable
                                      timing slop (exact, recorded WSOLA grain positions), its own
                                      pitch (unison+detune / harmony / octave) and formant shift.
Truth: for every A HuBERT frame i (50 fps) the fractional B frame j*(i) and a validity mask
(A frame audible, its source time reached by B). Correspondence error <= 0.1 ms (WSOLA grain
positions are recorded, not requested; Rubber Band R3 constant pitch shift is time-neutral, checked).

usage: gen_pairs.py OUTNAME NPAIRS [--split train|val] [--seed S] [--jobs 12] [--v 1|2]
  --v 3 (v7) = v2 with octave/harmony-heavy intervals (30 % unison, 40 % +-12 st, 30 % harmony).
  --v 2 adds per-singer pitch behaviour (vibrato, scoops, drift via time-neutral rubberband
  --pitchmap), a wider register (-8..+10 st) and register-following formant shifts.
writes training/pairs/OUTNAME_{A,B}.npy (float16 [N, CROP*16000]) + OUTNAME_meta.npz
"""
import argparse, json, os, subprocess, tempfile, multiprocessing as mp
import numpy as np, soundfile as sf, librosa
from numba import njit
from scipy.signal import lfilter, fftconvolve

TR = os.path.join(os.environ.get('RAB_WORK', 'work'), 'training')
SR = 44100
CROP = 6.4
NF = int((CROP * 16000 - 400) // 320) + 1          # HuBERT frames per crop (319)


def split_ids(split, src='src'):
    idx = json.load(open(f'{TR}/{src}/index.json'))
    if 'val' in idx[0]:                              # VocalSet: held-out singers; no long tones;
        out = []                                     # lyric excerpts x3 so exercises don't dominate
        for c in idx:
            if c['val'] != (split == 'val') or '/long_tones/' in c['file']: continue
            out += [(src, c['id'])] * (3 if '/excerpts/' in c['file'] else 1)
        return out
    files = sorted({c['file'] for c in idx})
    val_files = set(files[::10])                     # every 10th source file held out
    return [(src, c['id']) for c in idx if (c['file'] in val_files) == (split == 'val')]


# ------------------------------------------------------------------ timing warp
def energy_db(x, hop):
    n = len(x) // hop
    return 10 * np.log10(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12)


def onsets(e, thr):
    res, last = [], -99
    for k in range(4, len(e)):
        if e[k] >= thr and e[k] - e[k - 4:k].min() >= 6 and k - last >= 12:
            res.append(k); last = k
    return res


def make_warp(x, rng):
    """Displacement knots: input time t -> output time t + d(t). Per-phrase base offset,
    per-syllable jitter switching in the ~40-100 ms before each onset, slow drift, and
    free-er movement inside silences. Returns monotone (t_in, t_out) knots."""
    hop = int(0.01 * SR); e = energy_db(x, hop); dur = len(x) / SR
    thr = max(-55.0, np.percentile(e, 98) - 35.0); act = e >= thr
    spread = rng.choice([0.015, 0.03, 0.05, 0.08, 0.12])
    jit = spread * rng.uniform(0.2, 0.6)
    ons = set(onsets(e, thr))
    knots = [(0.0, rng.normal(0, spread))]
    cur = knots[0][1]; k = 0; n = len(act)
    drift_rate = rng.normal(0, 0.03)                 # slow tempo drift (s per s)
    while k < n:
        t = k * 0.01
        if (k in ons) or (k > 0 and act[k] and not act[k - 1]):
            new = (rng.normal(0, spread) if not act[max(0, k - 30):k].any() else cur + rng.normal(0, jit))
            lead = rng.uniform(0.03, 0.10)
            knots.append((max(knots[-1][0] + 1e-3, t - lead), cur + drift_rate * lead))
            cur = new; knots.append((t + 1e-3, cur))
        k += 1
        cur += drift_rate * 0.01
        if abs(cur) > 3 * spread + 0.05: drift_rate = -np.sign(cur) * abs(drift_rate)
    knots.append((dur, cur))
    # monotone with bounded local rate; looser inside silence
    out = [(knots[0][0], knots[0][0] + knots[0][1])]
    for (ti, di) in knots[1:]:
        pt, po = out[-1]
        if ti <= pt + 1e-4: continue
        quiet = not act[min(n - 1, int(ti / 0.01))] and not act[min(n - 1, int(pt / 0.01))]
        lo, hi = (0.35, 2.8) if quiet else (0.7, 1.45)
        to = min(max(ti + di, po + lo * (ti - pt)), po + hi * (ti - pt))
        out.append((ti, to))
    return np.array(out)


@njit(cache=True)
def wsola(x, src_pos, N, Hs, tol):
    """Output grain k (centre k*Hs + N/2) reads x near src_pos[k] (grain start), searching
    +-tol samples for the best match to the natural continuation of the previous grain.
    Returns output signal and the actually used grain starts."""
    K = len(src_pos)
    y = np.zeros(K * Hs + N)
    wsum = np.zeros(K * Hs + N)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(N) + 0.5) / N)
    used = np.zeros(K, np.int64)
    prev = -1
    L = len(x)
    for k in range(K):
        p = int(src_pos[k])
        best = p
        if prev >= 0:
            nat = prev + Hs
            if nat >= 0 and nat + N <= L:
                bc = -1e30
                for d in range(-tol, tol + 1):
                    q = p + d
                    if q < 0 or q + N > L: continue
                    c = 0.0
                    for s in range(0, N, 2):
                        c += x[q + s] * x[nat + s]
                    if c > bc: bc = c; best = q
        best = min(max(best, 0), L - N)
        used[k] = best
        o = k * Hs
        for s in range(N):
            y[o + s] += x[best + s] * w[s]; wsum[o + s] += w[s]
        prev = best
    for i in range(len(y)):
        if wsum[i] > 1e-3: y[i] /= wsum[i]
    return y, used


def warp_audio(x, rng):
    kn = make_warp(x, rng)                          # t_in -> t_out
    N, Hs, tol = 1764, 441, int(rng.choice([66, 110, 176]))
    K = int(len(x) / Hs)
    centres_out = (np.arange(K) * Hs + N / 2) / SR
    src_c = np.interp(centres_out, kn[:, 1], kn[:, 0])
    src_start = np.clip(np.round(src_c * SR - N / 2), 0, len(x) - N)
    y, used = wsola(x.astype(np.float64), src_start, N, Hs, tol)
    y = y[:len(x)].astype(np.float32)
    # actual correspondence: output grain centre <-> input grain centre
    t_out = centres_out
    t_in = (used + N / 2) / SR
    keep = np.r_[True, np.diff(t_in) > 0]
    return y, np.stack([t_in[keep], t_out[keep]], 1)


# ------------------------------------------------------------------ voice / colour
def rb(x, args, tmp):
    sf.write(f'{tmp}/i.wav', x, SR, subtype='FLOAT')
    subprocess.run(['rubberband', '-q', '--fine', *args, f'{tmp}/i.wav', f'{tmp}/o.wav'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    y = sf.read(f'{tmp}/o.wav', dtype='float32')[0]
    y = y.mean(1) if y.ndim > 1 else y
    return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)]


def pitch_contour(x, rng):
    """v2: a singer's own pitch behaviour: vibrato (own rate/depth, faded in on long notes),
    note-onset scoops, slow drift and per-phrase detune. Semitone offsets per 10 ms."""
    hop = int(0.01 * SR); e = energy_db(x, hop); n = len(e)
    thr = max(-55.0, np.percentile(e, 98) - 35.0); act = e >= thr
    t = np.arange(n) * 0.01
    p = np.zeros(n)
    if rng.random() < 0.7:
        rate = rng.uniform(4.5, 7.0); depth = rng.uniform(0.1, 0.7)
        run = np.zeros(n)                            # vibrato grows with time since onset
        for k in range(1, n): run[k] = run[k - 1] + 0.01 if act[k] else 0.0
        p += depth * np.clip((run - 0.25) / 0.3, 0, 1) * np.sin(2 * np.pi * rate * t + rng.uniform(0, 6.3))
    p += rng.uniform(0, 0.3) * np.sin(2 * np.pi * rng.uniform(0.05, 0.4) * t + rng.uniform(0, 6.3))
    if rng.random() < 0.6:                           # scoops into onsets
        for k in onsets(e, thr):
            if rng.random() < 0.5:
                d = rng.uniform(-1.5, 0.5); L = int(rng.uniform(5, 15))
                p[k:k + L] += d * np.linspace(1, 0, len(p[k:k + L]))
    return p


def revoice(x, pitch, formant, tmp, contour=None):
    x = x / (np.abs(x).max() + 1e-9) * 0.5
    pm = []
    if contour is not None:
        np.savetxt(f'{tmp}/pm.txt', np.stack([np.arange(len(contour)) * int(0.01 * SR), contour], 1), fmt='%d %.4f')
        pm = ['--pitchmap', f'{tmp}/pm.txt']
    if abs(formant) > 0.05:
        x = rb(x, ['-p', f'{formant:.3f}'], tmp)                 # pitch+formant move together
        rest = pitch - formant
        if abs(rest) > 0.01 or pm: x = rb(x, ['--formant', '-p', f'{rest:.3f}', *pm], tmp)
    elif abs(pitch) > 0.01 or pm:
        x = rb(x, ['--formant', '-p', f'{pitch:.3f}', *pm], tmp)
    return x


def biquad_peak(f0, gain_db, q):
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / SR; al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A]); a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    return b / a[0], a / a[0]


def colour(x, rng):
    y = x.astype(np.float64)
    for _ in range(rng.integers(0, 4)):
        b, a = biquad_peak(np.exp(rng.uniform(np.log(120), np.log(9000))), rng.uniform(-10, 10), rng.uniform(0.4, 2.5))
        y = lfilter(b, a, y)
    if rng.random() < 0.5:                             # tilt
        t = rng.uniform(-0.6, 0.6); y = y + t * np.diff(y, prepend=y[:1]) * 3
    if rng.random() < 0.3:                             # high-pass / telephone-ish low-pass
        c = rng.uniform(0.9, 0.99); y = lfilter([1, -1], [1, -c], y)
    if rng.random() < 0.15:
        c = rng.uniform(0.2, 0.6); y = lfilter([1 - c], [1, -c], y)
    if rng.random() < 0.4:                             # drive
        g = rng.uniform(1.5, 8); y = np.tanh(g * y / (np.abs(y).max() + 1e-9)) / np.tanh(g)
    if rng.random() < 0.4:                             # room / plate
        rt = rng.uniform(0.15, 1.2); L = int(rt * SR)
        ir = rng.standard_normal(L) * np.exp(-6.9 * np.arange(L) / L)
        ir[0] = 0; wet = rng.uniform(0.05, 0.35)
        r = fftconvolve(y, ir)[:len(y)]; r *= np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(r ** 2)) + 1e-12)
        y = (1 - wet) * y + wet * r
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    snr = rng.uniform(15, 60)
    nz = rng.standard_normal(len(y))
    if rng.random() < 0.5: nz = lfilter([1], [1, -0.97], nz); nz /= nz.std()
    y = y + nz * rms * 10 ** (-snr / 20)
    return (y / (np.abs(y).max() + 1e-9) * rng.uniform(0.2, 0.9)).astype(np.float32)


def pick_pitches(rng, v=1):
    pa = rng.uniform(-5, 6) if v == 1 else rng.uniform(-8, 10)
    r = rng.random()
    if v == 3:                                       # v3 (v7): octave/harmony-heavy, like real gang stacks
        if r < 0.3: pb = pa + rng.normal(0, 0.2)
        elif r < 0.7: pb = pa + rng.choice([-12, 12]) + rng.normal(0, 0.15)
        else: pb = pa + rng.choice([-9, -8, -7, -5, -4, -3, 3, 4, 5, 7, 8, 9]) + rng.normal(0, 0.15)
        pb = float(np.clip(pb, -15, 15))
        fa = np.clip(0.3 * pa + rng.normal(0, 1.5), -4, 4); fb = np.clip(0.3 * pb + rng.normal(0, 1.5), -4, 4)
        return pa, pb, fa, fb
    if r < 0.6: pb = pa + rng.normal(0, 0.2)                                  # unison double
    elif r < 0.85: pb = pa + rng.choice([-12, -7, -5, -4, -3, 3, 4, 5, 7, 12])   # harmony / octave
    else: pb = pa + rng.uniform(-2, 2)
    if v == 1:
        fa = rng.uniform(-3, 3) if rng.random() < 0.6 else 0.0
        fb = rng.uniform(-3, 3) if rng.random() < 0.6 else 0.0
    else:                                            # formants follow register (+- a personal offset)
        fa = np.clip(0.3 * pa + rng.normal(0, 1.5), -4, 4); fb = np.clip(0.3 * pb + rng.normal(0, 1.5), -4, 4)
    return pa, pb, fa, fb


# ------------------------------------------------------------------ one pair
def one_pair(args):
    seed, ids, v = args
    rng = np.random.default_rng(seed)
    src, cid = ids[rng.integers(len(ids))]
    x = np.load(f'{TR}/{src}/{cid:05d}.npy')
    L = int(CROP * SR)
    if len(x) > L:
        hop = int(0.01 * SR); e = energy_db(x, hop)
        for _ in range(10):
            s = rng.integers(0, len(x) - L)
            if (e[s // hop:(s + L) // hop] > np.percentile(e, 98) - 35).mean() > 0.4: break
        x = x[s:s + L]
    else:
        s = rng.integers(0, L - len(x) + 1); x = np.pad(x, (s, L - len(x) - s))
    x = x + rng.standard_normal(len(x)).astype(np.float32) * 1e-5
    y, kn = warp_audio(x, rng)                      # kn: (t_in, t_out)
    pa, pb, fa, fb = pick_pitches(rng, v)
    with tempfile.TemporaryDirectory() as tmp:
        A = colour(revoice(x, pa, fa, tmp, pitch_contour(x, rng) if v >= 2 else None), rng)
        B = colour(revoice(y, pb, fb, tmp, pitch_contour(y, rng) if v >= 2 else None), rng)
    A16 = librosa.resample(A, orig_sr=SR, target_sr=16000)[:int(CROP * 16000)]
    B16 = librosa.resample(B, orig_sr=SR, target_sr=16000)[:int(CROP * 16000)]
    # truth on HuBERT frames (centre 0.02 i + 0.0125 s)
    tA = 0.02 * np.arange(NF) + 0.0125
    tB = np.interp(tA, kn[:, 0], kn[:, 1])
    jf = (tB - 0.0125) / 0.02
    hop = int(0.02 * SR); ex = energy_db(np.pad(x, (int(0.0025 * SR), hop)), hop)[:NF]
    valid = (ex > np.percentile(ex, 98) - 35) & (tA > kn[0, 0] + 0.03) & (tA < kn[-1, 0] - 0.03) & (jf >= 0) & (jf <= NF - 1)
    return A16.astype(np.float16), B16.astype(np.float16), jf.astype(np.float32), valid, np.array([pa, pb, fa, fb], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('n', type=int)
    ap.add_argument('--split', default='train'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=12); ap.add_argument('--v', type=int, default=1)
    ap.add_argument('--src', default='src')          # comma list of chunk dirs, e.g. src,src_vs
    a = ap.parse_args()
    ids = sum((split_ids(a.split, s) for s in a.src.split(',')), [])
    os.makedirs(f'{TR}/pairs', exist_ok=True)
    S = int(CROP * 16000)
    XA = np.lib.format.open_memmap(f'{TR}/pairs/{a.name}_A.npy', 'w+', np.float16, (a.n, S))
    XB = np.lib.format.open_memmap(f'{TR}/pairs/{a.name}_B.npy', 'w+', np.float16, (a.n, S))
    JF = np.zeros((a.n, NF), np.float32); V = np.zeros((a.n, NF), bool); P = np.zeros((a.n, 4), np.float32)
    with mp.Pool(a.jobs) as pool:
        for i, (A, B, jf, v, p) in enumerate(pool.imap(one_pair, [(a.seed * 1000003 + i, ids, a.v) for i in range(a.n)], chunksize=4)):
            XA[i] = A; XB[i] = B; JF[i] = jf; V[i] = v; P[i] = p
            if i % 500 == 0: print(i, flush=True)
    XA.flush(); XB.flush()
    np.savez(f'{TR}/pairs/{a.name}_meta.npz', jf=JF, valid=V, pitch=P)
    print('done', a.n, 'valid frac', V.mean())


if __name__ == '__main__':
    main()
