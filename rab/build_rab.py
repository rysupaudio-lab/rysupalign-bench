#!/usr/bin/env python3
"""RysUp Align Bench (RAB) builder for suites R and S. Suite D: build_dcs.py.

Suite R  real unison pairs, real truth. Choral Singing Dataset (Cuesta et al. 2018, CC BY 4.0):
         16 singers recorded on separate close mics, unison per section, manually corrected
         per-singer f0 (5.8 ms hop). Truth events = each singer's voicing onsets and note changes,
         matched singer-to-singer in order.
Suite S  exact-truth cut-and-slide cases on real backing-vocal / gang takes (Cambridge-MT multitrack
         stems + the fury gang takes in data/fury). guide = the take re-voiced (tilt EQ, -45 dB noise;
         variant h also +4 st, variant o +12 st, formants kept, Rubber Band), double = the same take cut at
         silences / syllable dips and each segment slid by its own offset (8 ms crossfades, no stretching),
         so the truth is sample-exact. Variants: t typical (45 ms spread), s sloppy (90 ms), h harmony
         (+4 st guide), o octave (+12 st guide).

usage:
  python rab/build_rab.py R --csd DIR            DIR = the unzipped ChoralSingingDataset folder
  python rab/build_rab.py S --mt DIR [--fury DIR] DIR = folder holding the unzipped Cambridge-MT sessions
                                                   (searched recursively for <session>/.../<stem>.wav)
  options: --out CASES (default $RAB_CASES or <repo>/cases), --copy (copy R audio instead of symlinking)

Every case's truth is checked against the sha256 recorded in rab/manifest.json. S cases are rebuilt from
the per-case random-generator states in rab/s_cases.json, so each case is reproducible on its own; the
double (and its truth) is bit-exact, the h/o guides depend on your Rubber Band version (4.0.0 was used).

Output: CASES/<id>/{guide.wav,double.wav,truth.json}
truth.json: {"kind":"events","pairs":[[guideTime,doubleTime],...]}
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile, warnings
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
rng = np.random.default_rng(20261006)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


def truth_sha(pairs):
    return hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest()


def audio_sha(x):
    """sha256 of the mono float32 samples (how a source take is identified)."""
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


# ---------------------------------------------------------------- suite R
def f0_events(path):
    d = np.loadtxt(path)
    t, f = d[:, 0], d[:, 1]
    voiced = f > 0
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f, 1e-9) / 440.0), np.nan)
    ev = []
    i = 1
    while i < len(t):
        if voiced[i] and not voiced[i - 1]:
            # require >= 60 ms voiced and >= 60 ms silence before
            run = np.argmax(~voiced[i:]) if (~voiced[i:]).any() else len(t) - i
            gap = i - (np.where(voiced[:i])[0][-1] + 1 if voiced[:i].any() else 0)
            if run * 0.0058 >= 0.06 and gap * 0.0058 >= 0.06:
                ev.append(('on', t[i], np.nanmedian(midi[i:i + 17])))
        elif voiced[i] and voiced[i - 1] and i >= 9 and i + 9 < len(t):
            a, b = np.nanmedian(midi[i - 9:i]), np.nanmedian(midi[i:i + 9])
            prev_jump = abs(np.nanmedian(midi[i - 10:i - 1]) - np.nanmedian(midi[i - 1:i + 8]))
            if abs(b - a) >= 0.8 and abs(b - a) >= prev_jump:   # local maximum of the step
                ev.append(('note', t[i], b))
                i += 9
        i += 1
    return ev


def match_events(eg, ed, window=0.25):
    """Monotonic one-to-one match of same-kind, same-note events within +-window s."""
    pairs, j0 = [], 0
    for kind, tg, mg in eg:
        best, bj = None, None
        for j in range(j0, len(ed)):
            k, td, md = ed[j]
            if td > tg + window: break
            if k != kind or td < tg - window or not np.isfinite(md) or abs(md - mg) > 1.0: continue
            if best is None or abs(td - tg) < abs(best - tg): best, bj = td, j
        if best is not None:
            pairs.append([float(tg), float(best)]); j0 = bj + 1
    return pairs


def suite_r(CHORAL, OUT, copy=False):
    cases = []
    for piece in ('ER', 'LI', 'ND'):
        for sec in ('soprano', 'alto', 'tenor', 'bass'):
            ev = {k: f0_events(f'{CHORAL}/CSD_{piece}_{sec}_{k}.f0') for k in (1, 2, 3, 4)}
            for g, dd in ((1, 2), (1, 3), (1, 4), (2, 3)):
                pairs = match_events(ev[g], ev[dd])
                if len(pairs) < 20: continue
                cid = f'R_{piece}_{sec}_{g}{dd}'
                os.makedirs(f'{OUT}/{cid}', exist_ok=True)
                for name, k in (('guide', g), ('double', dd)):
                    dst = f'{OUT}/{cid}/{name}.wav'
                    src = os.path.abspath(f'{CHORAL}/CSD_{piece}_{sec}_{k}.wav')
                    if os.path.lexists(dst): os.remove(dst)
                    if copy:
                        import shutil; shutil.copyfile(src, dst)
                    else:
                        os.symlink(src, dst)
                json.dump({'kind': 'events', 'pairs': pairs}, open(f'{OUT}/{cid}/truth.json', 'w'))
                cases.append({'id': cid, 'suite': 'R', 'events': len(pairs), 'tags': ['choir', 'unison', sec]})
    return cases


# ---------------------------------------------------------------- suite S
def mono(x): return x.mean(1) if x.ndim > 1 else x


def phrases(x, sr):
    hop = int(0.01 * sr); n = len(x) // hop
    e = 10 * np.log10(np.array([np.mean(x[i * hop:(i + 1) * hop] ** 2) for i in range(n)]) + 1e-12)
    thr = max(-50.0, np.percentile(e, 95) - 35.0); act = e >= thr
    out, i = [], 0
    while i < n:
        if not act[i]: i += 1; continue
        s, last = i, i
        while i < n and (act[i] or i - last < 6):
            if act[i]: last = i
            i += 1
        if (last + 1 - s) * 0.01 >= 0.15: out.append((s * 0.01, (last + 1) * 0.01))
    return out


def onsets(x, sr, a, b):
    """Energy-rise onsets inside [a,b] seconds (5 ms hop)."""
    hop = int(0.005 * sr); i0, i1 = int(a * sr) // hop, int(b * sr) // hop
    e = np.array([10 * np.log10(np.mean(x[k * hop:k * hop + 2 * hop] ** 2) + 1e-12) for k in range(i0, i1)])
    res, last = [], -99
    for k in range(4, len(e)):
        if e[k] - e[k - 4:k].min() >= 6 and k - last >= 24:
            res.append((i0 + k) * hop / sr); last = k
    return res


def revoice(x, sr, semitones, tmp):
    """Same timing, different sound: optional pitch shift (formants kept), a tilt EQ and a little noise."""
    y = x
    if semitones:
        sf.write(f'{tmp}/rv_in.wav', x.astype(np.float32), sr, subtype='FLOAT')
        subprocess.run(['rubberband', '-q', '--fine', '--formant', '-p', str(semitones), f'{tmp}/rv_in.wav', f'{tmp}/rv_out.wav'], check=True, capture_output=True)
        y = mono(sf.read(f'{tmp}/rv_out.wav', dtype='float32')[0]); y = np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)]
    # tilt: +4 dB/oct-ish brightening via first-difference blend, then 45 dB-down noise
    y = 0.7 * y + 0.6 * np.diff(y, prepend=y[:1])
    y = y + rng.normal(0, 1, len(y)).astype(np.float32) * (np.sqrt(np.mean(y ** 2)) * 10 ** (-45 / 20))
    return (y / (np.abs(y).max() + 1e-9) * 0.7).astype(np.float32)


def splice_points(x, sr):
    """Cut points: middles of silences and syllable dips (>= 12 dB under both neighbours
    within +-80 ms), never closer than 120 ms. Energy on a 5 ms hop."""
    hop = int(0.005 * sr); n = len(x) // hop
    e = np.array([10 * np.log10(np.mean(x[k * hop:k * hop + 2 * hop] ** 2) + 1e-12) for k in range(n)])
    thr = max(-50.0, np.percentile(e, 95) - 35.0)
    pts, last = [], -999
    for k in range(16, n - 16):
        if k - last < 24: continue
        lo = e[k]
        if lo > min(e[k - 3:k + 4]): continue                       # local minimum only
        quiet = lo < thr
        dip = (e[k - 16:k].max() - lo >= 12) and (e[k + 1:k + 17].max() - lo >= 12)
        if quiet or dip:
            pts.append(k * hop); last = k
    return pts


def wreck_splice(x, sr, spread):
    """Cut and slide: every segment between cut points gets its own offset (phrase base plus
    per-syllable jitter); 8 ms equal-power crossfades at the cuts. Returns (y, truth pairs on a
    10 ms grid of voiced original frames as [orig_time, double_time])."""
    cuts = [0] + splice_points(x, sr) + [len(x)]
    ph = phrases(x, sr)
    def phrase_of(t):
        for k, (a, b) in enumerate(ph):
            if a - 0.05 <= t <= b + 0.05: return k
        return -1
    base = {k: rng.normal(0, spread) for k in range(len(ph))}
    offs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        k = phrase_of((a + b) / 2 / sr)
        offs.append(int(round(((base.get(k, 0.0) if k >= 0 else 0.0) + rng.normal(0, spread * 0.35)) * sr)))
    # keep segment order: each segment may not start before the previous one ends minus 40 ms
    y = np.zeros(len(x) + int(sr), np.float32); fade = int(0.008 * sr)
    placed, prev_end = [], 0
    occ = np.zeros(len(x) + int(sr), np.int16)
    for (a, b), o in zip(zip(cuts[:-1], cuts[1:]), offs):
        start = max(a + o, prev_end - int(0.04 * sr), 0)
        seg = x[a:b].astype(np.float32).copy()
        f = min(fade, len(seg) // 2)
        if f > 0:
            seg[:f] *= np.sin(np.linspace(0, np.pi / 2, f)) ; seg[-f:] *= np.cos(np.linspace(0, np.pi / 2, f))
        y[start:start + len(seg)] += seg
        occ[start:start + len(seg)] += 1
        placed.append((a, b, start - a)); prev_end = start + len(seg)
    y = y[:len(x)]
    pairs = []
    for a, b, o in placed:
        for t in np.arange(np.ceil(a / sr / 0.01) * 0.01, b / sr - 0.012, 0.01):   # skip crossfades
            if t * sr < a + fade or t * sr > b - fade: continue
            k = int(t * sr); seg = x[k:k + int(0.01 * sr)]
            if np.sqrt(np.mean(seg ** 2)) < 10 ** (-40 / 20) * (np.abs(x).max() + 1e-9): continue
            j = int((t + o / sr) * sr)
            if j + int(0.01 * sr) >= len(x) or occ[j:j + int(0.01 * sr)].max() > 1: continue
            pairs.append([round(float(t), 4), round(float(t + o / sr), 6)])
    return y, pairs


def find_stem(mt, session, stem):
    """<mt>/<session>/**/<stem>.wav without following directory symlinks."""
    base = os.path.join(mt, session)
    for root, dirs, files in os.walk(base if os.path.isdir(base) else mt, followlinks=False):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d)) and not d.startswith('__MACOSX')]
        if f'{stem}.wav' in files and (os.path.isdir(base) or session in root): return os.path.join(root, f'{stem}.wav')
    return None


def suite_s(MT, FURY, OUT, only=None):
    """Exact-truth self-recovery: guide = the take re-voiced with its timing untouched,
    double = the same take cut-and-slid per syllable (no stretching, sample-exact truth)."""
    global rng
    srcs = {s['id']: s for s in json.load(open(f'{HERE}/s_sources.json'))}
    plan = json.load(open(f'{HERE}/s_cases.json'))
    takes, cases, missing = {}, [], set()
    for c in plan:
        if only and not any(o in c['id'] for o in only): continue
        src = srcs[c['source']]
        if src['id'] not in takes:
            p = os.path.join(FURY, src['file']) if src['kind'] == 'gang' else (find_stem(MT, src['session'], src['stem']) if MT else None)
            if not p or not os.path.exists(p):
                missing.add(src['id']); takes[src['id']] = None; continue
            x, sr = sf.read(p, dtype='float32'); x = mono(x).astype(np.float32)
            if audio_sha(x) != src['sha256']:
                print(f'WARNING {src["id"]}: audio differs from the published source (sha256); truth will not match', flush=True)
            takes[src['id']] = (x, sr)
        if takes[src['id']] is None: continue
        x, sr = takes[src['id']]
        st = c['rng_state']
        rng = np.random.default_rng()
        rng.bit_generator.state = {'bit_generator': st['bit_generator'],
                                   'state': {k: int(v) for k, v in st['state'].items()},
                                   'has_uint32': st['has_uint32'], 'uinteger': st['uinteger']}
        cid = c['id']
        os.makedirs(f'{OUT}/{cid}', exist_ok=True)
        y, pairs = wreck_splice(x, sr, c['spread'])
        with tempfile.TemporaryDirectory() as tmp:
            g = revoice(x, sr, c['semitones'], tmp)
        for name, sig in (('guide', g), ('double', y)):
            fp = f'{OUT}/{cid}/{name}.wav'
            if os.path.lexists(fp): os.remove(fp)
            sf.write(fp, sig, sr, subtype='PCM_24')
        json.dump({'kind': 'events', 'pairs': pairs, 'source': src['id']}, open(f'{OUT}/{cid}/truth.json', 'w'))
        cases.append(cid)
        print(cid, len(pairs), flush=True)
    for m in sorted(missing): print('missing source (cases skipped):', m, srcs[m].get('download', ''))
    return cases


def verify(OUT, ids):
    man = {c['id']: c for c in json.load(open(f'{HERE}/manifest.json'))}
    bad = 0
    for cid in ids:
        pairs = json.load(open(f'{OUT}/{cid}/truth.json'))['pairs']
        if truth_sha(pairs) != man[cid]['truth_sha256']:
            bad += 1; print('TRUTH MISMATCH', cid, flush=True)
    print(f'{len(ids) - bad}/{len(ids)} cases match the published truth')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('suites', nargs='+', choices=['R', 'S'])
    ap.add_argument('--csd', help='unzipped ChoralSingingDataset folder (suite R)')
    ap.add_argument('--mt', help='folder with the unzipped Cambridge-MT sessions (suite S)')
    ap.add_argument('--fury', default=os.path.join(REPO, 'data', 'fury'), help='fury gang takes (suite S)')
    ap.add_argument('--out', default=os.environ.get('RAB_CASES', os.path.join(REPO, 'cases')))
    ap.add_argument('--copy', action='store_true', help='copy suite-R audio instead of symlinking it')
    ap.add_argument('--only', default='', help='comma-separated id substrings (suite S)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    built = []
    if 'R' in a.suites:
        if not a.csd: ap.error('suite R needs --csd')
        built += [c['id'] for c in suite_r(a.csd, a.out, a.copy)]
    if 'S' in a.suites:
        if not a.mt: print('no --mt given: only the fury cases of suite S are built')
        built += suite_s(a.mt, a.fury, a.out, [o for o in a.only.split(',') if o])
    sys.exit(1 if verify(a.out, built) else 0)


if __name__ == '__main__':
    main()
