#!/usr/bin/env python3
"""Suite D (held out, never used for training or model selection): Dagstuhl ChoirSet section takes,
pairs of DIFFERENT singers singing the same part together (Rosenzweig et al., TISMIR 2020, CC BY 4.0).

Audio = each singer's headset (HSM) or dynamic (DYN) mic. Truth events = voicing onsets and note
changes from CREPE f0 on each singer's LARYNX (LRX) mic, which carries no bleed from neighbours,
matched singer-to-singer in order (same rules as suite R).
Caveats: CREPE truth is automatic (not hand-corrected); open mics carry some neighbour bleed.
Suite D was evaluated once, at the end of the study (see README); please keep it as a held-out set.

usage: python rab/build_dcs.py DCS_ROOT [--out CASES]
  DCS_ROOT = the unzipped DagstuhlChoirSet_V1.2.3 folder (Zenodo 10.5281/zenodo.4618287)
"""
import argparse, glob, itertools, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from build_rab import verify


def events(csv):
    a = np.loadtxt(csv, delimiter=','); t, f, c = a[:, 0], a[:, 1], a[:, 2]
    v = (c > 0.5) & (f > 50)
    midi = np.where(v, 69 + 12 * np.log2(np.maximum(f, 1) / 440), np.nan)
    ev, i = [], 1
    while i < len(t):
        if v[i] and not v[i - 1]:
            run = np.argmax(~v[i:]) if (~v[i:]).any() else len(t) - i
            gap = i - (np.where(v[:i])[0][-1] + 1 if v[:i].any() else 0)
            if run >= 6 and gap >= 6:
                ev.append(('on', t[i], np.nanmedian(midi[i:i + 10])))
        elif v[i] and v[i - 1] and 6 <= i < len(t) - 5:
            a_, b_ = np.nanmedian(midi[i - 5:i]), np.nanmedian(midi[i:i + 5])
            prev = abs(np.nanmedian(midi[i - 6:i - 1]) - np.nanmedian(midi[i - 1:i + 4]))
            if abs(b_ - a_) >= 0.8 and abs(b_ - a_) >= prev:
                ev.append(('note', t[i], b_)); i += 5
        i += 1
    return ev


def match(eg, ed, window=0.25):
    pairs, j0 = [], 0
    for k, tg, mg in eg:
        best = bj = None
        for j in range(j0, len(ed)):
            kk, td, md = ed[j]
            if td > tg + window: break
            if kk != k or td < tg - window or not np.isfinite(md) or abs(md - mg) > 1.0: continue
            if best is None or abs(td - tg) < abs(best - tg): best, bj = td, j
        if best is not None:
            pairs.append([float(tg), float(best)]); j0 = bj + 1
    return pairs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dcs_root')
    ap.add_argument('--out', default=os.environ.get('RAB_CASES', os.path.join(os.path.dirname(ROOT), 'cases')))
    a = ap.parse_args(); D, OUT = a.dcs_root, a.out
    cases = []
    takes = sorted({os.path.basename(p).rsplit('_', 2)[0] for p in glob.glob(f'{D}/audio_wav_22050_mono/*_LRX.wav')})
    for take in takes:
        if 'Quartet' in take or 'FullChoir' in take: continue          # sections only: same part
        def audio(s):
            for mic in ('HSM', 'DYN'):
                p = f'{D}/audio_wav_22050_mono/{take}_{s}_{mic}.wav'
                if os.path.exists(p): return p
        singers = sorted({os.path.basename(p).split('_')[-2] for p in glob.glob(f'{D}/audio_wav_22050_mono/{take}_*_LRX.wav')})
        ev = {s: events(f'{D}/annotations_csv_F0_CREPE/{take}_{s}_LRX.csv') for s in singers
              if os.path.exists(f'{D}/annotations_csv_F0_CREPE/{take}_{s}_LRX.csv') and audio(s)}
        for a_, b_ in list(itertools.combinations(sorted(ev), 2))[:3]:
            pairs = match(ev[a_], ev[b_])
            if len(pairs) < 15: continue
            cid = 'D_' + take.replace('DCS_', '') + f'_{a_}{b_}'
            os.makedirs(f'{OUT}/{cid}', exist_ok=True)
            for nm, s in (('guide', a_), ('double', b_)):
                dst = f'{OUT}/{cid}/{nm}.wav'
                if os.path.lexists(dst): os.remove(dst)
                os.symlink(os.path.abspath(audio(s)), dst)
            json.dump({'kind': 'events', 'pairs': pairs, 'truth': 'CREPE on LRX mics'}, open(f'{OUT}/{cid}/truth.json', 'w'))
            cases.append(cid)
    print(len(cases), 'D cases')
    sys.exit(1 if verify(OUT, cases) else 0)


if __name__ == '__main__':
    main()
