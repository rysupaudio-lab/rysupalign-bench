#!/usr/bin/env python3
"""Print the README result tables from the result files in this folder.

Two kinds of input:
  results/<name>.json (+ <name>.errors.npz)  runs of the public code in this repository
      (rab_score.py ... --save-errors). With the per-event errors every pooled statistic is exact.
  results/original/*.json                     per-case results of the original research runs
      (2026-10-06..08). rab_score.py stores only per-case mean/median, so for these the pooled
      (event-weighted) MEAN is computed exactly here, while pooled median / <=20 ms / p90 are taken
      from the scorer output printed at the time (results/pooled_stats.json, "source": "logged") when it
      exists, and shown as '-' otherwise.

usage: python results/summarize.py             print the tables
       python results/summarize.py --refresh   recompute pooled_stats.json entries from the local
                                               *.errors.npz files (not in git: they are large)
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, 'original')
MAN = {c['id']: c for c in json.load(open(os.path.join(os.path.dirname(HERE), 'rab', 'manifest.json')))}
BAD_R = {'R_ND_tenor_13', 'R_ND_tenor_23'}      # invalid truth (CSD_ND_tenor_3.f0 duplicates tenor_2), see README
POOLED = os.path.join(HERE, 'pooled_stats.json')
LOGGED = json.load(open(POOLED))
R_ER = [c for c in MAN if c.startswith('R_ER_')]
S_FURY = [c for c in MAN if c.startswith('S_fury_') and c[-1] in 'tsh']


def load(name):
    p = os.path.join(HERE, name + '.json')
    if not os.path.exists(p): p = os.path.join(ORIG, name + '.json')
    rows = json.load(open(p))
    e = os.path.join(HERE, name + '.errors.npz')
    err = dict(np.load(e)) if os.path.exists(e) else None
    return rows, err


def pooled(name, ids, tag=None):
    """dict(mean, median, w20, p90, worse, n_cases) over the given case ids (None where unknown)."""
    rows, err = load(name)
    ids = [c for c in ids if c in rows]
    if not ids: return None
    n = sum(rows[c]['n'] for c in ids)
    out = dict(mean=sum(rows[c]['mean'] * rows[c]['n'] for c in ids) / n,
               raw=sum(rows[c]['raw_mean'] * rows[c]['n'] for c in ids) / n,
               worse=sum(rows[c]['mean'] > rows[c]['raw_mean'] + 2 for c in ids), cases=len(ids),
               median=None, w20=None, p90=None)
    if err is not None:
        a = np.concatenate([err[f'{c}|err'] for c in ids])
        out.update(median=float(np.median(a)), w20=float(100 * (a <= 20).mean()), p90=float(np.percentile(a, 90)))
    elif tag and tag in LOGGED.get(name, {}):
        out.update({k: v for k, v in LOGGED[name][tag].items() if k in ('median', 'w20', 'p90')})
    return out


def raw_pooled(name, ids, tag=None):
    rows, err = load(name)
    if err is None:
        ids = [c for c in ids if c in rows]; n = sum(rows[c]['n'] for c in ids)
        return dict(LOGGED['raw'][tag], mean=sum(rows[c]['raw_mean'] * rows[c]['n'] for c in ids) / n)
    a = np.concatenate([err[f'{c}|raw'] for c in ids if c in rows])
    return dict(mean=float(a.mean()), median=float(np.median(a)), w20=float(100 * (a <= 20).mean()), p90=float(np.percentile(a, 90)))


def f(x, d=1, pct=False):
    return '–' if x is None else (f'{x:.{d}f}%' if pct else f'{x:.{d}f}')


def suite_ids(s): return [c for c in MAN if MAN[c]['suite'] == s]


FULL = [  # (label, result name, licence / note)
    ('raw (no alignment)', None, ''),
    ('log-mel DTW', 'mel_dtw', ''),
    ('HuBERT-base L6', 'hubert6_dtw', 'Apache-2.0'),
    ('HuBERT-base L6 + log-mel', 'hubert6_mel_dtw', 'Apache-2.0'),
    ('HuBERT-base L6 + log-mel (gated)', 'hubert6_mel_gated_dtw', 'Apache-2.0'),
    ('WavLM-base+ L6', 'RS90_wavlm-base@6', 'original run'),
    ('MERT-v1-95M L4', 'R_mert95m@4', 'CC BY-NC; original run'),
    ('MERT-v1-95M L4 + log-mel', 'R_mert95m@4+mel', 'CC BY-NC; original run'),
    ('learned v6 alone', 'learned_v6_alone', 'ours'),
    ('learned v6 + log-mel (gated)', 'learned_v6_mel', 'ours'),
    ('learned v6 + log-mel + M1 DTW', 'learned_v6_mel_m1', 'ours'),
    ('learned v6 + log-mel + M1 DTW, int8 model', 'learned_v6_mel_m1_int8', 'ours'),
    ('RysUpAlign 2.1 engine (C++, int8)', 'R_engine_2.1', 'ours; original run'),
    ('RysUpAlign 2.0.4 engine (previous release)', 'R_engine_2.0.4', 'ours; original run'),
]


def table_R():
    R = suite_ids('R'); Rc = [c for c in R if c not in BAD_R]
    print('| system | R mean | R median | R ≤20 ms | R p90 | R worse than raw | Rc mean | Rc median | Rc ≤20 ms |')
    print('|---|---|---|---|---|---|---|---|---|')
    ref = 'learned_v6_mel_m1'
    raw, rawc = raw_pooled(ref, R, 'R'), raw_pooled(ref, Rc, 'Rc')
    print(f"| raw (no alignment) | {f(raw['mean'])} | {f(raw['median'])} | {f(raw['w20'], pct=True)} | {f(raw['p90'])} | – | {f(rawc['mean'])} | {f(rawc['median'])} | {f(rawc['w20'], pct=True)} |")
    for label, name, note in FULL[1:]:
        if not exists(name): continue
        a = pooled(name, R, 'R'); c = pooled(name, Rc, 'Rc')
        if a is None or a['cases'] < len(R): continue
        print(f"| {label} | {f(a['mean'])} | {f(a['median'])} | {f(a['w20'], pct=True)} | {f(a['p90'])} | {a['worse']}/48 | {f(c['mean'])} | {f(c['median'])} | {f(c['w20'], pct=True)} |")


def table_S():
    S = suite_ids('S')
    print('| system | S mean | S median | S p90 | worse than raw | t | s | h (+4 st) | o (+12 st) |')
    print('|---|---|---|---|---|---|---|---|---|')
    raw = raw_pooled('learned_v6_mel_m1', S, 'S')
    print(f"| raw (no alignment) | {f(raw['mean'])} | {f(raw['median'])} | {f(raw['p90'])} | – | " + ' | '.join(
        f(raw_pooled('learned_v6_mel_m1', [c for c in S if c.endswith('_' + v)], 'S_' + v)['mean']) for v in 'tsho') + ' |')
    for label, name, note in FULL[1:]:
        if name.startswith('R_') and exists('S_' + name[2:]): name = 'S_' + name[2:]
        if not exists(name): continue
        a = pooled(name, S, 'S')
        if a is None or a['cases'] < len(S): continue
        var = [pooled(name, [c for c in S if c.endswith('_' + v)], 'S_' + v) for v in 'tsho']
        print(f"| {label} | {f(a['mean'])} | {f(a['median'])} | {f(a['p90'])} | {a['worse']}/120 | " + ' | '.join(f(v['mean']) for v in var) + ' |')


SUBSET = [
    ('raw (no alignment)', None), ('log-mel DTW', 'mel_dtw'),
    ('HuBERT-base L6', 'Rsub_hubert@6'), ('HuBERT-base L6 + log-mel', 'hubert6_mel_dtw'), ('HuBERT-base L9', 'Rsub_hubert@9'), ('HuBERT-base L12', 'Rsub_hubert@12'),
    ('WavLM-base+ L6', 'Rsub_wavlm@6'), ('WavLM-base+ L9', 'Rsub_wavlm@9'), ('WavLM-base+ L12', 'Rsub_wavlm@12'),
    ('MERT-v1-95M L4', 'Rsub_mert@4'), ('MERT-v1-95M L4 + log-mel', 'Rsub_mert@4+mel'), ('MERT-v1-95M L7', 'Rsub_mert@7'), ('MERT-v1-95M L10', 'Rsub_mert@10'),
    ('MMS-300m aligner L12', 'Rsub_mms@12'), ('MMS-300m aligner L18', 'Rsub_mms@18'), ('MMS-300m aligner L24', 'Rsub_mms@24'),
    ('MERT-v1-330M L6', 'Rsub_mertL@6'), ('MERT-v1-330M L10', 'Rsub_mertL@10'), ('MERT-v1-330M L14', 'Rsub_mertL@14'),
    ('WavLM-large L8', 'Rsub_wavlmL@8'), ('WavLM-large L12', 'Rsub_wavlmL@12'), ('WavLM-large L16', 'Rsub_wavlmL@16'),
    ('HuBERT-large L8', 'Rsub_hubertL@8'), ('HuBERT-large L12', 'Rsub_hubertL@12'), ('HuBERT-large L16', 'Rsub_hubertL@16'),
    ('XLS-R 300M L8', 'Rsub_xlsr@8'), ('XLS-R 300M L12', 'Rsub_xlsr@12'), ('XLS-R 300M L16', 'Rsub_xlsr@16'),
    ('MMS-1B L16', 'Rsub_mms1b@16'), ('MMS-1B L24', 'Rsub_mms1b@24'),
    ('HubertFA SynthGT posteriors', 'Rsub_synthgt'), ('HubertFA SynthGT posteriors + log-mel', 'Rsub_synthgt+mel'),
    ('learned v6 alone', 'learned_v6_alone'), ('learned v6 + log-mel (gated)', 'learned_v6_mel'), ('learned v6 + log-mel + M1 DTW', 'learned_v6_mel_m1'),
]


def exists(name):
    return name and (os.path.exists(os.path.join(HERE, name + '.json')) or os.path.exists(os.path.join(ORIG, name + '.json')))


def table_subset():
    print('| features (dense DTW, no mel unless stated) | R_ER mean (16 cases) | worse than raw | S fury t/s/h mean (21 cases) | worse than raw |')
    print('|---|---|---|---|---|')
    for label, name in SUBSET:
        if name is None:
            r = pooled('learned_v6_mel_m1', R_ER); s = pooled('learned_v6_mel_m1', S_FURY)
            print(f"| {label} | {f(r['raw'])} | – | {f(s['raw'])} | – |"); continue
        sname = name.replace('Rsub_', 'Ssub_') if name.startswith('Rsub_') else name
        if not exists(name): continue
        r = pooled(name, R_ER); s = pooled(sname, S_FURY) if exists(sname) else None
        print(f"| {label} | {f(r['mean'])} | {r['worse']}/16 | {f(s['mean']) if s else '–'} | {str(s['worse']) + '/21' if s else '–'} |")


D_ROWS = [('HuBERT-base L6 + log-mel', 'D_hubert@6+mel'), ('MERT-v1-95M L4 + log-mel', 'D_mert@4+mel'),
          ('MERT-v1-330M L10 + log-mel', 'D_mertL@10+mel'), ('learned v6 + log-mel (gated)', 'D_v6'),
          ('RysUpAlign 2.1 engine (learned v6 + M1, C++)', 'D_engine_final')]


def table_D():
    D = suite_ids('D')
    print('| system | D mean | D median | D ≤20 ms | D p90 | cases better than raw |')
    print('|---|---|---|---|---|---|')
    a = pooled('D_v6', D); lg = LOGGED['D_v6']['raw']
    print(f"| raw (no alignment) | {f(a['raw'])} | {f(lg['median'])} | {f(lg['w20'], pct=True)} | – | – |")
    for label, name in D_ROWS:
        rows, _ = load(name); a = pooled(name, D, 'D')
        better = sum(rows[c]['mean'] < rows[c]['raw_mean'] for c in D)
        print(f"| {label} | {f(a['mean'])} | {f(a['median'])} | {f(a['w20'], pct=True)} | {f(a['p90'])} | {better}/12 |")


def refresh():
    tags = {'R': suite_ids('R'), 'Rc': [c for c in suite_ids('R') if c not in BAD_R], 'S': suite_ids('S'),
            **{'S_' + v: [c for c in suite_ids('S') if c.endswith('_' + v)] for v in 'tsho'}, 'D': suite_ids('D')}
    for fn in sorted(os.listdir(HERE)):
        if not fn.endswith('.errors.npz'): continue
        name = fn[:-len('.errors.npz')]; rows, err = load(name)
        for tag, ids in tags.items():
            ids = [c for c in ids if c in rows]
            if not ids or len(ids) < len(tags[tag]): continue
            a = np.concatenate([err[f'{c}|err'] for c in ids]); r = np.concatenate([err[f'{c}|raw'] for c in ids])
            LOGGED.setdefault(name, {})[tag] = dict(median=round(float(np.median(a)), 4), w20=round(float(100 * (a <= 20).mean()), 4),
                                                    p90=round(float(np.percentile(a, 90)), 4), source='computed from per-event errors')
            LOGGED.setdefault('raw', {})[tag] = dict(mean=round(float(r.mean()), 4), median=round(float(np.median(r)), 4),
                                                     w20=round(float(100 * (r <= 20).mean()), 4), p90=round(float(np.percentile(r, 90)), 4),
                                                     source='computed from per-event errors')
    json.dump(LOGGED, open(POOLED, 'w'), indent=1)
    print('updated', POOLED)


if __name__ == '__main__':
    import sys
    if '--refresh' in sys.argv: refresh(); sys.exit()
    print('## Suite R (48 cases, 5,365 events) and Rc (46 cases, without R_ND_tenor_13/23)\n'); table_R()
    print('\n## Suite S (120 cases, 126,525 frames)\n'); table_S()
    print('\n## Feature sweep on subsets (event-weighted mean error, ms)\n'); table_subset()
    print('\n## Suite D (12 held-out cases, 470 events; evaluated once)\n'); table_D()
