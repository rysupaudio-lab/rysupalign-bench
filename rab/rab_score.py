#!/usr/bin/env python3
"""Run an aligner over RAB and score it.

usage: python rab/rab_score.py NAME RUNNER [--suite R|S|D] [--filter substr] [--exclude id,id]
                               [--jobs 8] [--env K=V,K=V] [--model PATH] [--save-errors]

Runner contract: RUNNER is called once per case as
    RUNNER MODEL GUIDE.wav DOUBLE.wav
(a .py RUNNER is started with the current Python) and must print, as its LAST line of stdout, a JSON
object with "warp": [[guideTime, doubleTime], ...] in seconds, monotonic, meaning "the double's
material that belongs at guideTime is at doubleTime in the double file" (linearly interpolated; an
empty list = identity = do nothing). Other keys are optional ("moved", "seconds", "cand").
MODEL is --model / env RAB_MODEL (default '-'); runners may ignore it. Any aligner (a feature DTW,
a neural model, a commercial tool's output converted to a warp) can be scored this way.

Metric: absolute timing error in ms between where the aligner says the double's
material is and where it truly is, at
  suite R: every matched human-corrected event (voicing onset / note change),
  suite S: every 10 ms frame where the original take is singing.
Reported: mean, median, p90, % within 10/20/50 ms, and the identity (do-nothing) baseline.
"""
import argparse, json, os, subprocess, sys, concurrent.futures as cf
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
CASES = os.environ.get('RAB_CASES', os.path.join(REPO, 'cases'))
RESULTS = os.path.join(REPO, 'results')
MODEL = os.environ.get('RAB_MODEL', '-')


def interp(knots, t):
    if not knots: return t
    k = np.asarray(knots); return np.interp(t, k[:, 0], k[:, 1])


def case_errors(cid, warp):
    truth = json.load(open(f'{CASES}/{cid}/truth.json'))
    if truth['kind'] == 'events':
        p = np.asarray(truth['pairs']); g, d = p[:, 0], p[:, 1]
    else:
        tw = np.asarray(truth['warp'])
        g = np.concatenate([np.arange(a, b, 0.01) for a, b in truth['voiced']]) if truth['voiced'] else np.array([])
        d = np.interp(g, tw[:, 0], tw[:, 1])
    pred = interp(warp, g)
    return np.abs(pred - d) * 1000, np.abs(g - d) * 1000


def run_case(runner, cid, env):
    e = dict(os.environ); e.update(env)
    cmd = ([sys.executable] if runner.endswith('.py') else []) + [runner, MODEL, f'{CASES}/{cid}/guide.wav', f'{CASES}/{cid}/double.wav']
    p = subprocess.run(cmd, capture_output=True, text=True, env=e)
    out = p.stdout.strip().splitlines()
    try: res = json.loads(out[-1])
    except Exception:
        res = {'error': 'no json', 'warp': []}
        print(f'runner failed on {cid} (scored as identity): {p.stderr.strip()[-400:]}', file=sys.stderr, flush=True)
    err, raw = case_errors(cid, res.get('warp', []))
    return cid, res, err, raw


def summary(errs):
    a = np.concatenate(errs) if errs else np.array([0.0])
    return dict(mean=a.mean(), median=np.median(a), p90=np.percentile(a, 90),
                w10=100 * (a <= 10).mean(), w20=100 * (a <= 20).mean(), w50=100 * (a <= 50).mean())


def main():
    global MODEL
    ap = argparse.ArgumentParser(description='Run an aligner over RAB and score it (see module docstring).')
    ap.add_argument('name', help='result name: writes results/NAME.json')
    ap.add_argument('runner', help='executable called as RUNNER MODEL GUIDE.wav DOUBLE.wav')
    ap.add_argument('--suite', default='', help='R, S or D (default: every built case of R and S)')
    ap.add_argument('--filter', default='', help='only case ids containing this substring')
    ap.add_argument('--exclude', default='', help='comma-separated case ids to leave out (e.g. the R truth bug cases)')
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--env', default='', help='K=V,K=V passed to the runner')
    ap.add_argument('--model', default=None, help='MODEL argument for the runner (default env RAB_MODEL or "-")')
    ap.add_argument('--save-errors', action='store_true', help='also write results/NAME.errors.npz (per-event errors)')
    a = ap.parse_args()
    if a.model: MODEL = a.model
    env = dict(kv.split('=', 1) for kv in a.env.split(',') if kv)
    excl = set(x for x in a.exclude.split(',') if x)
    man = [c for c in json.load(open(f'{ROOT}/manifest.json'))
           if (c['suite'] == a.suite if a.suite else c['suite'] in 'RS') and a.filter in c['id'] and c['id'] not in excl
           and os.path.exists(f"{CASES}/{c['id']}/truth.json")]
    if not man: raise SystemExit(f'no built cases found in {CASES} (run rab/build_rab.py / build_dcs.py first)')
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda c: run_case(a.runner, c['id'], env), man))
    os.makedirs(RESULTS, exist_ok=True)
    rows = {}
    for cid, r, err, raw in res:
        rows[cid] = dict(moved=r.get('moved'), seconds=r.get('seconds'), cand=r.get('cand'),
                         median=float(np.median(err)), raw_median=float(np.median(raw)),
                         mean=float(err.mean()), raw_mean=float(raw.mean()), n=int(len(err)))
    json.dump(rows, open(f'{RESULTS}/{a.name}.json', 'w'), indent=1)
    if a.save_errors:
        np.savez_compressed(f'{RESULTS}/{a.name}.errors.npz', **{f'{cid}|err': err for cid, r, err, raw in res},
                            **{f'{cid}|raw': raw for cid, r, err, raw in res})
    for suite in sorted({c['suite'] for c in man}):
        ids = [c['id'] for c in man if c['suite'] == suite]
        E = [r[2] for r in res if r[0] in ids]; R = [r[3] for r in res if r[0] in ids]
        s, b = summary(E), summary(R)
        worse = sum(1 for cid in ids if rows[cid]['mean'] > rows[cid]['raw_mean'] + 2)
        print(f'{a.name:28s} suite {suite} ({len(ids)} cases) | mean {s["mean"]:6.1f} median {s["median"]:5.1f} p90 {s["p90"]:6.1f} ms '
              f'| <=10ms {s["w10"]:4.1f}% <=20ms {s["w20"]:4.1f}% <=50ms {s["w50"]:4.1f}% | worse-than-raw {worse} '
              f'| raw: mean {b["mean"]:5.1f} median {b["median"]:5.1f} <=20ms {b["w20"]:4.1f}%', flush=True)
        if suite == 'S':          # per S variant (id suffix _t time-only, _s, _h harmony, _o octave)
            for var in sorted({i.rsplit('_', 1)[-1] for i in ids}):
                vi = [i for i in ids if i.endswith('_' + var)]
                Ev = [r[2] for r in res if r[0] in vi]; sv = summary(Ev)
                wv = sum(1 for cid in vi if rows[cid]['mean'] > rows[cid]['raw_mean'] + 2)
                print(f'{"":28s}   S_{var} ({len(vi):2d}) | mean {sv["mean"]:6.1f} median {sv["median"]:5.1f} p90 {sv["p90"]:6.1f} ms '
                      f'| <=20ms {sv["w20"]:4.1f}% | worse-than-raw {wv}', flush=True)


if __name__ == '__main__':
    main()
