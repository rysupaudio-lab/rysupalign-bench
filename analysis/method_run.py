#!/usr/bin/env python3
"""Run method variants (method_variants.VARIANTS) over RAB R and S in-process.

usage: method_run.py TAG variant1,variant2,... [--suite R|S|RS] [--jobs 4]
One worker per case computes every requested variant (shared cost matrices), so variants are
compared on identical features. Prints R (all / even-index / odd-index cases) and S (all, p90, octave)
and saves per-event errors to results/method_<TAG>.npz.
R split: cases are ordered as in manifest.json; 'even' = index 0,2,4.. (tuning half), 'odd' = confirm half.
"""
import argparse, os, sys, concurrent.futures as cf
import numpy as np
_H = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(_H), 'runners'), os.path.join(os.path.dirname(_H), 'rab')]
import method_lib as ML, rabpaths


def work(args):
    cid, names = args
    import method_variants as MV
    z = ML.load(cid)
    out = {}
    for nm in names:
        g, d = MV.VARIANTS[nm](z)
        out[nm], raw = ML.errors(z, np.asarray(g, float), np.asarray(d, float))   # variants return seconds
    out['_raw'] = raw
    out['_unison'] = z.get('unison', None)
    return cid, out


def line(nm, res, ids_R, ids_S):
    s = f'{nm:30s}'
    if ids_R:
        bad = ('R_ND_tenor_13', 'R_ND_tenor_23')     # CSD_ND_tenor_3.f0 is a copy of tenor_2's annotation (method_truth.py)
        for lab, sel in (('R', ids_R), ('Rev', ids_R[0::2]), ('Rodd', ids_R[1::2]), ('Rc', [c for c in ids_R if c not in bad])):
            q = ML.summ([res[c][nm] for c in sel])
            s += f' | {lab} {q["mean"]:5.1f}/{q["median"]:4.1f}/{q["w20"]:4.1f}%'
        worse = sum(res[c][nm].mean() > res[c]['_raw'].mean() + 2 for c in ids_R)
        s += f' w{worse:d}'
    if ids_S:
        q = ML.summ([res[c][nm] for c in ids_S]); o = ML.summ([res[c][nm] for c in ids_S if c.endswith('_o')])
        s += f' | S {q["mean"]:4.2f} p90 {q["p90"]:4.1f} oct {o["mean"]:4.2f} [t/s/h ' + '/'.join(
            f'{ML.summ([res[c][nm] for c in ids_S if c.endswith("_" + v)])["mean"]:.2f}' for v in 'tsh') + ']'
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('tag'); ap.add_argument('variants')
    ap.add_argument('--suite', default='RS'); ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    names = a.variants.split(',')
    ids_R = ML.manifest('R') if 'R' in a.suite else []
    ids_S = ML.manifest('S') if 'S' in a.suite else []
    ids = ids_R + ids_S
    # big R cases first for load balance
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(work, [(c, names) for c in ids]))
    if ids_R:
        q = ML.summ([res[c]['_raw'] for c in ids_R]); q2 = ML.summ([res[c]['_raw'] for c in ids_R if c not in ('R_ND_tenor_13', 'R_ND_tenor_23')])
        print(f'{"raw":30s} | R {q["mean"]:5.1f}/{q["median"]:4.1f}/{q["w20"]:4.1f}% | Rc {q2["mean"]:5.1f}/{q2["median"]:4.1f}/{q2["w20"]:4.1f}%')
    for nm in names: print(line(nm, res, ids_R, ids_S), flush=True)
    np.savez(f'{rabpaths.RESULTS}/method_{a.tag}.npz',
             **{f'{c}|{k}': v for c, r in res.items() for k, v in r.items() if isinstance(v, np.ndarray)})


if __name__ == '__main__':
    main()
