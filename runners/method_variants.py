"""Method variants for method_run.py. Each takes a case dict z (method_lib.load) and returns
(guide_times_s, double_times_s) of the warp. Frame i of the 10 ms grid is at i / 100 s."""
import numpy as np
import method_lib as ML

FS = ML.FS
VARIANTS = {}


def variant(fn):
    VARIANTS[fn.__name__] = fn; return fn


def run(C, z, wd=2.0, wh=1.0, wv=1.0):
    pi, pj = ML.dtw_band(C, z['Bl'].shape[0], wd, wh, wv)
    g, d = ML.readout(pi, pj)
    return g / FS, d / FS


@variant
def base(z):            # learned v6 + mel x2, interval gate, band 1 s, diag 2
    return run(ML.baseline_cost(z), z)


@variant
def learned_only(z):
    ML.gate(z); return run(z['_Cl'], z)


@variant
def mel_only(z):
    C = ML.band_cost(np.ascontiguousarray(z['Am']), np.ascontiguousarray(z['Bm']), 100); return run(C, z)


@variant
def base_py(z):         # identical features + gate, fdtw.band_dtw's recursion (the documented Python numbers)
    C = ML.baseline_cost(z)
    pi, pj = ML.dtw_band_py(C, z['Bl'].shape[0], 2.0)
    g, d = ML.readout(pi, pj); return g / FS, d / FS


# ------------------------------------------------------------------ configurable pipeline: name 'k=v/k=v/...'
DEFAULT = dict(rec='py', wd=2.0, wh=1.0, melw=2.0, band=100, pw=0.0, dw=0.0, pcap=1.0, vm=0.5, dcap=1.0,
               sm=0, rb=0, rpw=None, rdw=None, rmelw=None, rwd=None, rrec=None, lw=1.0)


def semis(f):
    return np.where(np.isfinite(f) & (f > 0), 69 + 12 * np.log2(np.maximum(f, 1e-9) / 440.0), np.nan).astype(np.float64)


DCLIP = [0.0]; DC = [0.0]; DUNI = [0]; DK = [3]


def delta(s, k=3):
    """pitch slope (st per 10 ms) by +-k frame difference; nan where unvoiced. DCLIP>0 clips per-frame
    steps first (octave errors of the tracker become small steps)."""
    if DCLIP[0] > 0:
        st = np.diff(s, prepend=np.nan); st = np.clip(st, -DCLIP[0], DCLIP[0])
        c = np.nancumsum(np.nan_to_num(st)); c[np.isnan(s)] = np.nan; s = c
    d = np.full_like(s, np.nan)
    d[k:-k] = (s[2 * k:] - s[:-2 * k]) / (2 * k)
    return d


def cmn(F):
    F = F - F.mean(0, keepdims=True); return np.ascontiguousarray(ML.unit(F), np.float32)


OPT = {}


def cost(z, cen, band, melw, pw, dw, pcap, vm, dcap, lw=1.0):
    ML.gate(z)
    for nm, sides in (('mcmn', ('Am', 'Bm')), ('lcmn', ('Al', 'Bl'))):     # never mutate the case dict
        for sd in sides:
            if '_orig' + sd not in z: z['_orig' + sd] = z[sd]
            if OPT.get(nm):
                if '_cmn' + sd not in z: z['_cmn' + sd] = cmn(z['_orig' + sd])
                z[sd] = z['_cmn' + sd]
            else: z[sd] = z['_orig' + sd]
    n, m = z['Al'].shape[0], z['Bl'].shape[0]
    key = ('C', melw if z['unison'] else 0.0, band, cen.tobytes().__hash__(), lw, bool(OPT.get('mcmn')), bool(OPT.get('lcmn')))
    if key not in z:
        if z['unison'] and melw > 0:
            if lw == 1.0: A, B = ML.combo(z, melw, 'A'), ML.combo(z, melw, 'B')
            else:
                A = np.ascontiguousarray(np.concatenate([lw * z['Al'], melw * z['Am']], 1) / np.sqrt(lw * lw + melw * melw), np.float32)
                B = np.ascontiguousarray(np.concatenate([lw * z['Bl'], melw * z['Bm']], 1) / np.sqrt(lw * lw + melw * melw), np.float32)
        else: A, B = z['Al'], z['Bl']
        z[key] = ML.corr_cost(np.ascontiguousarray(A), np.ascontiguousarray(B), cen, band)
    C = z[key]
    if pw or dw or OPT.get('sil') is not None or OPT.get('ew'):
        fg, fd = (z['hg'], z['hd']) if 'hg' in z else (z['fg'], z['fd'])
        sg, sd = semis(fg)[:n], semis(fd)[:m]
        sg = np.concatenate([sg, np.full(max(0, n - len(sg)), np.nan)]); sd = np.concatenate([sd, np.full(max(0, m - len(sd)), np.nan)])
        iv = z['interval']
        if OPT.get('pmed'):
            from scipy.ndimage import median_filter
            def mf(x):
                y = median_filter(np.nan_to_num(x, nan=-1e3), size=int(OPT['pmed']), mode='nearest')
                y = np.where(y < -100, np.nan, y); y[np.isnan(x)] = np.nan; return y
            sg, sd = mf(sg), mf(sd)
        if pw and (z['unison'] or not OPT.get('puni')): C = C + pw * ML.corr_pitch(sg, sd, cen, band, iv, pcap, vm)
        if OPT.get('sil') is not None:
            # both frames inactive (low rms) -> fixed cost: silence carries no timing, the path is free there
            def act(r, nn):
                a = r >= max(1e-4, np.percentile(r, 90) * OPT.get('athr', 0.05))
                a = np.concatenate([a, np.zeros(max(0, nn - len(a)), bool)])[:nn]; return a
            ag, ad = act(z['rg'], n), act(z['rd'], m)
            jj = np.clip(cen[:, None] + np.arange(C.shape[1])[None, :] - (C.shape[1] - 1) // 2, 0, m - 1)
            both = (~ag[:, None]) & (~ad[jj])
            C = np.where(both & np.isfinite(C), OPT['sil'], C).astype(np.float32)
        if OPT.get('ew'):
            # loudness-slope cost: d/dt log-rms (dB per 10 ms over +-ek frames), unison only like dw
            def ld(r, nn):
                L = 20 * np.log10(np.maximum(r, 1e-5)); L = np.concatenate([L, np.full(max(0, nn - len(L)), L[-1])])[:nn]
                k = int(OPT.get('ek', 2)); d = np.zeros_like(L); d[k:-k] = (L[2 * k:] - L[:-2 * k]) / (2 * k); return d
            if z['unison'] or not DUNI[0]:
                C = C + OPT['ew'] * ML.corr_absdiff(ld(z['rg'], n), ld(z['rd'], m), cen, band, OPT.get('ecap', 3.0))
        if dw and (z['unison'] or not DUNI[0]):
            DCLIP[0] = DC[0]
            C = C + dw * ML.corr_absdiff(delta(sg, DK[0]), delta(sd, DK[0]), cen, band, dcap)
    return C


def medfilt(x, w):
    if w <= 1: return x
    from scipy.ndimage import median_filter
    return median_filter(x, size=w, mode='nearest')


def pipeline(z, cfg):
    c = dict(DEFAULT); c.update(cfg)
    OPT.clear(); OPT.update({k: c[k] for k in ('mcmn', 'lcmn', 'sil', 'athr', 'ew', 'ek', 'ecap', 'pmed', 'puni') if k in c})
    DC[0] = c.get('dclip', 0.0); DUNI[0] = int(c.get('duni', 0)); DK[0] = int(c.get('dk', 3))
    n, m = z['Al'].shape[0], z['Bl'].shape[0]
    band = int(c['band'])
    cen = np.arange(n, dtype=np.int64)
    C = cost(z, cen, band, c['melw'], c['pw'], c['dw'], c['pcap'], c['vm'], c['dcap'], c['lw'])
    pi, pj = ML.dtw_corr(C, cen, m, c['wd'], c['wh'], c['wh'], 1 if c['rec'] == 'py' else 0)
    g, dm = ML.readout(pi, pj)
    if c['rb']:
        # refinement: same DTW restricted to the first path +- rb frames (per-row [min j - rb, max j + rb])
        rb = int(c['rb'])
        jmin = np.full(n, m, np.int64); jmax = np.full(n, -1, np.int64)
        np.minimum.at(jmin, pi, pj); np.maximum.at(jmax, pi, pj)
        lo = jmin - rb; hi = jmax + rb
        C2 = cost(z, cen, band, c['rmelw'] if c['rmelw'] is not None else c['melw'],
                  c['rpw'] if c['rpw'] is not None else c['pw'], c['rdw'] if c['rdw'] is not None else c['dw'],
                  c['pcap'], c['vm'], c['dcap'], c['lw']).copy()
        jj = cen[:, None] + np.arange(2 * band + 1)[None, :] - band
        C2[(jj < lo[:, None]) | (jj > hi[:, None])] = np.inf
        rrec = c['rrec'] or c['rec']
        pi, pj = ML.dtw_corr(C2, cen, m, c['rwd'] or c['wd'], c['wh'], c['wh'], 1 if rrec == 'py' else 0)
        g, dm = ML.readout(pi, pj)
    if c['sm']:
        dm = g + medfilt(dm - g, int(c['sm']))
    return g / FS, dm / FS


class _Reg(dict):
    def __missing__(self, name):
        cfg = {}
        for kv in name.split('/'):
            k, v = kv.split('=')
            cfg[k] = v if k in ('rec', 'rrec') else float(v)
        return lambda z: pipeline(z, cfg)


VARIANTS = _Reg(VARIANTS)
