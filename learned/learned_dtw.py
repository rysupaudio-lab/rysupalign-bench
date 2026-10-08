#!/opt/homebrew/bin/python3.13
"""RAB runner (training-side, PyTorch checkpoint): learned alignment embeddings (AlignHead over frozen HuBERT-base) + the same dense
band DTW and path read-out as ../fdtw.py.

usage (as a rab_score RUNNER): learned_dtw.py model guide.wav double.wav
env CKPT     checkpoint name in training/ckpt (default 'v1') or a path; 'a;b' (semicolon list) concatenates
             the models' blocks (each 1/sqrt(n) weight, together = one learned block)
    FEAT     'learned' (default) or 'learned+mel' (adds fdtw's unit-normalised log-mel block)
    MELW     weight of the mel block relative to the learned block (default 1.0)
    CENTER   1 = subtract the take's mean embedding before normalising (as fdtw does for SSL feats)
    F0MAX    gate f0 ceiling in Hz (default 800 = v6; v7 runner uses 1600, see f0())
    TX       1 (with GATE=1, v7) = non-unison doubles are transposed to the guide's pitch for analysis
             (proto_v5 resample) and aligned with the full FEAT instead of the learned block alone
    GATE     1 = proto_v5 interval gate: '+mel' blocks are used only when the double is in unison
             with the guide (median f0 interval <= INT_ST semitones, default 0.5), else learned only
    HOP_MS (10), BAND_S (1.0)
Learned frames are placed on the 10 ms grid by their true centre times (HuBERT frame i at
20 i + 12.5 ms, 100 fps head frame k at 10 k + 12.5 ms).
"""
import hashlib, json, os, sys
import numpy as np, librosa
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'runners'))
import fdtw
import model as M

CACHE = f'{M.RAZER}/rab_cache'
os.makedirs(CACHE, exist_ok=True)


def save_atomic(f, a):
    tmp = f'{f}.{os.getpid()}.tmp.npy'
    np.save(tmp, a); os.replace(tmp, f)


def file_key(p):
    return hashlib.sha1((os.path.realpath(p) + str(os.path.getmtime(p))).encode()).hexdigest()[:16]


def hub_layers(path, x16, hub=None):
    """[nL, T, 768] float16 for the whole file (30 s chunks, 2 s overlap trimmed), cached."""
    f = f'{CACHE}/{file_key(path)}_hub{M.L_LO}-{M.L_HI}.npy'
    if os.path.exists(f) and hub is None: return np.load(f)
    import torch
    m = hub or M.load_hubert()
    y = (x16 - x16.mean()) / (x16.std() + 1e-7)
    chunk, hop = 16000 * 30, 16000 * 28; outs = []
    with torch.no_grad():
        for s0 in range(0, max(1, len(y)), hop):
            seg = torch.from_numpy(y[s0:s0 + chunk].astype(np.float32))[None].to(M.DEV)
            if seg.shape[1] < 8000: break
            h = M.hubert_layers(m, seg)[0].float().cpu().numpy()
            a = 0 if s0 == 0 else 50
            b = h.shape[1] if s0 + chunk >= len(y) else 50 * 29
            outs.append(h[:, a:b])
            if s0 + chunk >= len(y): break
    H = np.concatenate(outs, 1).astype(np.float16)
    if hub is None: save_atomic(f, H)
    return H


_head = {}


def load_ckpt(name):
    import torch
    if name in _head: return _head[name]
    p = name if os.path.exists(name) else f'{M.RAZER}/training/ckpt/{name}.pt'
    sd = torch.load(p, map_location='cpu', weights_only=False)
    head = M.AlignHead(**sd['cfg']); head.load_state_dict(sd['head']); head.eval().to(M.DEV)
    hub = None
    if sd.get('hub_layers'):
        hub = M.load_hubert()
        for i, s in sd['hub_layers'].items(): hub.encoder.layers[int(i)].load_state_dict(s)
    _head[name] = (head, hub, os.path.basename(p)[:-3])
    return _head[name]


def embed(path, x, ck, cache=True):
    import torch
    head, hub, tag = load_ckpt(ck)
    f = f'{CACHE}/{file_key(path)}_emb_{tag}.npy'
    if cache and os.path.exists(f): return np.load(f), (100.0 if head.mel else 50.0)
    x16 = librosa.resample(x, orig_sr=fdtw.SR, target_sr=16000).astype(np.float32)
    H = hub_layers(path, x16, hub if (hub is not None or cache) else M.load_hubert())
    with torch.no_grad():
        Ht = torch.from_numpy(H.astype(np.float32))[None].to(M.DEV)
        Mel = None
        if head.mel:
            Mel = M.mel_input(torch.from_numpy(x16)[None].to(M.DEV), head.cfg)
        S = M.skip_mel(torch.from_numpy(x16)[None].to(M.DEV), head.melskip) if head.melskip else None
        E = head(Ht, Mel, S)[0].cpu().numpy()
    if cache: save_atomic(f, E.astype(np.float32))
    return E, (100.0 if head.mel else 50.0)


def learned_block(path, x, hop, ck, center, cache=True):
    E, fps = embed(path, x, ck, cache)
    if center: E = E - E.mean(0, keepdims=True)
    n = 1 + len(x) // hop
    tgrid = np.arange(n) * hop / fdtw.SR
    src = np.clip((tgrid - 0.0125) * fps, 0, len(E) - 1)
    F = np.stack([np.interp(src, np.arange(len(E)), c) for c in E.T])
    return F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)


def features(path, x, hop, feat=None, cache=True):
    ck = os.environ.get('CKPT', 'v1'); feat = feat or os.environ.get('FEAT', 'learned')
    center = os.environ.get('CENTER', '0') == '1'
    blocks = [learned_block(path, x, hop, c, center, cache) for c in ck.split(';')]   # 'a;b' = ensemble
    wts = [1.0 / np.sqrt(len(blocks))] * len(blocks)
    for b in feat.split('+')[1:]:
        blocks.append(fdtw.block(b, path, x, hop)); wts.append(float(os.environ.get('MELW', '1.0')))
    n = min(b.shape[1] for b in blocks)
    F = np.concatenate([w * b[:, :n] for w, b in zip(wts, blocks)]) / np.sqrt(sum(w * w for w in wts))
    return np.ascontiguousarray(F.T.astype(np.float32))


def precache():
    """Fill the HuBERT layer cache for every RAB case file once (serial, one model load)."""
    rab = os.path.dirname(HERE); hub = M.load_hubert(); done = set()
    for c in json.load(open(f'{rab}/manifest.json')):
        for nm in ('guide', 'double'):
            p = f"{rab}/cases/{c['id']}/{nm}.wav"; f = f'{CACHE}/{file_key(p)}_hub{M.L_LO}-{M.L_HI}.npy'
            if f in done or os.path.exists(f): continue
            x = fdtw.load(p); x16 = librosa.resample(x, orig_sr=fdtw.SR, target_sr=16000).astype(np.float32)
            save_atomic(f, hub_layers(p, x16, hub)); done.add(f); print(c['id'], nm, flush=True)


def f0(x):
    """proto_v5.f0 with a configurable ceiling. F0MAX (default 800 Hz = v6 behaviour). v7 uses 1600: with an
    800 Hz ceiling a guide sung (or shifted) an octave up above ~800 Hz is tracked at its sub-octave, the gate
    reads the pair as unison and adds mel x2 -> octave failures (S_o DavidTyo_OhLife_22/23: 62 / 12 ms -> fixed)."""
    fmax = float(os.environ.get('F0MAX', '800'))
    y = librosa.resample(x, orig_sr=fdtw.SR, target_sr=8000)
    f = librosa.yin(y, fmin=70, fmax=fmax, sr=8000, frame_length=512, hop_length=80)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=80)[0][:len(f)]
    f[rms < max(1e-4, np.percentile(rms, 90) * 0.05)] = np.nan
    return f


def main():
    if sys.argv[1] == '--precache': return precache()
    _, guide, dbl = sys.argv[1:4]
    hop = int(round(float(os.environ.get('HOP_MS', '10')) / 1000 * fdtw.SR))
    band = int(float(os.environ.get('BAND_S', '1.0')) * fdtw.SR / hop)
    xg, xd = fdtw.load(guide), fdtw.load(dbl)
    cand = os.environ.get('FEAT', 'learned'); interval = None
    if os.environ.get('GATE', '0') == '1' and '+' in cand:
        # proto_v5-style interval gate: align on the learned block alone, read guide/double f0 at
        # corresponding frames; spectral (mel) blocks are only added for unison (|interval| <= INT_ST)
        A, B = features(guide, xg, hop, 'learned'), features(dbl, xd, hop, 'learned')
        p = np.array(fdtw.band_dtw(A, B, band)); cnt = np.bincount(p[:, 0]).astype(float)
        dm = np.bincount(p[:, 0], p[:, 1].astype(float)) / np.maximum(cnt, 1)
        fg, fd = f0(xg), f0(xd)
        di = np.clip(np.round(dm).astype(int), 0, len(fd) - 1); gg = np.clip(np.arange(len(dm)), 0, len(fg) - 1)
        ok = np.isfinite(fg[gg]) & np.isfinite(fd[di])
        interval = float(np.median(12 * np.log2(fd[di][ok] / fg[gg][ok]))) if ok.sum() > 50 else 0.0
        if abs(interval) > float(os.environ.get('INT_ST', '0.5')):
            cand = 'learned'
            if os.environ.get('TX', '0') == '1':
                # v7 option: analysis-only transposition (proto_v5): resample the double to the guide's
                # pitch, embed it there with the full FEAT (learned + mel), read frames back by the time scale
                import soundfile as sf, tempfile
                y = librosa.resample(xd, orig_sr=fdtw.SR, target_sr=int(round(fdtw.SR * 2 ** (interval / 12))))
                tp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name; sf.write(tp, y, fdtw.SR)
                try:
                    Bt = features(tp, y, hop, os.environ.get('FEAT'), cache=False)
                finally:
                    os.remove(tp)
                nB = 1 + len(xd) // hop; src = np.clip(np.arange(nB) * len(y) / len(xd), 0, len(Bt) - 1)
                Bm = np.stack([np.interp(src, np.arange(len(Bt)), c) for c in Bt.T], 1)
                B = np.ascontiguousarray(Bm / (np.linalg.norm(Bm, axis=1, keepdims=True) + 1e-9), np.float32)
                A = features(guide, xg, hop, os.environ.get('FEAT')); cand = 'transposed'
    if cand != 'transposed' and (interval is None or cand != 'learned' or os.environ.get('GATE', '0') != '1'):
        A, B = features(guide, xg, hop, cand), features(dbl, xd, hop, cand)
    path = fdtw.band_dtw(A, B, band)
    fs = fdtw.SR / hop
    g = np.array([p[0] for p in path]); d = np.array([p[1] for p in path], float)
    gu, inv = np.unique(g, return_inverse=True)
    dm = np.bincount(inv, d) / np.bincount(inv)
    print(json.dumps({'ok': True, 'moved': True, 'cand': cand, 'interval': interval, 'warp': [[float(a / fs), float(b / fs)] for a, b in zip(gu, dm)]}))


if __name__ == '__main__':
    main()
