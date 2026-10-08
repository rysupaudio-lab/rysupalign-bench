"""Learned alignment embeddings from the released ONNX model (rysupalign_learned_v6[.int8].onnx).

Model contract: input 'wav' [1, S] float32, 16 kHz mono, normalised to zero mean / unit variance over
the whole file; output 'emb' [1, T, 128] unit-norm embeddings at 100 fps, frame k centred at
10 k + 12.5 ms. Long files are processed in 30 s chunks with 2 s overlap (1 s trimmed on each side of a
join), like the plug-in.

Model path: env RYSUPALIGN_ONNX (default <repo>/models/rysupalign_learned_v6.onnx). Download it from the
GitHub release (see README). Embeddings are cached in $RAB_WORK/emb_cache.
"""
import hashlib, os
import numpy as np, librosa
import rabpaths

MODEL = os.environ.get('RYSUPALIGN_ONNX', os.path.join(rabpaths.REPO, 'models', 'rysupalign_learned_v6.onnx'))
CACHE = os.path.join(rabpaths.WORK, 'emb_cache')
_sess = {}


def _session(path):
    if path not in _sess:
        import onnxruntime as ort
        if not os.path.exists(path):
            raise SystemExit(f'model not found: {path} (set RYSUPALIGN_ONNX or download it from the release)')
        so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get('RAB_ORT_THREADS', '4'))
        _sess[path] = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    return _sess[path]


def embed16(y, model=MODEL):
    """y: 16 kHz mono float -> [T, 128] at 100 fps."""
    y = ((y - y.mean()) / (y.std() + 1e-7)).astype(np.float32)
    s = _session(model)
    chunk, hop = 16000 * 30, 16000 * 28
    outs = []
    for s0 in range(0, max(1, len(y)), hop):
        seg = y[s0:s0 + chunk]
        if len(seg) < 8000: break
        e = s.run(None, {'wav': seg[None]})[0][0]
        a = 0 if s0 == 0 else 100
        b = e.shape[0] if s0 + chunk >= len(y) else 100 * 29
        outs.append(e[a:b])
        if s0 + chunk >= len(y): break
    return np.concatenate(outs).astype(np.float32)


def embed(path, x, sr, model=MODEL):
    """Cached embeddings for an audio file already loaded as x at rate sr."""
    os.makedirs(CACHE, exist_ok=True)
    k = hashlib.sha1((os.path.realpath(path) + str(os.path.getmtime(path)) + os.path.basename(model)).encode()).hexdigest()[:16]
    f = os.path.join(CACHE, f'{k}.npy')
    if os.path.exists(f): return np.load(f)
    E = embed16(librosa.resample(x, orig_sr=sr, target_sr=16000), model)
    tmp = f'{f}.{os.getpid()}.tmp.npy'; np.save(tmp, E); os.replace(tmp, f)
    return E


def learned_block(path, x, sr, hop):
    """[128, n] unit-norm block on the hop grid of x (frame i at i * hop / sr s), as fdtw.block returns."""
    E = embed(path, x, sr)
    n = 1 + len(x) // hop
    src = np.clip((np.arange(n) * hop / sr - 0.0125) * 100.0, 0, len(E) - 1)
    F = np.stack([np.interp(src, np.arange(len(E)), c) for c in E.T])
    return F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)
