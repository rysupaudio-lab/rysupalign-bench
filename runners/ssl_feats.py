"""Self-supervised / frontier model embeddings for RAB feature-DTW (research only).

name@layer: hubert@L (facebook/hubert-base-ls960, Apache-2.0), wavlm@L (microsoft/wavlm-base-plus),
            mert@L (m-a-p/MERT-v1-95M, CC BY-NC 4.0), mms@L (MMS-300m forced aligner, CC BY-NC 4.0), and the
            large models below. Check each model's own licence before using it for anything but research.
Weights are fetched from the Hugging Face Hub on first use. Embeddings are cached as float16 in
$RAB_WORK/ssl_cache. Returns frames x dims at the model's native rate (~50 fps) plus that rate.
"""
import hashlib, os
import numpy as np, librosa, torch

import rabpaths
CACHE = os.path.join(rabpaths.WORK, 'ssl_cache'); os.makedirs(CACHE, exist_ok=True)
REPOS = {'hubert': ('facebook/hubert-base-ls960', 16000), 'wavlm': ('microsoft/wavlm-base-plus', 16000),
         'mert': ('m-a-p/MERT-v1-95M', 24000), 'mms': ('MahmoudAshraf/mms-300m-1130-forced-aligner', 16000),
         # large models (2026-10-07 frontier check)
         'mertL': ('m-a-p/MERT-v1-330M', 24000), 'wavlmL': ('microsoft/wavlm-large', 16000),
         'hubertL': ('facebook/hubert-large-ll60k', 16000), 'xlsr': ('facebook/wav2vec2-xls-r-300m', 16000),
         'mms1b': ('facebook/mms-1b', 16000), 'whisper': ('openai/whisper-large-v3', 16000)}
_models = {}
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'


def _model(name):
    if name not in _models:
        from transformers import AutoModel
        repo, _ = REPOS[name]
        if name == 'whisper':
            from transformers import WhisperModel, WhisperFeatureExtractor
            m = WhisperModel.from_pretrained(repo).encoder
            m.fe = WhisperFeatureExtractor.from_pretrained(repo)
            _models[name] = m.eval().to(DEV); return _models[name]
        m = AutoModel.from_pretrained(repo, trust_remote_code=name.startswith('mert'), output_hidden_states=True)
        m.config.output_hidden_states = True
        _models[name] = m.eval().to(DEV)
    return _models[name]


def embed(path, x, sr, name, layer):
    key = hashlib.sha1((os.path.realpath(path) + str(os.path.getmtime(path))).encode()).hexdigest()[:16]
    f = f'{CACHE}/{key}_{name}_{layer}.npy'
    if os.path.exists(f): return np.load(f).astype(np.float32), 50.0
    repo, rate = REPOS[name]
    y = librosa.resample(x, orig_sr=sr, target_sr=rate).astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-7)
    m = _model(name)
    outs = []
    chunk = rate * 30; hop = rate * 28                  # 30 s chunks, 2 s overlap (trimmed 1 s each side)
    with torch.no_grad():
        for s0 in range(0, max(1, len(y)), hop):
            seg = torch.from_numpy(y[s0:s0 + chunk])[None].to(DEV)
            if seg.shape[1] < rate // 2: break
            if name == 'whisper':   # 30 s windows, 50 fps encoder frames; keep the frames covering this chunk
                feats = m.fe(seg[0].cpu().numpy(), sampling_rate=16000, return_tensors='pt').input_features.to(DEV)
                hs = m(feats, output_hidden_states=True).hidden_states[layer][0].float().cpu().numpy()
                h = hs[:int(round(seg.shape[1] / 16000 * 50))]
            elif name.startswith('mert'):   # its remote code drops hidden_states on current transformers: hook layer output
                got = {}
                hk = m.encoder.layers[layer - 1].register_forward_hook(lambda mod, i, out: got.__setitem__('h', out[0] if isinstance(out, tuple) else out))
                m(seg); hk.remove(); h = got['h'][0].float().cpu().numpy()
            else:
                h = m(seg, output_hidden_states=True).hidden_states[layer][0].float().cpu().numpy()
            fps = h.shape[0] / (seg.shape[1] / rate)
            a = 0 if s0 == 0 else int(fps * 1.0)
            b = h.shape[0] if s0 + chunk >= len(y) else int(fps * 29.0)
            outs.append(h[a:b])
            if s0 + chunk >= len(y): break
    E = np.concatenate(outs)
    np.save(f, E.astype(np.float16))
    return E, 50.0
