"""Learned alignment embeddings over a frozen HuBERT-base (facebook/hubert-base-ls960, Apache-2.0).

AlignHead: per-layer LayerNorm of HuBERT hidden states L_LO..L_HI, softmax-weighted sum,
a small residual dilated-conv stack, linear to DIM, L2-normalised. 50 fps (HuBERT rate),
optionally x2 upsampled to 100 fps with a log-mel branch (cfg['mel']).
"""
import os, warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAZER = os.environ.get('RAB_WORK', 'work')   # working dir for caches, training data and checkpoints (env RAB_WORK)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
L_LO, L_HI = 3, 9


def load_hubert(n_layers=L_HI):
    from transformers import HubertModel
    m = HubertModel.from_pretrained('facebook/hubert-base-ls960')
    m.encoder.layers = m.encoder.layers[:n_layers]            # drop unused top layers
    m.config.num_hidden_layers = n_layers
    return m.eval().to(DEV)


def hubert_layers(m, wav, grad=False):
    """wav [b, S] float (already normalised) -> [b, nL, T, 768] for layers L_LO..L_HI."""
    with torch.set_grad_enabled(grad):
        hs = m(wav, output_hidden_states=True).hidden_states
    return torch.stack(hs[L_LO:L_HI + 1], 1)


class ResConv(nn.Module):
    def __init__(self, c, k, d):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=d * (k - 1) // 2, dilation=d)
        self.norm = nn.GroupNorm(1, c)

    def forward(self, x):
        return x + self.conv(F.gelu(self.norm(x)))


class AlignHead(nn.Module):
    def __init__(self, n_layers=L_HI - L_LO + 1, width=256, dim=128, blocks=((5, 1), (5, 2), (5, 4)), mel=0, melhi=0, melskip=0):
        super().__init__()
        self.cfg = dict(n_layers=n_layers, width=width, dim=dim, blocks=blocks, mel=mel, melhi=melhi, melskip=melskip)
        self.melhi = melhi; self.melskip = melskip
        if melskip: self.skip_w = nn.Parameter(torch.tensor(0.55))   # softplus -> ~1.0: equal weight
        self.lw = nn.Parameter(torch.zeros(n_layers))
        self.inp = nn.Linear(768, width)
        self.mel = mel
        if mel:                                              # 100 fps branch: log-mel bins -> width
            self.up = nn.ConvTranspose1d(width, width, 4, stride=2, padding=1)
            self.mel_in = nn.Conv1d(mel + melhi, width, 5, padding=2)
        self.blocks = nn.Sequential(*[ResConv(width, k, d) for k, d in blocks])
        self.out = nn.Linear(width, dim)

    def forward(self, H, M=None, S=None):
        """H [b, nL, T, 768]; M [b, mel, 2T] log-mel (100 fps) when cfg mel>0 -> [b, T', dim] unit."""
        H = F.layer_norm(H, H.shape[-1:])
        w = torch.softmax(self.lw, 0)
        x = (H * w[None, :, None, None]).sum(1)
        x = self.inp(x).transpose(1, 2)                      # [b, width, T]
        if self.mel:
            x = self.up(x)
            T2 = min(x.shape[-1], M.shape[-1])
            x = x[..., :T2] + self.mel_in(M[..., :T2])
        x = self.blocks(x).transpose(1, 2)
        e = F.normalize(self.out(x), dim=-1)
        if self.melskip:                                     # fixed spectral-shape block, learned weight
            S = S.transpose(1, 2); T2 = min(e.shape[1], S.shape[1])
            a = F.softplus(self.skip_w)
            e = torch.cat([e[:, :T2], a * S[:, :T2]], -1) / torch.sqrt(1 + a * a)
        return e


_melfb = {}


def mel_input(wav16, cfg):
    """Spectral input of the 100 fps branch: 25 ms log-mel (cfg mel bins, 60-8000 Hz) plus, when
    cfg melhi > 0, a 64 ms high-resolution log-mel over 80-2000 Hz (resolves sung harmonics/pitch).
    Both on the same 10 ms grid, frame k centred 10 k + 12.5 ms."""
    M = logmel(wav16, cfg['mel'])
    if cfg.get('melhi'):
        pad = (1024 - 400) // 2
        H = logmel(torch.nn.functional.pad(wav16, (pad, pad)), cfg['melhi'], n_fft=1024, win=1024, fmin=80, fmax=2000)
        T = min(M.shape[-1], H.shape[-1]); M = torch.cat([M[..., :T], H[..., :T]], 1)
    return M


def skip_mel(wav16, n_mels=80):
    """fdtw-style spectral-shape block on the 10 ms grid: 46 ms log-mel (40-8000 Hz), each frame
    centred across bins and unit-normalised. [b, n_mels, frames]"""
    pad = (736 - 400) // 2
    M = logmel(torch.nn.functional.pad(wav16, (pad, pad)), n_mels, n_fft=1024, win=736, fmin=40, fmax=8000, center_time=False)
    M = M - M.mean(1, keepdim=True)
    return M / (M.norm(dim=1, keepdim=True) + 1e-9)


def logmel(wav16, n_mels=64, n_fft=512, win=400, fmin=60, fmax=8000, center_time=True):
    """[b, S] 16 kHz -> [b, n_mels, frames] at 10 ms hop (frame k covers samples 160k..160k+400,
    so frames 2i/2i+1 straddle HuBERT frame i), per-utterance mean removed."""
    import librosa
    key = (wav16.device.type, n_mels, n_fft, win, fmin, fmax)
    if key not in _melfb:
        fb = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        _melfb[key] = (torch.from_numpy(fb).float().to(wav16.device), torch.hann_window(win).to(wav16.device))
    fb, w = _melfb[key]
    fr = wav16.unfold(-1, win, 160) * w                     # [b, frames, win]
    P = torch.fft.rfft(fr, n=n_fft).abs() ** 2
    M = torch.log(P @ fb.T + 1e-5).transpose(1, 2)
    return M - M.mean(-1, keepdim=True) if center_time else M


def norm_wav(w):
    return (w - w.mean(-1, keepdim=True)) / (w.std(-1, keepdim=True) + 1e-7)
