"""Export one ONNX graph: normalised 16 kHz wav [1,S] -> learned embeddings [1,T,128] at 100 fps
(frame k centred at 10k + 12.5 ms). HuBERT-base layers 1-9 (frozen) + AlignHead v6 + its log-mel
front end, the STFT expressed as fixed conv1d kernels so the graph is plain ONNX.

usage: python learned/export_learned.py CKPT.pt OUT.onnx"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, librosa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
CKPT, OUT = sys.argv[1], sys.argv[2]
M.DEV = "cpu"

class ConvMel(nn.Module):
    """|rfft(window * frame)|^2 @ melfb via conv1d; frames start every 160 samples, as model.logmel."""
    def __init__(s, n_mels, n_fft, win, fmin, fmax):
        super().__init__()
        k = np.arange(n_fft // 2 + 1)[:, None]; t = np.arange(win)[None, :]
        w = np.hanning(win + 1)[:-1] if False else torch.hann_window(win).numpy()
        re = np.cos(2 * np.pi * k * t / n_fft) * w; im = -np.sin(2 * np.pi * k * t / n_fft) * w
        s.register_buffer("kre", torch.tensor(re[:, None, :], dtype=torch.float32))
        s.register_buffer("kim", torch.tensor(im[:, None, :], dtype=torch.float32))
        fb = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        s.register_buffer("fb", torch.tensor(fb, dtype=torch.float32))
    def forward(s, x):            # x [1, S] -> [1, n_mels, frames], per-utterance mean removed
        xx = x[:, None, :]
        P = F.conv1d(xx, s.kre, stride=160) ** 2 + F.conv1d(xx, s.kim, stride=160) ** 2
        Mm = torch.log(torch.einsum("mk,bkt->bmt", s.fb, P) + 1e-5)
        return Mm - Mm.mean(-1, keepdim=True)

class Full(nn.Module):
    def __init__(s, hub, head):
        super().__init__(); s.hub, s.head = hub, head
        cfg = head.cfg
        s.m1 = ConvMel(cfg["mel"], 512, 400, 60, 8000)
        s.m2 = ConvMel(cfg["melhi"], 1024, 1024, 80, 2000) if cfg.get("melhi") else None
    def forward(s, wav):
        hs = s.hub(wav, output_hidden_states=True).hidden_states
        H = torch.stack(hs[M.L_LO:M.L_HI + 1], 1)
        Mel = s.m1(wav)
        if s.m2 is not None:
            pad = (1024 - 400) // 2
            Hm = s.m2(F.pad(wav, (pad, pad)))
            T = min(Mel.shape[-1], Hm.shape[-1]); Mel = torch.cat([Mel[..., :T], Hm[..., :T]], 1)
        h = s.head                                           # head.forward with static shapes for export
        H = F.layer_norm(H, (768,))
        w = torch.softmax(h.lw, 0)
        x = (H * w[None, :, None, None]).sum(1)
        x = h.inp(x).transpose(1, 2)
        x = h.up(x)
        T2 = torch.minimum(torch.tensor(x.shape[-1]), torch.tensor(Mel.shape[-1]))
        x = x[..., :T2] + h.mel_in(Mel[..., :T2])
        x = h.blocks(x).transpose(1, 2)
        return F.normalize(h.out(x), dim=-1)

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
head = M.AlignHead(**sd["cfg"]); head.load_state_dict(sd["head"]); head.eval()
hub = M.load_hubert().cpu().eval()
full = Full(hub, head).eval()
x = torch.randn(1, 16000 * 7)
x = (x - x.mean()) / x.std()
with torch.no_grad():
    ref = head(M.hubert_layers(hub, x), M.mel_input(x, head.cfg))[0].numpy()
    out = full(x)[0].numpy()
print("torch vs reference max diff", float(np.abs(out - ref).max()), out.shape, ref.shape)
torch.onnx.export(full, (x,), OUT, input_names=["wav"], output_names=["emb"],
                  dynamic_axes={"wav": {1: "n"}, "emb": {1: "t"}}, opset_version=17, dynamo=False)
import onnxruntime as ort
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
for sec in (7, 13.37):
    xx = torch.randn(1, int(16000 * sec)); xx = (xx - xx.mean()) / xx.std()
    with torch.no_grad(): r = full(xx)[0].numpy()
    o = sess.run(None, {"wav": xx.numpy()})[0][0]
    print(sec, "s onnx vs torch", float(np.abs(o - r).max()), o.shape)
print("MB", os.path.getsize(OUT) / 1e6)
