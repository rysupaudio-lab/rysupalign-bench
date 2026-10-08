#!/opt/homebrew/bin/python3.13
"""Train AlignHead on self-supervised pairs (gen_pairs.py) with a frame-level InfoNCE loss.

For every audible A frame the true (fractional) B frame is the positive (soft label split
between its two neighbouring frames); every other B frame within +-1 s is a negative. The
loss is applied in both directions (A->B, B->A). HuBERT is frozen (optionally the top
FT layers are fine-tuned with a small learning rate).

usage: train.py NAME [--pairs tr1[,ps1]] [--mix 0.7,0.3] [--init CKPT] [--val val1] [--steps 6000] [--bs 16] [--mel 0|64]
       [--width 256] [--dim 128] [--ft 0] [--tau 0.07] [--lr 2e-3] [--unison 3 | 3,3,1.5 (per set)] [--snap 1]
checkpoint -> training/ckpt/NAME.pt (head state, cfg, fine-tuned HuBERT layers if any)
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M

TR = f'{M.RAZER}/training'


def load_pairs(name):
    A = np.load(f'{TR}/pairs/{name}_A.npy', mmap_mode='r'); B = np.load(f'{TR}/pairs/{name}_B.npy', mmap_mode='r')
    meta = np.load(f'{TR}/pairs/{name}_meta.npz')
    sig = float(meta['sigma']) if 'sigma' in meta else 0.0      # >0: pseudo-labelled set, soft Gaussian target
    P = meta['pitch']; uni = np.abs(P[:, 1] - P[:, 0]) < 0.7     # unison (+-detune) pairs
    return A, B, meta['jf'], meta['valid'], sig, uni


def inverse(jf, valid):
    """B frame j -> fractional A frame, valid where inside the mapped range of valid A frames."""
    T = len(jf); i = np.arange(T, dtype=np.float32)
    jj = np.maximum.accumulate(jf)
    it = np.interp(i, jj, i)
    vi = np.zeros(T, bool)
    near = np.clip(np.round(it).astype(int), 0, T - 1)
    vi = valid[near] & (i >= jj[valid].min() if valid.any() else False) & (i <= jj[valid].max() if valid.any() else False)
    return it.astype(np.float32), vi


def nce(EA, EB, tgt, val, scale, W, sigma=0.0):
    """EA,EB [b,T,D]; tgt [b,T] fractional index into B frames; val [b,T] bool.
    sigma 0: exact truth, target split linearly between the two neighbouring frames;
    sigma > 0 (frames): Gaussian soft target for pseudo labels."""
    b, T, _ = EA.shape
    S = torch.einsum('bid,bjd->bij', EA, EB) * scale
    j = torch.arange(EB.shape[1], device=EA.device)[None, None, :].float()
    t = tgt[:, :, None]
    S = S.masked_fill((j - t).abs() > W, -1e4)
    if sigma > 0:
        Y = torch.exp(-0.5 * ((j - t) / sigma) ** 2); Y = Y / Y.sum(-1, keepdim=True).clamp(min=1e-6)
    else:
        lo = torch.floor(t); fr = t - lo
        Y = (j == lo).float() * (1 - fr) + (j == lo + 1).float() * fr
    lp = torch.log_softmax(S, -1)
    l = -(Y * lp).sum(-1)
    v = val.float()
    with torch.no_grad():
        hit = ((lp.argmax(-1).float() - tgt).abs() <= 1.0).float()
    return (l * v).sum() / v.sum().clamp(min=1), (hit * v).sum() / v.sum().clamp(min=1)


class Trainer:
    def __init__(self, a):
        self.a = a
        self.hub = M.load_hubert()
        for p in self.hub.parameters(): p.requires_grad = False
        self.ft = a.ft
        if a.ft:
            for L in self.hub.encoder.layers[M.L_HI - a.ft:M.L_HI]:
                for p in L.parameters(): p.requires_grad = True
        self.head = M.AlignHead(width=a.width, dim=a.dim, mel=a.mel, melhi=a.melhi, melskip=a.melskip).to(M.DEV)
        self.logscale = torch.nn.Parameter(torch.tensor(math.log(1 / a.tau), device=M.DEV))
        groups = [{'params': list(self.head.parameters()) + [self.logscale], 'lr': a.lr}]
        if a.ft: groups.append({'params': [p for p in self.hub.parameters() if p.requires_grad], 'lr': a.lr * 0.02})
        self.opt = torch.optim.AdamW(groups, weight_decay=1e-2)
        self.fps_mult = 2 if a.mel else 1

    def embed(self, wav):
        w = M.norm_wav(wav)
        H = M.hubert_layers(self.hub, w, grad=bool(self.ft) and self.head.training)
        Mel = M.mel_input(wav, self.head.cfg) if self.a.mel else None
        S = M.skip_mel(wav, self.a.melskip) if self.a.melskip else None
        return self.head(H, Mel, S)

    def batch_loss(self, A, B, jf, valid, sigma=0.0):
        bA = torch.from_numpy(np.asarray(A, np.float32)).to(M.DEV)
        bB = torch.from_numpy(np.asarray(B, np.float32)).to(M.DEV)
        inv = [inverse(j, v) for j, v in zip(jf, valid)]
        it = np.stack([x[0] for x in inv]); vi = np.stack([x[1] for x in inv])
        m = self.fps_mult
        EA, EB = self.embed(bA), self.embed(bB)
        T = min(EA.shape[1], EB.shape[1]); EA, EB = EA[:, :T], EB[:, :T]
        def up(t, v):   # 50 fps truth -> output rate
            if m == 1: return t[:, :T], v[:, :T]
            t2 = np.repeat(t, 2, 1) * 2 + np.tile([0.0, 1.0], t.shape[1])[None]     # frame 2i+1 is 10 ms later
            return t2[:, :T], np.repeat(v, 2, 1)[:, :T]
        tA, vA = up(jf, valid); tB, vB = up(it, vi)
        tA = torch.from_numpy(np.asarray(tA, np.float32)).to(M.DEV); vA = torch.from_numpy(vA).to(M.DEV)
        tB = torch.from_numpy(np.asarray(tB, np.float32)).to(M.DEV); vB = torch.from_numpy(vB).to(M.DEV)
        scale = self.logscale.exp().clamp(max=100)
        W = 50 * m
        sg = sigma * m
        l1, h1 = nce(EA, EB, tA.float(), vA & (tA <= T - 1), scale, W, sg)
        l2, h2 = nce(EB, EA, tB.float(), vB & (tB <= T - 1), scale, W, sg)
        return (l1 + l2) / 2, (h1 + h2) / 2, EA, EB

    def save(self, path, extra):
        sd = {'head': self.head.state_dict(), 'cfg': self.head.cfg, 'logscale': float(self.logscale.detach()), 'args': vars(self.a), **extra}
        if self.ft:
            sd['hub_layers'] = {i: self.hub.encoder.layers[i].state_dict() for i in range(M.L_HI - self.ft, M.L_HI)}
        torch.save(sd, path)


def val_dtw(EA, EB, jf, valid, fps):
    import fdtw
    errs = []
    for ea, eb, j, v in zip(EA, EB, jf, valid):
        p = np.array(fdtw.band_dtw(np.ascontiguousarray(ea), np.ascontiguousarray(eb), int(fps)))
        cnt = np.bincount(p[:, 0], minlength=len(ea)).astype(float)
        dm = np.bincount(p[:, 0], p[:, 1].astype(float), minlength=len(ea)) / np.maximum(cnt, 1)
        n = min(len(dm), len(j))
        errs.append(np.abs(dm[:n] - j[:n])[v[:n]] / fps * 1000)
    e = np.concatenate(errs)
    return float(np.mean(e)), float(np.median(e)), float(np.mean(e <= 20) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('--pairs', default='tr1'); ap.add_argument('--val', default='val1')
    ap.add_argument('--steps', type=int, default=6000); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--mel', type=int, default=0); ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--dim', type=int, default=128); ap.add_argument('--ft', type=int, default=0)
    ap.add_argument('--tau', type=float, default=0.07); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--mix', default='')
    ap.add_argument('--unison', default='1.0'); ap.add_argument('--melhi', type=int, default=0); ap.add_argument('--melskip', type=int, default=0)    # sampling weight of unison pairs vs others
    ap.add_argument('--init', default=''); ap.add_argument('--snap', type=int, default=1)   # save NAMEs<k>k.pt every 1000 steps
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(f'{TR}/ckpt', exist_ok=True)
    pairs = [load_pairs(p) for p in a.pairs.split(',')]
    VA, VB, Vjf, Vv, _, _ = load_pairs(a.val)
    wts = np.array([float(w) for w in a.mix.split(',')]) if a.mix else np.ones(len(pairs))
    wts = wts / wts.sum()
    uw = [float(u) for u in a.unison.split(',')]; uw = uw * len(pairs) if len(uw) == 1 else uw   # per-set unison weight
    tr = Trainer(a)
    if a.init:
        sd = torch.load(f'{TR}/ckpt/{a.init}.pt', map_location='cpu', weights_only=False)
        tr.head.load_state_dict(sd['head'], strict=False); tr.logscale.data.fill_(sd['logscale'])
    nparam = sum(p.numel() for p in tr.head.parameters())
    print(f'head params {nparam/1e6:.2f} M', flush=True)
    best = 1e9; t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        lr_f = min(1.0, step / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps)))
        for g, base in zip(tr.opt.param_groups, [a.lr, a.lr * 0.02]): g['lr'] = base * lr_f
        tr.head.train()
        si = rng.choice(len(pairs), p=wts); P = pairs[si]
        pw = np.where(P[5], uw[si], 1.0); pw = pw / pw.sum()
        idx = np.sort(rng.choice(len(P[0]), a.bs, replace=False, p=pw))
        loss, hit, _, _ = tr.batch_loss(P[0][idx], P[1][idx], P[2][idx], P[3][idx], P[4])
        tr.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(tr.head.parameters()), 5.0)
        tr.opt.step()
        if step % 100 == 0:
            print(f'step {step} loss {loss.item():.3f} hit {hit.item():.3f} scale {tr.logscale.exp().item():.1f} '
                  f'skip {float(torch.nn.functional.softplus(tr.head.skip_w)) if tr.head.melskip else 0:.2f} lw {np.round(torch.softmax(tr.head.lw, 0).detach().cpu().numpy(), 2).tolist()} {time.time() - t0:.0f}s', flush=True)
        if step % 1000 == 0 or step == a.steps:
            tr.head.eval(); ls, hs, ea_all, eb_all = [], [], [], []
            with torch.no_grad():
                for s in range(0, len(VA), 25):
                    l, h, EA, EB = tr.batch_loss(VA[s:s + 25], VB[s:s + 25], Vjf[s:s + 25], Vv[s:s + 25])
                    ls.append(l.item()); hs.append(h.item())
                    ea_all += list(EA.cpu().numpy()); eb_all += list(EB.cpu().numpy())
            fps = 50 * tr.fps_mult
            m = tr.fps_mult
            jf_o = np.repeat(Vjf, m, 1) * m + (np.tile([0.0, 1.0], Vjf.shape[1])[None] if m == 2 else 0)
            v_o = np.repeat(Vv, m, 1)
            dm = val_dtw(ea_all, eb_all, jf_o, v_o, fps)
            vl = float(np.mean(ls))
            print(f'VAL step {step} loss {vl:.3f} hit {np.mean(hs):.3f} dtw mean {dm[0]:.1f} median {dm[1]:.1f} <=20 {dm[2]:.1f}%', flush=True)
            hist.append(dict(step=step, loss=vl, hit=float(np.mean(hs)), dtw=dm))
            tr.save(f'{TR}/ckpt/{a.name}_last.pt', {'hist': hist})
            if a.snap: tr.save(f'{TR}/ckpt/{a.name}s{step // 1000}k.pt', {'hist': hist})
            if dm[0] < best:
                best = dm[0]; tr.save(f'{TR}/ckpt/{a.name}.pt', {'hist': hist})
    print('done best val dtw mean', best)


if __name__ == '__main__':
    main()
